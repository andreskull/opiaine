from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Dict, Set, Optional
import random
import string
import uvicorn
import json
import websockets
import asyncio
from pathlib import Path
import binascii
import soundfile as sf
import sounddevice as sd
import whisper
import numpy as np
import librosa
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(
        "OpenAI API key not found! Please set the OPENAI_API_KEY environment variable. "
        "You can get your API key from https://platform.openai.com/account/api-keys"
    )

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Session storage
session_questions = {}  # Store questions for each session
session_transcripts = {}  # Store transcripts for each session
session_stats = {}  # Store Q&A statistics for each session
session_sections = {}  # {session_code: [LectureSection]}
playback_active = {}  # Track playback state for each session
audio_positions = {}  # {session_code: current_position_in_samples}

app = FastAPI()

# Initialize Whisper model at startup
model = whisper.load_model("base")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SectionManager:
    def __init__(self):
        self._current_section = {}  # Private dict storage
    
    def get(self, session_code: str, default: int = 1) -> int:
        """Get current section number for session"""
        if not isinstance(self._current_section, dict):
            print(f"Warning: Resetting corrupted current_section from {type(self._current_section)} to dict")
            self._current_section = {}
        return self._current_section.get(session_code, default)
    
    def set(self, session_code: str, value: int):
        """Set current section number for session"""
        if not isinstance(value, int):
            raise ValueError(f"Section number must be int, not {type(value)}")
        if not isinstance(self._current_section, dict):
            self._current_section = {}
        self._current_section[session_code] = value
        
    def remove(self, session_code: str):
        """Remove session from tracking"""
        if session_code in self._current_section:
            del self._current_section[session_code]

# Replace the global current_section with an instance
section_manager = SectionManager()

# Create lectures directory
Path("lectures").mkdir(exist_ok=True)

# Add a specific endpoint for fetching the test lecture
@app.get("/lectures/{filename}")
async def get_lecture_file(filename: str):
    file_path = Path("lectures") / filename
    if not file_path.exists():
        return {"error": "File not found"}
    return FileResponse(
        file_path,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
        }
    )

class SessionManager:
    def __init__(self):
        self.sessions = {}  # {session_code: set(student_websockets)}
        self.lecturer_sessions = {}  # {session_code: lecturer_websocket}
        print("SessionManager initialized")

    async def create_lecturer_session(self, websocket: WebSocket):
        """Create a new session for a lecturer"""
        session_code = self.generate_session_code()
        print(f"Generated session code: {session_code}")
        self.sessions[session_code] = set()  # Set of student WebSockets
        self.lecturer_sessions[session_code] = websocket
        print(f"Session {session_code} created. Active sessions: {list(self.sessions.keys())}")
        return session_code

    def generate_session_code(self) -> str:
        """Generate a unique session code"""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    
    async def add_student(self, session_code: str, websocket: WebSocket) -> bool:
        """Add a student to a session"""
        if session_code in self.sessions:
            self.sessions[session_code].add(websocket)
            count = len(self.sessions[session_code])
            print(f"Sending student count update: {count} for session {session_code}")
            
            # Send count update to lecturer
            if session_code in self.lecturer_sessions:
                lecturer_ws = self.lecturer_sessions[session_code]
                try:
                    await lecturer_ws.send_json({
                        "type": "student_count",
                        "count": count
                    })
                    print(f"Student count update sent to lecturer: {count}")
                except Exception as e:
                    print(f"Error sending student count: {e}")
            return True
        return False
    
    async def remove_student(self, session_code: str, websocket: WebSocket):
        """Remove a student from a session"""
        if session_code in self.sessions and websocket in self.sessions[session_code]:
            self.sessions[session_code].remove(websocket)
            # Send updated student count to lecturer
            count = len(self.sessions[session_code])
            print(f"Sending student count update: {count} for session {session_code}")
            if session_code in self.lecturer_sessions:
                try:
                    await self.lecturer_sessions[session_code].send_json({
                        "type": "student_count",
                        "count": count
                    })
                except Exception as e:
                    print(f"Error sending student count: {e}")
    
    async def end_session(self, session_code: str):
        """End a session and clean up all related data"""
        try:
            print(f"Ending session: {session_code}")
            
            # Clean up student connections
            if session_code in self.sessions:
                # Notify all students that session has ended
                for student_ws in self.sessions[session_code]:
                    try:
                        await student_ws.send_json({"type": "session_ended"})
                    except:
                        pass  # Handle disconnected students
                del self.sessions[session_code]
            
            # Clean up lecturer connection
            if session_code in self.lecturer_sessions:
                del self.lecturer_sessions[session_code]
            
            # Clean up session data
            if session_code in session_sections:
                del session_sections[session_code]
            
            if session_code in session_questions:
                del session_questions[session_code]
            
            if session_code in session_transcripts:
                del session_transcripts[session_code]
            
            if session_code in session_stats:
                del session_stats[session_code]
            
            if session_code in playback_active:
                del playback_active[session_code]
            
            if session_code in audio_positions:
                del audio_positions[session_code]
            
            # Clean up current section tracking
            section_manager.remove(session_code)
            
            print(f"Session {session_code} ended. Active sessions: {list(self.sessions.keys())}")
            
        except Exception as e:
            print(f"Error during session cleanup: {e}")
            import traceback
            print(traceback.format_exc())
    
    async def update_student_count(self, session_code: str):
        """Send updated student count to lecturer"""
        if session_code in self.lecturer_sessions:
            lecturer_ws = self.lecturer_sessions[session_code]
            try:
                count = len(self.sessions[session_code])
                print(f"Sending student count update: {count} for session {session_code}")  # Debug
                await lecturer_ws.send_json({
                    "type": "student_count",
                    "count": count
                })
            except Exception as e:
                print(f"Error updating student count: {str(e)}")  # Log the actual error
    
    def get_session_student_count(self, session_code: str) -> int:
        """Get number of students in a session"""
        return len(self.sessions.get(session_code, set()))
    
    def session_exists(self, session_code: str) -> bool:
        """Check if a session exists"""
        exists = session_code in self.sessions
        print(f"=== Session Check ===")
        print(f"Checking session: {session_code}")
        print(f"Session exists: {exists}")
        print(f"All active sessions: {list(self.sessions.keys())}")
        print(f"Sessions dict: {self.sessions}")
        return exists
    
    async def cleanup_dead_connections(self, session_code: str):
        """Remove any dead connections from the session"""
        if session_code in self.sessions:
            dead_connections = set()
            for ws in self.sessions[session_code]:
                if ws.client_state == websockets.protocol.State.CLOSED:
                    dead_connections.add(ws)
            
            for ws in dead_connections:
                self.sessions[session_code].discard(ws)
            
            await self.update_student_count(session_code)

    async def connect_lecturer(self, session_code: str, websocket: WebSocket):
        """Connect lecturer and handle reconnection"""
        try:
            # Handle existing connection
            if session_code in self.lecturer_sessions:
                try:
                    await self.lecturer_sessions[session_code].close()
                except:
                    pass
                    
            self.lecturer_sessions[session_code] = websocket
            print(f"Lecturer connected to session: {session_code}")
            
        except Exception as e:
            print(f"Error connecting lecturer: {e}")
            raise

    async def disconnect_lecturer(self, session_code: str):
        """Disconnect lecturer and handle reconnection"""
        try:
            # Handle existing connection
            if session_code in self.lecturer_sessions:
                try:
                    await self.lecturer_sessions[session_code].close()
                except:
                    pass
                    
            del self.lecturer_sessions[session_code]
            print(f"Lecturer disconnected from session: {session_code}")
            
        except Exception as e:
            print(f"Error disconnecting lecturer: {e}")
            raise

# Create global session manager - make it clearly global
global_session_manager = SessionManager()

async def generate_questions(transcript):
    """Generate questions based on lecture transcript"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Generate 3 questions based on the following lecture transcript. "
                          "Format as JSON array with 'id' and 'text' for each question."
            }, {
                "role": "user",
                "content": transcript
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

async def grade_answer(session_code: str, question_id: str, question: str, answer: str, transcript: str):
    """Grade student answer using OpenAI"""
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "Grade the following answer (0-100) and provide constructive feedback. "
                              "Format as JSON with 'score' and 'comments' fields."
                }, {
                    "role": "user",
                    "content": f"Lecture context: {transcript}\n\n"
                              f"Question: {question}\n"
                              f"Student answer: {answer}"
                }]
            )
        )
        
        feedback_text = response.choices[0].message.content
        print(f"Raw feedback: {feedback_text}")
        
        try:
            feedback = json.loads(feedback_text)
            return feedback
        except json.JSONDecodeError as je:
            print(f"JSON parsing error: {je}")
            # Return a properly formatted feedback object
            return {
                "score": 0,
                "comments": "Error processing feedback. Please try again."
            }
            
    except Exception as e:
        print(f"Error grading answer: {e}")
        return {
            "score": 0,
            "comments": f"Error grading answer: {str(e)}"
        }

# Add at the top with other global variables
playback_active = {}  # Track playback state for each session
current_chunk_position = {}  # Track current position for each session

class LectureSection:
    def __init__(self, transcript: str, section_number: int):
        self.section_number = section_number
        self.transcript = transcript
        self.question = None  # Will be set when Q&A starts
        self.answers = []     # List of student answers
        self.stats = {
            'total_answers': 0,
            'average_score': 0.0,
            'answers': []  # List of {student_id, answer, score, feedback}
        }

    def set_question(self, question: dict):
        self.question = question

# Initialize session data
async def initialize_session_data(session_code: str):
    """Initialize all session-related data structures"""
    print(f"\n=== Initializing Session Data ===")
    print(f"Session: {session_code}")
    
    # Initialize session data
    session_sections[session_code] = []
    section_manager.set(session_code, 1)  # Ensure we set it as a number
    session_questions[session_code] = {}
    session_transcripts[session_code] = ""
    session_stats[session_code] = {
        'total_answers': 0,
        'average_score': 0.0,
        'questions': {}
    }
    playback_active[session_code] = False
    
    # Create first section
    first_section = LectureSection("", 1)
    session_sections[session_code].append(first_section)
    
    print(f"Session initialized with first section")
    print(f"Current section number: {section_manager.get(session_code)}")

# Update your session cleanup
async def cleanup_session(session_code: str):
    """Clean up session data when a session ends"""
    if session_code in session_sections:
        del session_sections[session_code]
    if session_code in section_manager.get(session_code):
        section_manager.remove(session_code)
    if session_code in session_questions:
        del session_questions[session_code]
    if session_code in session_transcripts:
        del session_transcripts[session_code]
    if session_code in session_stats:
        del session_stats[session_code]

# Helper functions for section management
async def start_new_section(session_code: str) -> LectureSection:
    """Start a new section for the given session"""
    try:
        print(f"\n=== Starting New Section ===")
        print(f"Session: {session_code}")
        
        # Get next section number
        current_num = section_manager.get(session_code, 1)
        next_section_num = current_num + 1
        print(f"Current section: {current_num}, Creating new section: {next_section_num}")
        
        # Create new section
        new_section = LectureSection("", next_section_num)
        
        # Add to sections list
        if session_code not in session_sections:
            session_sections[session_code] = []
        session_sections[session_code].append(new_section)
        
        # Update current section number
        section_manager.set(session_code, next_section_num)
        
        print(f"New section created and section number updated: {section_manager.get(session_code)}")
        return new_section
        
    except Exception as e:
        print(f"Error in start_new_section: {e}")
        import traceback
        print(traceback.format_exc())
        raise

async def get_current_section(session_code: str) -> Optional[LectureSection]:
    """Get the current lecture section"""
    print(f"Getting current section for {session_code}")
    print(f"Available sections: {session_sections}")
    if session_code in session_sections and session_sections[session_code]:
        current_idx = section_manager.get(session_code) - 1
        print(f"Current index: {current_idx}")
        if current_idx < len(session_sections[session_code]):
            return session_sections[session_code][current_idx]
    print("No section found!")
    return None

async def update_section_transcript(session_code: str, new_text: str):
    """Add text to current section's transcript"""
    if session_code in session_sections and session_sections[session_code]:
        current_idx = section_manager.get(session_code) - 1
        if current_idx < len(session_sections[session_code]):
            session_sections[session_code][current_idx].transcript += " " + new_text

@app.websocket("/ws/lecturer")
async def lecturer_endpoint(websocket: WebSocket):
    print("\n=== Debug: lecturer_endpoint start ===")
    
    await websocket.accept()
    session_code = None
    
    try:
        session_code = await global_session_manager.create_lecturer_session(websocket)
        print(f"Created session with code: {session_code}")
        
        # Initialize session data
        session_sections[session_code] = []
        section_manager.set(session_code, 1)  # Initialize first section number
        print(f"Initialized session data for {session_code}")
        
        await websocket.send_json({
            "type": "session_created",
            "code": session_code
        })
        
        print(f"Lecturer connected. Session: {session_code}")
        
        while True:
            data = await websocket.receive_json()
            
            # Handle heartbeat with minimal logging
            if data["type"] == "heartbeat":
                await websocket.send_json({"type": "heartbeat_ack"})
                continue  # Skip the rest of the logging for heartbeats
            
            # Full logging for non-heartbeat messages
            print("\n=== WebSocket Message Received ===")
            print(f"Message type: {data.get('type')}")
            print(f"Session code: {session_code}")
            print(f"Current playback active: {playback_active.get(session_code, False)}")
            
            if data["type"] == "stop_playback":
                print("\n=== Stop Playback Command Received ===")
                if session_code in current_chunk_position:
                    pos = current_chunk_position[session_code]
                    audio_positions[session_code] = pos
                    print(f"Storing position: {pos} samples ({pos/sample_rate:.2f} seconds)")
                sd.stop()
                playback_active[session_code] = False
                await websocket.send_json({
                    "type": "playback_stopped"
                })
            
            elif data["type"] == "start_qa_session":
                print("\n=== Starting Q&A Session ===")
                print("1. Ensuring playback is stopped")
                sd.stop()
                playback_active[session_code] = False
                
                print("2. Processing transcript")
                current_section = await get_current_section(session_code)
                print(f"Current section: {current_section}")
                print(f"Section number: {current_section.section_number if current_section else 'None'}")
                print(f"Section transcript: {current_section.transcript if current_section else 'None'}")
                
                if current_section:
                    try:
                        print("3. Generating question with OpenAI")
                        response = await asyncio.to_thread(
                            lambda: client.chat.completions.create(
                                model="gpt-4",
                                messages=[{
                                    "role": "system",
                                    "content": "Generate one question based on the lecture transcript."
                                }, {
                                    "role": "user",
                                    "content": current_section.transcript
                                }]
                            )
                        )
                        
                        print("4. OpenAI response received")
                        print(f"Raw response: {response.choices[0].message.content}")
                        
                        question = {
                            "id": f"q{current_section.section_number}",
                            "text": response.choices[0].message.content
                        }
                        print(f"5. Formatted question: {question}")
                        
                        current_section.question = question
                        
                        # Store for the current section
                        if session_code not in session_questions:
                            session_questions[session_code] = {}
                        session_questions[session_code][current_section.section_number] = question
                        
                        print("6. Broadcasting question to students")
                        print(f"Active sessions: {list(global_session_manager.sessions.keys())}")
                        print(f"Current session: {session_code}")
                        print(f"Number of students: {len(global_session_manager.sessions.get(session_code, []))}")
                        
                        if session_code in global_session_manager.sessions:
                            # Create message once
                            message = {
                                "type": "qa_session_started",
                                "section_number": section_manager.get(session_code),
                                "transcript": current_section.transcript,
                                "question": question
                            }
                            
                            # Send to students
                            for student_ws in global_session_manager.sessions[session_code]:
                                try:
                                    print(f"7. Sending to student: {message}")
                                    await student_ws.send_json(message)
                                    print("8. Message sent successfully")
                                except Exception as e:
                                    print(f"Error sending question to student: {e}")
                                    print(traceback.format_exc())
                            
                            # Send to lecturer
                            try:
                                print("9. Sending to lecturer")
                                await websocket.send_json(message)  # websocket here is the lecturer's websocket
                                print("10. Message sent to lecturer successfully")
                            except Exception as e:
                                print(f"Error sending question to lecturer: {e}")
                                print(traceback.format_exc())
                        else:
                            print(f"No students found in session {session_code}")
                        
                        print("Successfully started Q&A session")
                        
                    except Exception as e:
                        print(f"Error in Q&A session: {e}")
                        print("Full error:")
                        import traceback
                        print(traceback.format_exc())
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                else:
                    print("No current section found!")
                    await websocket.send_json({
                        "type": "error",
                        "message": "No current section found"
                    })
            
            elif data["type"] == "play_test_lecture":
                try:
                    file_path = Path("lectures/test_lecture.wav")
                    if file_path.exists():
                        print(f"\n=== Starting Lecture Playback ===")
                        print(f"File: {file_path}")
                        
                        # Set playback active at the start
                        playback_active[session_code] = True
                        
                        # Load audio file
                        audio_data, sample_rate = sf.read(str(file_path))
                        
                        # Define chunk parameters
                        chunk_duration = 5  # seconds
                        chunk_samples = int(chunk_duration * sample_rate)
                        
                        # Get starting position and convert to chunk index
                        current_pos = audio_positions.get(session_code, 0)
                        chunk_start = (current_pos // chunk_samples) * chunk_samples  # Align to chunk boundary
                        
                        print(f"Resuming from position: {current_pos} samples")
                        print(f"Chunk-aligned start: {chunk_start} samples")
                        print(f"Time position: {current_pos/sample_rate:.2f} seconds")
                        
                        # Start from stored position
                        for i in range(chunk_start, len(audio_data), chunk_samples):
                            # Check for new messages without blocking
                            try:
                                data = await asyncio.wait_for(
                                    websocket.receive_json(),
                                    timeout=0.1  # Short timeout to check messages
                                )
                                print("\n=== Received message during playback ===")
                                print(f"Message type: {data.get('type')}")
                                if data.get('type') == 'stop_playback':
                                    print("Stopping playback immediately")
                                    current_chunk_position[session_code] = i
                                    audio_positions[session_code] = i
                                    print(f"Storing position: {i} samples ({i/sample_rate:.2f} seconds)")
                                    sd.stop()
                                    playback_active[session_code] = False
                                    await websocket.send_json({
                                        "type": "playback_stopped",
                                        "position": i
                                    })
                                    break
                            except asyncio.TimeoutError:
                                # No new messages, continue with playback
                                pass
                            
                            if not playback_active.get(session_code, False):
                                # Store current position before stopping
                                audio_positions[session_code] = i
                                print(f"Storing position {i} samples ({i/sample_rate:.2f} seconds) for session {session_code}")
                                break
                                
                            chunk = audio_data[i:i + chunk_samples]
                            
                            try:
                                # Convert to mono if stereo
                                if len(chunk.shape) > 1:
                                    chunk = chunk.mean(axis=1)
                                
                                # Ensure float32 type
                                chunk = chunk.astype(np.float32)
                                print(f"Chunk dtype: {chunk.dtype}, shape: {chunk.shape}")
                                
                                # Transcribe chunk
                                result = model.transcribe(chunk)
                                transcribed_text = result["text"].strip()
                                
                                if transcribed_text:
                                    print(f"Transcribed chunk: {transcribed_text}")
                                    
                                    # Ensure we have a current section
                                    if not session_sections.get(session_code):
                                        print("Creating initial section")
                                        session_sections[session_code] = []
                                        section_manager.set(session_code, 1)  # Use section_manager instead of direct assignment
                                        first_section = LectureSection("", 1)
                                        session_sections[session_code].append(first_section)
                                    
                                    # Update current section transcript
                                    current_idx = section_manager.get(session_code) - 1
                                    if current_idx < len(session_sections[session_code]):
                                        session_sections[session_code][current_idx].transcript += " " + transcribed_text
                                        print(f"Updated section {current_idx + 1} transcript. Current length: "
                                              f"{len(session_sections[session_code][current_idx].transcript)}")
                                    
                                    # Send to lecturer
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "text": transcribed_text,
                                        "section_number": section_manager.get(session_code)
                                    })
                                    
                                    # Broadcast to students
                                    if session_code in global_session_manager.sessions:
                                        for student_ws in global_session_manager.sessions[session_code]:
                                            try:
                                                await student_ws.send_json({
                                                    "type": "transcription",
                                                    "text": transcribed_text,
                                                    "section_number": section_manager.get(session_code)
                                                })
                                            except Exception as e:
                                                print(f"Error sending to student: {e}")
                                
                                # Play the original chunk for audio output
                                sd.play(audio_data[i:i + chunk_samples], sample_rate)
                                sd.wait()  # Wait until chunk is done playing
                                
                                # Check playback state after playing
                                if not playback_active.get(session_code, False):
                                    print("Playback stopped after chunk")
                                    break
                                    
                            except Exception as e:
                                print(f"Error processing chunk: {e}")
                                print(f"Error details: {str(e)}")
                                import traceback
                                print(traceback.format_exc())
                        
                        print("\n=== Playback Finished ===")
                        playback_active[session_code] = False
                        
                    else:
                        print("Test lecture file not found")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Test lecture file not found"
                        })
                        
                except Exception as e:
                    print(f"\n=== Playback Error ===")
                    print(f"Error: {e}")
                    import traceback
                    print(traceback.format_exc())
                    playback_active[session_code] = False
                    
            elif data["type"] == "continue_lecture":
                try:
                    print("\n=== Debug: continue_lecture ===")
                    print(f"Session code: {session_code}")
                    
                    section_num = section_manager.get(session_code)
                    print(f"Current section number: {section_num}")
                    
                    # Continue with the existing logic...
                    playback_active[session_code] = True
                    await handle_playback_resume(session_code)
                    
                except Exception as e:
                    print(f"Error continuing lecture: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            elif data["type"] == "resume_playback":
                try:
                    print("Handling resume playback request")
                    await handle_playback_resume(session_code)
                    await websocket.send_json({
                        "type": "playback_resumed",
                        "success": True
                    })
                except Exception as e:
                    print(f"Error resuming playback: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to resume playback"
                    })
    
    except WebSocketDisconnect:
        print(f"Lecturer disconnected: {session_code}")
    finally:
        if session_code:
            await global_session_manager.end_session(session_code)

@app.websocket("/ws/student/{session_code}")
async def student_endpoint(websocket: WebSocket, session_code: str):
    global global_session_manager
    await websocket.accept()
    
    print(f"Student attempting to connect to session: {session_code}")
    
    # Add retry logic with delay
    for attempt in range(3):
        if global_session_manager.session_exists(session_code):
            break
        print(f"Attempt {attempt + 1}: Waiting for session to be available...")
        await asyncio.sleep(1)
    
    if not global_session_manager.session_exists(session_code):
        print(f"Session not found after retries: {session_code}")
        await websocket.send_json({
            "type": "error",
            "message": "Invalid session code"
        })
        await websocket.close()
        return
    
    try:
        success = await global_session_manager.add_student(session_code, websocket)
        if success:
            print(f"Student successfully connected to session: {session_code}")
            await websocket.send_json({
                "type": "connected",
                "message": "Successfully connected to session"
            })
            
            while True:
                data = await websocket.receive_json()
                
                # Handle heartbeat with minimal logging
                if data["type"] == "heartbeat":
                    await websocket.send_json({"type": "heartbeat_ack"})
                    continue
                
                # Full logging for non-heartbeat messages
                print(f"Student message: {data}")
                
                if data["type"] == "submit_answer":
                    question_id = data["questionId"]
                    answer = data["answer"]
                    print(f"\n=== Processing Student Answer ===")
                    print(f"Question ID: {question_id}")
                    print(f"Answer: {answer}")
                    print(f"Session: {session_code}")
                    
                    try:
                        # Get section from session_sections directly
                        if session_code in session_sections and session_sections[session_code]:
                            # Find the section that matches the question ID
                            section_number = int(question_id[1:])  # Extract number from 'q1', 'q2' etc
                            section = None
                            
                            for s in session_sections[session_code]:
                                if s.section_number == section_number:
                                    section = s
                                    break
                                    
                            if section:
                                print(f"\nSection object details:")
                                print(f"Type: {type(section)}")
                                print(f"Section number: {section.section_number}")
                                print(f"Question: {section.question}")
                                print(f"Transcript preview: {section.transcript[:100]}")
                                
                                if section.question:
                                    # Grade answer
                                    feedback = await grade_answer(
                                        session_code,
                                        question_id,
                                        section.question["text"],
                                        answer,
                                        section.transcript
                                    )
                                    
                                    # Store answer in section
                                    section.answers.append({
                                        "answer": answer,
                                        "feedback": feedback
                                    })
                                    
                                    print(f"\nGenerated feedback: {feedback}")
                                    
                                    # Send feedback to student with section info
                                    print("\nSending feedback to student...")
                                    await websocket.send_json({
                                        "type": "answer_feedback",
                                        "section_number": section.section_number,
                                        "feedback": feedback,
                                        "submitted_answer": answer
                                    })
                                    print("Feedback sent to student")
                                    
                                    # Update lecturer stats
                                    await update_lecturer_stats(session_code, question_id, feedback)
                            else:
                                print(f"Section {section_number} not found")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Section {section_number} not found"
                                })
                        else:
                            print(f"No sections found for session {session_code}")
                            await websocket.send_json({
                                "type": "error",
                                "message": "Section not found"
                            })
                            
                    except Exception as e:
                        print(f"\n=== Error Processing Answer ===")
                        print(f"Error type: {type(e)}")
                        print(f"Error message: {str(e)}")
                        print(f"Session: {session_code}")
                        import traceback
                        print(traceback.format_exc())
                        await websocket.send_json({
                            "type": "error",
                            "message": "Error processing answer"
                        })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "Could not join session"
            })
            await websocket.close()
            
    except WebSocketDisconnect:
        await global_session_manager.remove_student(session_code, websocket)
    except Exception as e:
        print(f"Student error: {str(e)}")
        await global_session_manager.remove_student(session_code, websocket)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Lecture Assistant Backend"}

async def update_lecturer_stats(session_code: str, question_id: str, feedback: dict):
    """Update statistics for the lecturer view"""
    import traceback
    print("\n=== Update Lecturer Stats Called ===")
    print(f"Called from: {traceback.extract_stack()[-2][2]}")  # Print caller function name
    print(f"Session: {session_code}")
    print(f"Question: {question_id}")
    print(f"Feedback: {feedback}")
    try:
        if session_code not in session_stats:
            session_stats[session_code] = {
                'total_answers': 0,
                'average_score': 0.0,
                'questions': {}
            }
        
        stats = session_stats[session_code]
        
        # Initialize question stats if not exists
        if question_id not in stats['questions']:
            stats['questions'][question_id] = {
                'attempts': 0,
                'scores': [],
                'average_score': 0.0
            }
        
        # Update question-specific stats ONCE
        q_stats = stats['questions'][question_id]
        score = float(feedback['score'])
        q_stats['attempts'] += 1
        q_stats['scores'].append(score)
        q_stats['average_score'] = sum(q_stats['scores']) / len(q_stats['scores'])
        
        # Calculate total answers as sum of attempts across all questions
        stats['total_answers'] = sum(q['attempts'] for q in stats['questions'].values())
        
        # Calculate overall average from all scores
        all_scores = []
        for q in stats['questions'].values():
            all_scores.extend(q['scores'])
        stats['average_score'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        print(f"Updated stats for question {question_id} (single update): {stats}")
        
        # Send updated stats to lecturer ONCE
        if session_code in global_session_manager.lecturer_sessions:
            lecturer_ws = global_session_manager.lecturer_sessions[session_code]
            await lecturer_ws.send_json({
                'type': 'qa_stats',
                'stats': stats
            })
            
    except Exception as e:
        print(f"Error updating lecturer stats: {e}")
        import traceback
        print(traceback.format_exc())

async def handle_playback_resume(session_code: str):
    """Handle resuming lecture playback"""
    try:
        print(f"\n=== Resuming Playback ===")
        print(f"Session: {session_code}")
        
        if session_code not in playback_active:
            print(f"No active playback found for session {session_code}")
            return
            
        playback_active[session_code] = True
        
        # Create new section
        current_num = section_manager.get(session_code)
        new_section_num = current_num + 1
        section_manager.set(session_code, new_section_num)
        
        # Create new LectureSection object
        new_section = LectureSection("", new_section_num)
        session_sections[session_code].append(new_section)
        
        # Note: We're not modifying audio_positions here to preserve the position
        print(f"Created new section {new_section_num}")
        print(f"Preserved audio position: {audio_positions.get(session_code, 0)}")
        
        try:
            # Notify lecturer about new section
            if session_code in global_session_manager.lecturer_sessions:
                lecturer_ws = global_session_manager.lecturer_sessions[session_code]
                try:
                    await lecturer_ws.send_json({
                        "type": "new_section_created",
                        "section_number": new_section_num,
                        "message": "New section started"
                    })
                except Exception as e:
                    print(f"Error notifying lecturer: {e}")
            
            # Notify students about playback resume with new section
            if session_code in global_session_manager.sessions:
                for student_ws in global_session_manager.sessions[session_code]:
                    try:
                        await student_ws.send_json({
                            "type": "new_section_created",
                            "section_number": new_section_num,
                            "message": "New section started"
                        })
                    except WebSocketDisconnect:
                        print(f"Student disconnected during resume notification")
                        continue
                    except Exception as e:
                        print(f"Error notifying student: {e}")
                        continue
            
            print(f"Playback resumed for session {session_code} in section {new_section_num}")
            
        except Exception as e:
            print(f"Error during playback resume: {e}")
            playback_active[session_code] = False
            raise
            
    except Exception as e:
        print(f"Error handling playback resume: {e}")
        import traceback
        print(traceback.format_exc())
        raise

# Add this helper function
def validate_audio_position(position: int, audio_length: int) -> int:
    """Ensure audio position is valid"""
    if position < 0:
        return 0
    if position >= audio_length:
        return 0
    return position

if __name__ == "__main__":
    print("\n=== Lecture Assistant Backend ===")
    print("Server starting at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")