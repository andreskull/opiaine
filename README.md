# Opiaine - Interactive Lecture Companion

Opiaine is an innovative EdTech solution developed during the EdTech Estonia hackathon to make lectures more engaging and interactive. The platform bridges the gap between traditional lecture formats and modern students' engagement needs by introducing real-time interaction and gamification elements.

## 🎯 Problem Statement

Modern students, raised in a highly engaging digital environment with constant streams of content, often struggle to maintain focus during traditional 90-minute lectures. Opiaine addresses this challenge by transforming passive lecture attendance into an interactive learning experience.

## 🚀 Features

- **Real-time Lecture Transcription**  
  Automatically transcribes lectures as they happen, making content immediately accessible to students

- **Interactive Q&A Sessions**  
  Lecturers can initiate AI-generated test questions based on just-covered material

- **Instant Feedback**
  - Students receive immediate AI-powered scoring of their answers
  - Lecturers get real-time insights into student comprehension and participation

- **Gamification**  
  Encourages student engagement through performance tracking and recognition

- **Analytics Dashboard**  
  Helps lecturers identify areas of their presentation that need improvement

## 🛠 Technology Stack

- **Backend**: Python FastAPI
- **Frontend**: React
- **AI Components**:
  - OpenAI Whisper for Speech-to-Text
  - GPT-4 for Question Generation and Answer Grading
- **Real-time Communication**: WebSocket

## 🏗 Project Structure

```
opiaine/
├── frontend/          # React frontend application
│   ├── public/
│   └── src/
├── lectures/          # Lecture audio storage
├── main.py           # FastAPI backend server
├── requirements.txt  # Python dependencies
└── .env             # Environment variables
```

## 🚦 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/andreskull/opiaine.git
cd opiaine
```

2. Set up the backend:

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env-sample .env
# Edit .env with your OpenAI API key
```

3. Set up the frontend:

```bash
cd frontend
npm install
```

4. Run the application:

```bash
# Terminal 1 - Backend
# The FastAPI server will start at http://127.0.0.1:8000

python main.py

# Terminal 2 - Frontend
cd frontend
npm start # React will start at http://localhost:3000
```

## 🔑 Environment Variables

Create a `.env` file in the root directory with:

```
OPENAI_API_KEY=your_api_key_here
```

## 👥 Target Users

- **Lecturers**  
  Who want to make their lectures more engaging and receive real-time feedback

- **Students**  
  Who benefit from interactive learning and immediate content reinforcement

- **Educational Institutions**  
  Looking to modernize their lecture delivery methods

## 🎮 How It Works

1. Lecturer starts recording their lecture through the app
2. Students join the lecture session via their devices
3. As the lecture progresses:
   - Content is transcribed in real-time
   - Lecturer can pause at any time to initiate Q&A sessions
   - AI generates relevant questions based on the recent content
   - Students receive and answer questions
   - Both students and lecturers get immediate feedback
4. Analytics help lecturers improve their teaching methods

## 🤝 Contributing

This project was developed during a hackathon and is open for contributions. Feel free to submit issues and pull requests.

## 📝 License

[MIT License](LICENSE)

---

*Developed during EdTech Estonia Hackathon 2024*
