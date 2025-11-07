SALT Holofan Conversational AI — Setup Guide

An interactive AI-powered chatbot with speech-to-text, text-to-speech, and retrieval over local documents.
Frontend built with React + TailwindCSS, backend with FastAPI + LangChain + Google Gemini, and TTS via ElevenLabs.

Features

- Voice input via browser speech recognition

- Chat log with user + AI messages

- Realistic voice replies with ElevenLabs TTS

- Retrieval-Augmented Generation (RAG) with local documents

- Animated avatar with Lottie

- Requirements

Node.js >=18

Python >=3.10

A Google Gemini API key

An ElevenLabs API key

Installation
1. Clone Repository
git clone https://github.com/your-username/ai-elder-chat.git
cd ai-elder-chat

2. Backend Setup
cd backend
python -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows

pip install -r requirements.txt


Create a .env file inside backend/:

GOOGLE_API_KEY=your_gemini_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key


Run the backend:

uvicorn main:app --reload --port 8000


Backend will be live at:

http://127.0.0.1:8000

3. Frontend Setup
cd ../frontend
npm install


Run the frontend:

npm start


Frontend will be live at:

http://localhost:3000

Usage

Start backend (uvicorn) and frontend (npm start).

Click the mic button to start talking.

The AI elder will:

Transcribe your speech

Generate a reply via Gemini

Speak back using ElevenLabs

Project Structure
ai-elder-chat/
│── backend/           # FastAPI + LangChain server
│   ├── main.py        # API entrypoint
│   ├── rag.py         # RAG logic (Gemini + embeddings)
│   ├── tts.py         # ElevenLabs integration
│   └── .env           # API keys
│
│── frontend/          # React app
│   ├── src/
│   │   ├── components/
│   │   │   ├── Avatar.jsx
│   │   │   ├── ChatLog.jsx
│   │   │   └── SpeechControls.jsx
│   │   ├── App.jsx
│   │   └── tts.js
│   └── public/
│
└── README.md

Troubleshooting
- 500 ELEVENLABS_API_KEY not configured → Ensure .env in backend/ has your key.
- CORS error → Check backend has CORSMiddleware allowing http://localhost:3000.
- Model not found → Ensure correct Gemini model name in backend (gemini-1.5-flash).
- TTS SSL errors → Sometimes ElevenLabs rate-limits free accounts; retry later.
- 500 ELEVENLABS_API_KEY not configured → Ensure .env in backend/ has your key.
- CORS error → Check backend has CORSMiddleware allowing http://localhost:3000.
- Model not found → Ensure correct Gemini model name in backend (gemini-1.5-flash).
- TTS SSL errors → Sometimes ElevenLabs rate-limits free accounts; retry later.

