import React, { useState, useEffect, useRef } from "react";
import Avatar from "./components/Avatar";
import ChatLog from "./components/ChatLog";
import SpeechControls from "./components/SpeechControls";
import axios from "axios";
import { speak as ttsSpeak } from "./lib/tts";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const recognitionRef = useRef(null);
  const silenceTimer = useRef(null);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.error("Speech recognition not supported in this browser");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join("");
      handleStopListening();
      handleSend(transcript);
    };

    recognition.onspeechend = () => handleStopListening();
    recognitionRef.current = recognition;
  }, []);

  function handleStartListening() {
    if (recognitionRef.current) {
      recognitionRef.current.start();
      setListening(true);

      silenceTimer.current = setTimeout(() => {
        handleStopListening();
      }, 3000);
    }
  }

  function handleStopListening() {
    if (recognitionRef.current) recognitionRef.current.stop();
    clearTimeout(silenceTimer.current);
    setListening(false);
  }

  async function handleSend(userText) {
    if (!userText.trim()) return;

    setMessages((prev) => [...prev, { from: "user", text: userText }]);

    try {
      const res = await axios.get(`${API_BASE}/ask`, {
        params: { question: userText },
      });

      console.log("API response:", res.status, res.data);
      const payload = res.data || {};
      const botReply = payload.answer || payload.result || payload.error || "";

      if (botReply) {
        setMessages((prev) => [...prev, { from: "bot", text: botReply }]);
        try {
          setSpeaking(true);
          await ttsSpeak(botReply);
        } catch (e) {
          console.error("TTS error:", e);
        } finally {
          setSpeaking(false);
        }
      } else {
        setMessages((prev) => [
          ...prev,
          { from: "bot", text: "I heard you, but I could not form a reply." },
        ]);
      }
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prev) => [
        ...prev,
        { from: "bot", text: "Network problem speaking back. Please try again." },
      ]);
    }
  }


  return (
    <div className="flex flex-col items-center justify-between h-screen bg-gradient-to-b from-gray-900 to-black text-white px-4 py-6">
      <Avatar listening={listening} speaking={speaking} />

      <div className="flex-1 w-full max-w-lg overflow-y-auto space-y-3 p-4 bg-gray-800 rounded-2xl shadow-inner">
        <ChatLog messages={messages} />
      </div>

      <SpeechControls
        listening={listening}
        onStart={handleStartListening}
        onStop={handleStopListening}
      />
    </div>
  );
}
