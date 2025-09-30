import React from "react";
import { Mic, MicOff } from "lucide-react";

export default function SpeechControls({ listening, onStart, onStop }) {
  return (
    <div className="flex justify-center">
      <button
        onClick={listening ? onStop : onStart}
        className={`p-5 rounded-full shadow-xl transition-all duration-300 focus:outline-none ${
          listening
            ? "bg-red-500 animate-pulse scale-110"
            : "bg-blue-500 hover:scale-110"
        }`}
      >
        {listening ? <MicOff size={28} /> : <Mic size={28} />}
      </button>
    </div>
  );
}
