import React from "react";

export default function ChatLog({ messages }) {
  return (
    <div className="space-y-3">
      {messages.map((msg, idx) => (
        <div
          key={idx}
          className={`px-4 py-3 rounded-2xl text-sm shadow-md transition-all animate-fadeIn ${
            msg.from === "user"
              ? "bg-blue-500 ml-auto text-right text-white"
              : "bg-gray-700 mr-auto text-left text-gray-100"
          }`}
        >
          {msg.text}
        </div>
      ))}
    </div>
  );
}
