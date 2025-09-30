import React from "react";
import Lottie from "lottie-react";
import idleAnimation from "../assets/animations/avatar_idle.json";
import listeningAnimation from "../assets/animations/avatar_listening.json";
import speakingAnimation from "../assets/animations/avatar_speaking.json";

export default function Avatar({ listening, speaking }) {
  let animationData = idleAnimation;
  if (listening) animationData = listeningAnimation;
  else if (speaking) animationData = speakingAnimation;

  return (
    <div className="flex items-center justify-center w-48 h-48 md:w-56 md:h-56 bg-gray-700 rounded-full shadow-lg">
      <Lottie animationData={animationData} loop />
    </div>
  );
}
