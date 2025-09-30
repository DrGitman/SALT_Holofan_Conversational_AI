import OBSWebSocket from "obs-websocket-js";

const obs = new OBSWebSocket();

(async () => {
  await obs.connect("ws://localhost:4455", "your_password");
  console.log("Connected to OBS");

  // Example: switch to a scene called "TalkingScene"
  await obs.call("SetCurrentProgramScene", { sceneName: "TalkingScene" });
})();
