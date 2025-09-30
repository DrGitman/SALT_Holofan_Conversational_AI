const API_BASE = process.env.REACT_APP_API_BASE || 'http://127.0.0.1:8000';

export async function speak(text) {
  if (!text || !text.trim()) return;
  const resp = await fetch(`${API_BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`TTS failed: ${resp.status} ${err}`);
  }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  await new Promise((resolve, reject) => {
    audio.onended = resolve;
    audio.onerror = reject;
    audio.play().catch(reject);
  });
  URL.revokeObjectURL(url);
}
