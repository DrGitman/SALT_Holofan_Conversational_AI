import os
import re
from pathlib import Path

# Paths (edit if your paths are different)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "wise sayings.md"
OUT_DIR = ROOT / "data" / "wise-sayings"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    raise FileNotFoundError(f"Source file not found: {SRC}")

with SRC.open("r", encoding="utf-8") as f:
    lines = f.read().splitlines()

entries = []
current = {"original": "", "translation": "", "emotion": ""}

for ln in lines:
    s = (ln or "").strip()
    if s.startswith("- Original:"):
        # flush previous
        if current["original"] or current["translation"]:
            entries.append(current)
            current = {"original": "", "translation": "", "emotion": ""}
        current["original"] = s.split(":", 1)[1].strip().strip('"')
    elif s.lower().startswith("translation:"):
        current["translation"] = s.split(":", 1)[1].strip().strip('"')
    elif s.lower().startswith("emotion:"):
        current["emotion"] = s.split(":", 1)[1].strip()

# flush last
if current["original"] or current["translation"]:
    entries.append(current)

if not entries:
    raise SystemExit("No entries found. Ensure the file uses '- Original:', 'Translation:', and 'Emotion:' lines.")

slug_re = re.compile(r"[^A-Za-z0-9]+")

def slugify(text: str) -> str:
    text = slug_re.sub("-", text).strip("-")
    return (text or "saying").lower()[:60]

count = 0
for idx, e in enumerate(entries, 1):
    trans = e["translation"] or e["original"]
    if not trans:
        continue
    name = f"{idx:02d}-{slugify(trans[:30])}.md"
    dest = OUT_DIR / name
    with dest.open("w", encoding="utf-8") as w:
        w.write("# Wise Saying\n\n")
        if e["original"]:
            w.write(f'Original: "{e["original"]}"\n')
        w.write(f'Translation: "{trans}"\n')
        if e["emotion"]:
            w.write(f'Emotion: {e["emotion"]}\n')
    count += 1

print(f"Created {count} files in {OUT_DIR}")
