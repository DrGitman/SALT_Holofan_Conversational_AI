import argparse
import re
from pathlib import Path

slug_re = re.compile(r"[^A-Za-z0-9]+")


def slugify(s: str) -> str:
    return slug_re.sub("-", (s or "").strip()).strip('-').lower()[:60]


def parse_args():
    ap = argparse.ArgumentParser(description="Split a Markdown file into entries starting with '1.', '2.', etc.")
    ap.add_argument("--src", required=True, help="Source markdown file")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out_dir = Path(args.out_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    text = src.read_text(encoding="utf-8")
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split on lines that begin with a number + dot (e.g., "1.", "2.")
    # Keep the delimiter by using a capturing group
    parts = re.split(r"(^\s*\d+\.\s*)", text, flags=re.MULTILINE)

    # parts structure: [before, delim1, body1, delim2, body2, ...]
    entries = []
    i = 1
    for k in range(1, len(parts), 2):
        delim = parts[k]
        body = parts[k + 1] if k + 1 < len(parts) else ""
        # Combine delim + body for full entry text
        full = (delim + body).strip()
        if not full:
            continue
        # Derive a short title from the first sentence/words
        first_line = full.splitlines()[0].strip()
        # Remove the leading number "1." etc.
        first_line = re.sub(r"^\s*\d+\.\s*", "", first_line)
        title_words = first_line.split()
        title = " ".join(title_words[:8]) or f"entry-{i}"
        entries.append((title, full))
        i += 1

    if not entries:
        print("No numbered entries found (lines starting with '1.', '2.', ...). Nothing created.")
        return

    count = 0
    for idx, (title, content) in enumerate(entries, 1):
        name = f"{idx:02d}-{slugify(title)}.md"
        dest = out_dir / name
        # Ensure each file has at least an H2 heading with the title for clarity
        if not content.lstrip().startswith("## "):
            content = f"## {title}\n\n" + content
        dest.write_text(content.strip() + "\n", encoding="utf-8")
        count += 1

    print(f"Created {count} files in {out_dir}")


if __name__ == "__main__":
    main()
