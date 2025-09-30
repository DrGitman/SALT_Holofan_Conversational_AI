import argparse
import os
import re
from pathlib import Path

HEADING_RE = {
    1: re.compile(r"^\s*#\s*(.+)$"),
    2: re.compile(r"^\s*##\s*(.+)$"),
    3: re.compile(r"^\s*###\s*(.+)$"),
}

slug_re = re.compile(r"[^A-Za-z0-9]+")

def slugify(text: str) -> str:
    text = slug_re.sub("-", text).strip("-")
    return (text or "entry").lower()[:80]


def split_markdown(src: Path, out_dir: Path, level: int = 1) -> int:
    if level not in HEADING_RE:
        raise ValueError("level must be 1, 2, or 3")
    pattern = HEADING_RE[level]

    out_dir.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    entries = []
    current_title = None
    current_lines = []

    def flush():
        nonlocal current_title, current_lines
        if current_title and current_lines:
            entries.append((current_title, current_lines))
        current_title = None
        current_lines = []

    for ln in lines:
        m = pattern.match(ln)
        if m:
            # New entry
            flush()
            current_title = m.group(1).strip()
            current_lines = [ln]
        else:
            if current_lines is not None:
                current_lines.append(ln)

    flush()

    # If no headings at given level, try lower level automatically once
    if not entries and level < 3:
        return split_markdown(src, out_dir, level + 1)

    # Final fallback: split by explicit pattern even if not matched in loop
    if not entries:
        raw = "\n".join(lines)
        parts = re.split(rf"^\s*{'#'*level}\s*", raw, flags=re.MULTILINE)
        rebuilt = []
        if len(parts) > 1:
            for p in parts[1:]:
                first_line, *rest = p.splitlines()
                title = first_line.strip()
                body = "\n".join(rest)
                rebuilt.append((title, [f"{'#'*level} {title}", body]))
        if rebuilt:
            entries = rebuilt

    count = 0
    for idx, (title, content_lines) in enumerate(entries, 1):
        name = f"{idx:02d}-{slugify(title)}.md"
        dest = out_dir / name
        with dest.open("w", encoding="utf-8") as w:
            w.write("\n".join(content_lines).strip() + "\n")
        count += 1

    return count


def main():
    ap = argparse.ArgumentParser(description="Split a Markdown file into multiple files by heading level.")
    ap.add_argument("--src", required=True, help="Path to source .md file")
    ap.add_argument("--out-dir", required=True, help="Output directory for split files")
    ap.add_argument("--level", type=int, default=1, help="Heading level to split on (1|2|3). Default 1")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    count = split_markdown(src, out_dir, level=args.level)
    print(f"Created {count} files in {out_dir}")


if __name__ == "__main__":
    main()
