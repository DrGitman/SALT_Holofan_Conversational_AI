import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

slug_re = re.compile(r"[^A-Za-z0-9]+")
heading_re = re.compile(r"^\s*#{1,6}\s+(.+)$")


def slugify(s: str) -> str:
    return slug_re.sub("-", (s or "").strip()).strip('-').lower()[:60]


def parse_args():
    ap = argparse.ArgumentParser(description="Split motivational stories by Speaker marker")
    ap.add_argument("--src", default=str(ROOT / "data" / "motivational stories.md"), help="Source markdown file")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "motivational-stories"), help="Output directory")
    ap.add_argument(
        "--speaker-pattern",
        default=r"^\s*(Speaker|By|Author|Narrator)\s*[:\-]\s*(.+)$",
        help="Regex to detect speaker lines; default matches 'Speaker: Name', 'By - Name', etc.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    lines = src.read_text(encoding="utf-8").splitlines()
    speaker_re = re.compile(args.speaker_pattern, re.IGNORECASE)

    entries = []
    current = {"speaker": None, "title": None, "buf": []}

    def flush():
        nonlocal current
        if current["buf"]:
            sp = (current["speaker"] or "speaker").strip()
            tl = (current["title"] or "story").strip()
            entries.append({"speaker": sp, "title": tl, "buf": current["buf"][:]})
        current = {"speaker": None, "title": None, "buf": []}

    for ln in lines:
        m = speaker_re.match(ln)
        if m:
            flush()
            current["speaker"] = (m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)).strip()
            current["buf"].append(ln)
            continue
        if current["speaker"] and not current["title"]:
            mh = heading_re.match(ln)
            if mh:
                current["title"] = mh.group(1).strip()
        current["buf"].append(ln)

    flush()

    # If no explicit speaker markers found, fallback to split by top-level headings (# or ##)
    if not any(e.get("speaker") for e in entries) or len(entries) <= 1:
        # Reuse generic heading splitter behavior
        content = "\n".join(lines)
        parts = re.split(r"^\s*#\s+", content, flags=re.MULTILINE)
        rebuilt = []
        if len(parts) > 1:
            # First part before first heading is discardable
            for p in parts[1:]:
                first_line, *rest = p.splitlines()
                title = first_line.strip()
                body = "\n".join(rest)
                rebuilt.append({"speaker": "speaker", "title": title, "buf": [f"# {title}", body]})
        entries = rebuilt or entries

    if not entries:
        raise SystemExit("No entries found. Provide a 'Speaker:' line or headings (# Title) per story, or set --speaker-pattern.")

    count = 0
    for i, e in enumerate(entries, 1):
        speaker_slug = slugify(e["speaker"]) or f"speaker-{i:02d}"
        title_slug = slugify(e["title"]) or "story"
        name = f"{i:02d}-{speaker_slug}-{title_slug}.md"
        dest = out_dir / name
        dest.write_text("\n".join([ln for ln in e["buf"] if ln is not None]).strip() + "\n", encoding="utf-8")
        count += 1

    print(f"Created {count} files in {out_dir}")


if __name__ == "__main__":
    main()
