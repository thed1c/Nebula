#!/usr/bin/env python3
"""The constellation GIFs from greenfield/constellation_demo.sh's frames: each frame fitted to one size, a caption bar
under it (whose view it is, and what happens), the map's frames played in time and each step held.

    python3 greenfield/gifs.py <frames dir> <out dir>
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, BAR = 720, 760, 74
INK, MUT, GROUND, ACCENT = (236, 230, 255), (155, 147, 184), (11, 9, 24), (255, 179, 71)
FONTS = ["/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
BODY = ["/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(paths, size):
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def wrap(draw, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + [cur] if cur else lines


def frame(src: Path, who: str, caption: str) -> Image.Image:
    im = Image.open(src).convert("RGB")
    im.thumbnail((W, H - BAR), Image.LANCZOS)
    out = Image.new("RGB", (W, H), GROUND)
    out.paste(im, ((W - im.width) // 2, (H - BAR - im.height) // 2))
    d = ImageDraw.Draw(out)
    d.rectangle([0, H - BAR, W, H], fill=(18, 15, 36))
    d.line([0, H - BAR, W, H - BAR], fill=ACCENT, width=2)
    tag, body = font(FONTS, 15), font(BODY, 17)
    d.text((18, H - BAR + 10), f"{who.upper()}'S NEBULA", font=tag, fill=ACCENT)
    for i, line in enumerate(wrap(d, caption, body, W - 36)[:2]):
        d.text((18, H - BAR + 30 + i * 21), line, font=body, fill=INK)
    return out


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    frames = json.loads((src / "frames.json").read_text())["frames"]
    for gif in sorted({f["gif"] for f in frames}):
        part = [f for f in frames if f["gif"] == gif]
        ims = [frame(src / f["file"], f["who"], f["caption"]) for f in part]
        ms = [f["ms"] if f["ms"] < 1800 else 2600 for f in part]            # each step held long enough to read
        out = dst / f"constellation-{gif}.gif"
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
        print(out, len(ims), "frames", f"{out.stat().st_size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
