"""The gallery's contact sheet (VEH3f audit) -- one tile per gallery frame,
labelled with the player's OWN note for it (class, row, label, body), never a
list this script keeps.

    python tools/demo/contact_sheet.py <out-dir> <sheet.png>

Reads `<out-dir>/hero.csv` for the `INF_PIE_GALLERY k/n ...` notes and
`<out-dir>/2NN-veh3f-gallery-*.png` for the frames. Exit 0 = written; 2 = no
frames.
"""

import glob
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    out_dir, sheet = sys.argv[1], sys.argv[2]
    notes = {}
    try:
        with open(os.path.join(out_dir, "hero.csv"), encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r"INF_PIE_GALLERY (\d+)/(\d+) (.*)", line)
                if m:
                    notes[int(m.group(1))] = m.group(3).strip()
    except OSError:
        pass
    frames = sorted(glob.glob(os.path.join(out_dir, "2[0-9][0-9]-veh3f-gallery-*.png")))
    frames = [f for f in frames if "contact-sheet" not in f]
    if not frames:
        print("no gallery frames")
        return 2
    tile_w, tile_h, band = 480, 270, 40
    cols = 5
    rows = (len(frames) + cols - 1) // cols
    img = Image.new("RGB", (cols * tile_w, rows * (tile_h + band)), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for i, path in enumerate(frames):
        k = int(re.search(r"gallery-(\d+)-", os.path.basename(path)).group(1))
        shot = Image.open(path).convert("RGB")
        w, h = shot.size
        # Drop the window's title bar (the top 3 %), keep the game.
        shot = shot.crop((0, int(h * 0.03), w, h)).resize((tile_w, tile_h))
        x, y = (i % cols) * tile_w, (i // cols) * (tile_h + band)
        img.paste(shot, (x, y + band))
        note = notes.get(k, "(no note)")
        fields = dict(re.findall(r"(class|row|label|body)=(\S+)", note))
        label_m = re.search(r"label=(.*?) body=", note)
        label = label_m.group(1) if label_m else fields.get("label", "")
        line1 = f"{k}. {fields.get('class', '?')} -- {label}"
        line2 = f"{fields.get('row', '?')} [{fields.get('body', '?')}]"
        draw.text((x + 6, y + 3), line1, fill=(240, 240, 240), font=font)
        draw.text((x + 6, y + 21), line2, fill=(170, 200, 255), font=font)
    img.save(sheet)
    print(f"sheet {sheet}: {len(frames)} tiles, {len(notes)} notes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
