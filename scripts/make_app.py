"""Build ~/Applications/Card Pre-Grader.app (a launcher for this project) with an icon.

    .venv/bin/python scripts/make_app.py
"""
from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.expanduser("~/Applications/Card Pre-Grader.app")
ICON_PNG = os.path.join(ROOT, "pregrader", "icon.png")


def draw_icon(size: int = 1024) -> Image.Image:
    s = size / 1024
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # macOS-style rounded square background
    d.rounded_rectangle((60 * s, 60 * s, 964 * s, 964 * s), radius=200 * s, fill=(28, 30, 38, 255))
    # card
    cw, ch = 470 * s, 658 * s
    x0, y0 = (size - cw) / 2, (size - ch) / 2
    d.rounded_rectangle((x0, y0, x0 + cw, y0 + ch), radius=34 * s, fill=(222, 178, 64, 255))
    # artwork window, slightly off-centre
    ix0, iy0 = x0 + 46 * s, y0 + 42 * s
    ix1, iy1 = x0 + cw - 34 * s, y0 + ch - 42 * s
    d.rectangle((ix0, iy0, ix1, iy1), fill=(52, 92, 168, 255))
    d.rectangle((ix0, iy0 + (iy1 - iy0) * 0.62, ix1, iy1), fill=(236, 238, 242, 255))
    # centring guide lines
    green, pink = (70, 230, 90, 255), (255, 70, 230, 255)
    w = max(2, int(10 * s))
    for x in (x0, x0 + cw):
        d.line((x, y0 - 50 * s, x, y0 + ch + 50 * s), fill=green, width=w)
    for x in (ix0, ix1):
        d.line((x, y0 - 50 * s, x, y0 + ch + 50 * s), fill=pink, width=w)
    # magnifier
    cx, cy, r = 700 * s, 690 * s, 120 * s
    d.line((cx + r * 0.7, cy + r * 0.7, cx + r * 1.6, cy + r * 1.6), fill=(240, 240, 240, 255), width=int(46 * s))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(240, 240, 240, 255), width=int(30 * s),
              fill=(255, 255, 255, 60))
    return im


def build_icns(png: Image.Image, out: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "icon.iconset")
        os.makedirs(iconset)
        for base in (16, 32, 128, 256, 512):
            png.resize((base, base), Image.LANCZOS).save(os.path.join(iconset, f"icon_{base}x{base}.png"))
            png.resize((base * 2, base * 2), Image.LANCZOS).save(os.path.join(iconset, f"icon_{base}x{base}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)


def main() -> None:
    icon = draw_icon()
    icon.resize((512, 512), Image.LANCZOS).save(ICON_PNG)  # also used for the Dock icon while running
    if os.path.exists(APP):
        shutil.rmtree(APP)
    macos = os.path.join(APP, "Contents", "MacOS")
    res = os.path.join(APP, "Contents", "Resources")
    os.makedirs(macos)
    os.makedirs(res)
    build_icns(icon, os.path.join(res, "AppIcon.icns"))
    launcher = os.path.join(macos, "Card Pre-Grader")
    with open(launcher, "w") as fh:
        fh.write(f'#!/bin/sh\ncd "{ROOT}"\nexec "{ROOT}/.venv/bin/python" -m pregrader gui "$@"\n')
    os.chmod(launcher, 0o755)
    with open(os.path.join(APP, "Contents", "Info.plist"), "wb") as fh:
        plistlib.dump({
            "CFBundleName": "Card Pre-Grader",
            "CFBundleDisplayName": "Card Pre-Grader",
            "CFBundleIdentifier": "local.cardpregrader",
            "CFBundleVersion": "1.0",
            "CFBundleShortVersionString": "1.0",
            "CFBundlePackageType": "APPL",
            "CFBundleExecutable": "Card Pre-Grader",
            "CFBundleIconFile": "AppIcon",
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
        }, fh)
    subprocess.run(["touch", APP])
    print(APP)


if __name__ == "__main__":
    main()
