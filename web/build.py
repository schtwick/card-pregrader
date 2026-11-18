"""Package the grader for the web version: web/py/pregrader.zip + app icons.

    .venv/bin/python web/build.py
"""
import os
import zipfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "pregrader")
SKIP = {"gui.py", "__main__.py", "icon.png"}  # desktop-only

os.makedirs(os.path.join(HERE, "py"), exist_ok=True)
out = os.path.join(HERE, "py", "pregrader.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for dirpath, _, names in os.walk(PKG):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if n in SKIP or n.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, n)
            z.write(full, os.path.relpath(full, ROOT))
print(out, os.path.getsize(out), "bytes")

icon = Image.open(os.path.join(PKG, "icon.png")).convert("RGBA")
for size in (180, 512):
    bg = Image.new("RGBA", icon.size, (28, 30, 38, 255))   # iOS shows transparency as black
    bg.alpha_composite(icon)
    bg.convert("RGB").resize((size, size), Image.LANCZOS).save(os.path.join(HERE, f"icon-{size}.png"))
print("icons written")
