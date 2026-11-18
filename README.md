# Card Pre-Grader

Offline trading-card pre-grading on your own machine: flattens a photo of a card,
measures centering to sub-pixel precision, checks it against PSA / BGS / CGC
centering limits, flags edge and corner whitening, and exports a visual report.
No network access, no paid APIs.

## Phone version (no computer needed)

Open **https://schtwick.github.io/card-pregrader/** in Safari (iPhone) or Chrome (Android).
The first visit downloads about 75 MB, so use Wi-Fi; after that it works offline. Tap
Share, then **Add to Home Screen** for an app icon. All grading happens on the phone and
photos are never uploaded.

The phone app is `web/` (the same `pregrader` engine running in Pyodide). After changing
the engine, run `.venv/bin/python web/build.py`, commit, and publish the web folder with
`git subtree split --prefix web -b gh-pages && git push -f origin gh-pages`; bump
`VERSION` in `web/sw.js` so phones pick up the update.

## Install

You need Python 3.9+ **with Tk 8.6 or newer**. Apple's built-in `/usr/bin/python3`
ships Tk 8.5, which draws blank windows on current macOS, so don't use it for the
GUI. On this Mac a standalone Python 3.12 (Tk 9.0) was installed with `uv`:

```bash
cd ~/card-pregrader
python3 -m pip install --user uv          # skip if you already have uv / python.org Python
python3 -m uv python install 3.12
$(python3 -m uv python find 3.12) -m venv .venv
.venv/bin/pip install -r requirements.txt  # opencv-python, numpy, pillow, pillow-heif
.venv/bin/pip install pytest               # optional, for the tests
```

(A python.org installer or `brew install python-tk` also works.)

## Run

```bash
./pregrade                                  # GUI
./pregrade gui front.jpg --back back.jpg    # GUI with images loaded
./pregrade analyze front.jpg --back back.jpg -o report.png   # automatic, no GUI
```

`analyze` writes `report.png` plus `report.json` with every measurement
(add `--save-warped` to keep the flattened cards, `--sensitivity 1.5` for a
more sensitive whitening check).

## Reading the result

The top of the report (and the GUI panel) gives a **verdict** for a PSA 10:

- **Good PSA 10 candidate**: centering clears the limits *including* measurement
  error, no edge whitening, back measured, photo quality OK.
- **Borderline**: something the photo can't settle (centering within error of a limit,
  possible marks on a glitter/foil border, missing back, poor photo). Check the listed
  items by eye before paying to submit.
- **Unlikely to get a PSA 10**: centering misses even allowing for error, or clear
  whitening.

Each centering target shows **PASS / BORDERLINE / FAIL**. Ratios carry a ± (1 SD),
built from line-placement error plus how much the ratio changes along the card
(a skewed print or warped photo widens it). PASS means it clears the limit at +2 SD.

Whitening boxes: **red** = whitening, **yellow** = on a glitter/foil border, verify by
eye (doesn't lower the estimate), **orange** = long strip, the outer line is probably
on the background.

## Taking photos

- The whole card in frame, flat, on a plain background that contrasts with the
  border (dark cloth for light/colored borders, grey for black borders).
- Even, diffuse light, no glare. Shoot straight down if you can; the
  perspective is corrected, but glare and blur aren't.
- iPhone HEIC photos open directly.
- The app checks every photo and warns when it's too dark, unevenly lit, blurry,
  has glare, is low resolution or has a low-contrast background. Retake on a "bad".
- The higher the resolution the better. The card is flattened at roughly native
  resolution (up to 3000 px on the long side).
- Out of a toploader or sleeve if possible, since plastic edges confuse edge detection.

## Using the GUI

**Centering mode** (default). Green lines = physical card edge, magenta lines =
where the border meets the artwork. The yellow dashed cross is the card centre;
the magenta dot is the artwork centre.

| Action | How |
| --- | --- |
| Move a line | click within ~10 px of it and drag (quarter-pixel snapping) |
| Select a line | `Tab` / `Shift+Tab`, or keys `1`–`8` |
| Nudge | arrow keys by the *Nudge px* step (0.1–5 px); `Shift` = ×10 |
| Pixel-level check | the **Loupe** follows the mouse, or the selected line when you use the keyboard |
| Re-run auto borders | **Auto inner borders** |

Ratios, PASS/FAIL against the four targets, and the grade ceiling update live.

**Card corners mode.** Use it if auto-detection missed or you want to adjust the
flattening. Drag the 4 handles to where the straight edges *would* meet (ignore
the rounded corner), then press **Apply warp**.

**Scan edges** reruns the whitening check (it also reruns automatically when you
move an outer line). Red boxes = whitening/silvering; orange = a long bright strip
along an edge, which usually means the outer line is sitting on a light background.

**Export report…** saves the PNG + JSON and opens the PNG.

## How it works

1. **Detection**: several independent segmentations (distance from background
   colour, Otsu on brightness and saturation, Canny edges) are scored for
   "card-like" contours. Straight lines are fitted to the middle of each side
   and intersected, so rounded corners don't bias the result. Each side is then
   re-fitted at full resolution from sub-pixel edge profiles.
2. **Flattening**: perspective warp to the standard 63.5 × 88.9 mm aspect with a
   3 mm margin so the true edge stays visible and adjustable.
3. **Inner border**: per side, the median colour across the middle 60 % of the
   card; the border ends where the colour first departs from the border's own
   colour, snapped to the sharpest step there.
4. **Ratios**: `left / (left + right)` etc. Limits are compared on whole-percent
   ratios (55.4/44.6 counts as 55/45).
5. **Whitening**: inside a rounded-rectangle card mask, pixels in a 0.6 mm band
   along the edge that are much lighter and more neutral than the local border
   colour (sampled just further in) are grouped into spots. The threshold rises with
   the local speckle of the border (glitter/foil/holo), and a spot must touch the
   edge and sit mostly in its outer half, since real wear starts at the edge. Works
   on any coloured border, including yellow. It **can't** judge white or silver
   borders, and says so.
6. **Uncertainty**: both opposite borders are re-measured at 6 positions along the
   card; the spread of those ratios plus a 0.75 px line error gives the ±.
7. **Photo quality**: brightness, per-side lighting balance, glare, edge sharpness
   (step size / steepest slope across the edge), resolution, background contrast.

## Grading thresholds

All thresholds are in `pregrader/standards.py`. The rows you specified are marked
`"spec"`; everything else (lower grades, CGC Gem Mint 10) is a commonly cited
approximation marked `"approx"`. Check those against current published standards
before relying on them. The defect → grade caps (`DEFECT_CAPS`) are judgement
calls, not official rules.

## Limits

This is a pre-screen, not a grade. It has been validated on synthetic cards with
known answers and checked by eye on one real photo. Before trusting it to pick
submissions, run it on a few cards whose grades you already know. It doesn't judge surface scratches, print
lines, dents, or how sharp the corners really are, and the whitening check only
sees what the photo shows.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Synthetic cards with known geometry (several border colours and backgrounds,
rotations, perspective, whitening nicks) check corner accuracy (< 1.5 px),
border widths (< 0.05 mm), centering (< 0.5 pt), spot detection and location,
the grading logic, rotation round-trips and report rendering.
