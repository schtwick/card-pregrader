"""Overlay drawing and the side-by-side PNG report."""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import defects, findings, standards
from .centering import Lines, ratio_text
from .pipeline import Analysis, Side, to_json

OUTER_BGR = (60, 220, 60)
INNER_BGR = (255, 60, 255)
GRID_BGR = (0, 230, 255)

_FONTS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]


_BUNDLED = os.path.join(os.path.dirname(__file__), "fonts")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    bundled = os.path.join(_BUNDLED, "Inter-Bold.ttf" if bold else "Inter-Regular.ttf")
    if os.path.exists(bundled):  # same look on every platform, incl. the phone version
        try:
            return ImageFont.truetype(bundled, size)
        except OSError:
            pass
    for p in _FONTS:
        try:
            if p.endswith(".ttc"):
                return ImageFont.truetype(p, size, index=1 if bold else 0)
            if bold:
                bp = p.replace("Arial.ttf", "Arial Bold.ttf").replace("arial.ttf", "arialbd.ttf") \
                      .replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
                if os.path.exists(bp):
                    return ImageFont.truetype(bp, size)
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _dashed(img, p0, p1, color, t, dash):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = max(1, int(np.hypot(*(p1 - p0)) // dash))
    for i in range(0, n, 2):
        a = p0 + (p1 - p0) * i / n
        b = p0 + (p1 - p0) * min(i + 1, n) / n
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, t, cv2.LINE_AA)


def overlay(side: Side, grid: bool = True, show_defects: bool = True) -> np.ndarray:
    if side.borderless:
        grid = False
    """Full-resolution card image with border lines, centre grid and defect boxes."""
    img = side.img
    if show_defects and side.scan is not None:
        img = defects.annotate(img, side.scan, side.px_per_mm)
    else:
        img = img.copy()
    H, W = img.shape[:2]
    ln: Lines = side.lines
    o, i = ln.outer, ln.inner
    t = max(1, int(round(W / 700)))
    for v in (o.left, o.right):
        cv2.line(img, (int(round(v)), 0), (int(round(v)), H), OUTER_BGR, t, cv2.LINE_AA)
    for v in (o.top, o.bottom):
        cv2.line(img, (0, int(round(v))), (W, int(round(v))), OUTER_BGR, t, cv2.LINE_AA)
    if not side.borderless:
        for v in (i.left, i.right):
            cv2.line(img, (int(round(v)), 0), (int(round(v)), H), INNER_BGR, t, cv2.LINE_AA)
        for v in (i.top, i.bottom):
            cv2.line(img, (0, int(round(v))), (W, int(round(v))), INNER_BGR, t, cv2.LINE_AA)
    if grid:
        dash = max(6, W // 80)
        cx, cy = (o.left + o.right) / 2, (o.top + o.bottom) / 2
        ax, ay = (i.left + i.right) / 2, (i.top + i.bottom) / 2
        _dashed(img, (cx, o.top), (cx, o.bottom), GRID_BGR, t, dash)
        _dashed(img, (o.left, cy), (o.right, cy), GRID_BGR, t, dash)
        _dashed(img, (ax, i.top), (ax, i.bottom), INNER_BGR, t, dash)
        _dashed(img, (i.left, ay), (i.right, ay), INNER_BGR, t, dash)
        r = max(4, W // 120)
        cv2.circle(img, (int(cx), int(cy)), r, GRID_BGR, t, cv2.LINE_AA)
        cv2.circle(img, (int(ax), int(ay)), r, INNER_BGR, -1, cv2.LINE_AA)
    return img


def _pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _fit(bgr: np.ndarray, w: int, h: Optional[int] = None) -> Image.Image:
    ih, iw = bgr.shape[:2]
    s = w / iw if h is None else min(w / iw, h / ih)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return _pil(cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=interp))


BG = (247, 247, 245)
INK = (28, 28, 30)
MUTED = (110, 110, 115)
GOOD = (24, 140, 70)
BAD = (200, 40, 40)
WARN = (200, 120, 0)


def _side_column(side: Side, width: int) -> Image.Image:
    f_h, f_s = font(26, True), font(18)
    ov = overlay(side)
    card = _fit(ov, width, 1100)
    annotated = defects.annotate(side.img, side.scan, side.px_per_mm) if side.scan else side.img
    tiles = defects.corner_crops(annotated, side.lines.outer, side.px_per_mm, size_mm=5.0)
    tile = (width - 16) // 2
    strips = defects.edge_strips(annotated, side.lines.outer, side.px_per_mm)
    strip_imgs = []
    for k, v in strips.items():  # each edge in two halves -> twice the zoom
        half = v.shape[1] // 2
        strip_imgs.append((f"{k} edge (1st half)", _fit(v[:, :half + 20], width)))
        strip_imgs.append((f"{k} edge (2nd half)", _fit(v[:, half - 20:], width)))
    c = side.centering
    h = 44 + card.height + 40 + 2 * tile + 16 + 34 + sum(s.height + 30 for _, s in strip_imgs)
    col = Image.new("RGB", (width, h), BG)
    d = ImageDraw.Draw(col)
    head = "no border - centering judged by eye" if side.borderless else f"L/R {ratio_text(c.lr)}   T/B {ratio_text(c.tb)}"
    d.text((0, 4), f"{side.name.upper()}   {head}", font=f_h, fill=INK)
    y = 44
    col.paste(card, ((width - card.width) // 2, y))
    # border widths printed on the card image
    s = card.width / side.img.shape[1]
    ox = (width - card.width) // 2
    o = side.lines.outer
    lab = font(17, True)
    for txt, (px, py), right_align in () if side.borderless else (
        (f"{c.mm('left'):.2f}mm", (o.left + 4 / s, (o.top + o.bottom) / 2), False),
        (f"{c.mm('right'):.2f}mm", (o.right - 4 / s, (o.top + o.bottom) / 2 + 30 / s), True),
        (f"{c.mm('top'):.2f}mm", ((o.left + o.right) / 2 + 12 / s, o.top + 2 / s), False),
        (f"{c.mm('bottom'):.2f}mm", ((o.left + o.right) / 2 + 12 / s, o.bottom - 24 / s), False),
    ):
        x0, y0 = ox + px * s, y + py * s
        bb = d.textbbox((0, 0), txt, font=lab)
        if right_align:
            x0 -= bb[2]
        d.rectangle((x0 - 3, y0 - 2, x0 + bb[2] + 3, y0 + bb[3] + 3), fill=(0, 0, 0))
        d.text((x0, y0), txt, font=lab, fill=(255, 255, 255))
    y += card.height + 10
    d.text((0, y), "Corners (zoomed)", font=font(20, True), fill=INK)
    y += 30
    order = ["top-left", "top-right", "bottom-left", "bottom-right"]
    for k, name in enumerate(order):
        im = _fit(tiles[name], tile, tile)
        x0 = (k % 2) * (tile + 16)
        y0 = y + (k // 2) * (tile + 16)
        col.paste(im, (x0, y0))
        d.rectangle((x0, y0, x0 + im.width - 1, y0 + im.height - 1), outline=(255, 255, 255), width=3)
        d.rectangle((x0, y0, x0 + 150, y0 + 26), fill=(0, 0, 0))
        d.text((x0 + 6, y0 + 3), name, font=f_s, fill=(255, 255, 255))
    y += 2 * tile + 16 + 4
    d.text((0, y), "Edges (card edge at top of each strip)", font=font(20, True), fill=INK)
    y += 30
    for name, im in strip_imgs:
        d.text((0, y), name, font=f_s, fill=MUTED)
        col.paste(im, (0, y + 22))
        y += im.height + 30
    return col


def _text_panel(a: Analysis, width: int) -> Image.Image:
    lines: List[Tuple[str, ImageFont.ImageFont, tuple, int]] = []  # text, font, colour, indent
    H1, H2, B, S = font(34, True), font(23, True), font(19), font(16)

    def add(t, f=B, c=INK, ind=0):
        lines.append((t, f, c, ind))

    def gap(n=10):
        lines.append(("", None, INK, n))

    add("Card Pre-Grade Report", H1)
    add(_dt.datetime.now().strftime("%Y-%m-%d %H:%M"), S, MUTED)
    for s in a.sides.values():
        if s.path:
            add(f"{s.name}: {os.path.basename(s.path)}", S, MUTED)
    gap(18)

    if a.verdict:
        vcol = {"send": GOOD, "maybe": WARN, "skip": BAD}[a.verdict.level]
        for chunk in _wrap(a.verdict.headline, 40):
            add(chunk, font(27, True), vcol)
        for r in a.verdict.reasons:
            for k, chunk in enumerate(_wrap(r, 70)):
                add(("- " if k == 0 else "  ") + chunk, S, INK, 8)
        gap(18)

    add("Findings in plain words", H2)
    kcol = {findings.GOOD: GOOD, findings.PROBLEM: BAD, findings.CHECK: WARN, findings.INFO: MUTED}
    topic = None
    for it in findings.build(a):
        if it.topic == "Overall":  # already shown as the headline above
            continue
        if it.topic != topic:
            topic = it.topic
            gap(6)
            add(topic, font(19, True), INK)
        tag = findings.LABEL[it.kind]
        for k, chunk in enumerate(_wrap(it.text, 66)):
            if k == 0:
                lines.append((f"{tag:<8}", font(15, True), kcol[it.kind], 8))
                lines.append(("__SAME_LINE__" + chunk, S, INK, 92))
            else:
                add(chunk, S, INK, 92)
    gap(18)

    add("Centering", H2)
    for s in a.sides.values():
        c = s.centering
        add(f"{s.name.title()}", font(20, True))
        if s.borderless:
            add("No printed border - not measured (judge by eye)", B, WARN, 16)
            continue
        add(f"Left/Right  {ratio_text(c.lr)}   ({c.lr[0]:.1f} / {c.lr[1]:.1f}  ±{c.sigma_lr:.1f})", B, INK, 16)
        add(f"   L {c.mm('left'):.2f} mm   R {c.mm('right'):.2f} mm", S, MUTED, 16)
        add(f"Top/Bottom  {ratio_text(c.tb)}   ({c.tb[0]:.1f} / {c.tb[1]:.1f}  ±{c.sigma_tb:.1f})", B, INK, 16)
        add(f"   T {c.mm('top'):.2f} mm   B {c.mm('bottom'):.2f} mm", S, MUTED, 16)
    gap(18)

    add("Centering vs. targets", H2)
    for ch in a.checks:
        tag, col = {"pass": ("PASS", GOOD), "borderline": ("BORDERLINE", WARN),
                    "fail": ("FAIL", BAD)}.get(ch.status, ("n/a", MUTED))
        add(f"{tag}   {ch.name}", font(20, True), col)
        parts = []
        if ch.front_value is not None:
            parts.append(f"front {ch.front_value:.1f} (max {ch.front_limit:.0f}/{100 - ch.front_limit:.0f})")
        if ch.back_value is not None:
            parts.append(f"back {ch.back_value:.1f} (max {ch.back_limit:.0f}/{100 - ch.back_limit:.0f})")
        add("   " + "   ".join(parts), S, MUTED, 16)
    gap(18)

    add("Edge & corner whitening", H2)
    sev_col = {"none": GOOD, "minor": WARN, "moderate": BAD, "heavy": BAD}
    for s in a.sides.values():
        sc = s.scan
        add(f"{s.name.title()}: {sc.severity}  ({len(sc.whitening)} spot(s))", font(20, True), sev_col[sc.severity])
        for sp in sc.spots[:8]:
            kind = {"whitening": "", "verify": "  [glitter border: verify by eye]"}.get(sp.kind, "  [check outer line]")
            add(f"- {sp.location}: {sp.area_mm2:.2f} mm²{kind}", S, INK, 16)
        if len(sc.spots) > 8:
            add(f"... and {len(sc.spots) - 8} more", S, MUTED, 16)
    gap(18)

    if a.predictions:
        add("Predicted grade", H2)
        add("from centering and edges - surface not checked", S, MUTED)
        ccol = {"high": GOOD, "medium": WARN, "low": BAD}
        for pr in a.predictions:
            if not pr.available:
                add(f"{pr.company}  n/a", font(28, True), MUTED)
                add("   nothing measurable in this photo", S, MUTED, 16)
                continue
            if not pr.has_odds:
                add(f"{pr.company}  up to {pr.label}", font(28, True))
                add(f"   {pr.basis} - centering not included", S, WARN, 16)
                continue
            add(f"{pr.company}  {pr.label}   {pr.probability:.0%}", font(28, True))
            others = [f"{lab} {p:.0%}" for lab, p in pr.distribution if lab != pr.label]
            if others:
                add("   otherwise " + ", ".join(others), S, MUTED, 16)
            if pr.if_marks_real:
                add(f"   if the yellow-boxed marks are real wear: {pr.if_marks_real}", S, WARN, 16)
        pr = a.predictions[0]
        add(f"Confidence: {pr.confidence}", font(19, True), ccol[pr.confidence])
        for chunk in _wrap(", ".join(pr.confidence_why), 70) if pr.confidence_why else []:
            add("   " + chunk, S, MUTED, 0)
        gap(18)

    add("Best possible grade (ceiling)", H2)
    for ce in a.ceilings:
        add(f"{ce.company}  {ce.label}", font(28, True))
        add(f"   limited by: {ce.limited_by}", S, MUTED, 16)
    gap(18)

    if a.notes:
        add("Notes", H2)
        for n in a.notes:
            for chunk in _wrap(n, 62):
                add(chunk, S, WARN, 0)
        gap(12)
    for chunk in _wrap("Pre-grade estimate from image analysis only. Surface scratches, print lines, "
                       "dents and true corner sharpness are not judged. ± is one standard deviation "
                       "of the ratio. Approximate lower-grade thresholds: see pregrader/standards.py.", 62):
        add(chunk, S, MUTED, 0)

    h = 20
    for t, f, _, ind in lines:
        if f is not None and t.startswith("__SAME_LINE__"):
            continue
        h += ind if f is None else (f.size + 10 if hasattr(f, "size") else 26)
    panel = Image.new("RGB", (width, h + 20), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    y = 20
    prev_y = y
    for t, f, c, ind in lines:
        if f is None:
            y += ind
            continue
        if t.startswith("__SAME_LINE__"):
            d.text((24 + ind, prev_y), t[len("__SAME_LINE__"):], font=f, fill=c)
            continue
        d.text((24 + ind, y), t, font=f, fill=c)
        prev_y = y
        y += (f.size + 10) if hasattr(f, "size") else 26
    return panel


def _wrap(text: str, n: int) -> List[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def render(a: Analysis, col_width: int = 760, panel_width: int = 800) -> Image.Image:
    cols = [_side_column(s, col_width) for s in a.sides.values()]
    panel = _text_panel(a, panel_width)
    gap = 28
    W = gap + sum(c.width + gap for c in cols) + panel.width + gap
    H = gap + max([panel.height] + [c.height for c in cols]) + gap
    out = Image.new("RGB", (W, H), BG)
    x = gap
    for c in cols:
        out.paste(c, (x, gap))
        x += c.width + gap
    out.paste(panel, (x, gap))
    return out


def save(a: Analysis, path: str) -> Tuple[str, str]:
    """Write the PNG report and a JSON of all measurements next to it."""
    img = render(a)
    img.save(path)
    jpath = os.path.splitext(path)[0] + ".json"
    items = findings.build(a)
    data = to_json(a)
    data["findings"] = [vars(it) for it in items]
    with open(jpath, "w") as fh:
        json.dump(data, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.splitext(path)[0] + ".txt", "w") as fh:
        fh.write("CARD PRE-GRADE REPORT\n")
        for s in a.sides.values():
            if s.path:
                fh.write(f"{s.name}: {os.path.basename(s.path)}\n")
        fh.write("\n" + findings.as_text(a, items))
    return path, jpath
