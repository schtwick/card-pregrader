"""Edge and corner whitening / silvering detector, plus zoom crops."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .centering import Box
from .geometry import CORNER_RADIUS_MM


@dataclass
class Spot:
    x: int
    y: int
    w: int
    h: int
    location: str      # e.g. "top-left corner", "right edge"
    area_mm2: float
    contrast: float    # mean whiteness above the local border colour
    kind: str = "whitening"  # | "verify": on a glitter/foil border, check by eye
    #                          | "check-line": long strip, likely a misplaced outer line

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class DefectScan:
    spots: List[Spot]
    severity: str                      # none | minor | moderate | heavy
    unreliable_sides: List[str] = field(default_factory=list)  # white/silver borders
    textured_sides: List[str] = field(default_factory=list)    # glitter/foil/holo borders
    low_res: bool = False  # photo too coarse to judge whitening: every spot is "verify"

    @property
    def whitening(self) -> List[Spot]:
        return [s for s in self.spots if s.kind == "whitening"]


@dataclass
class ScanParams:
    inset_mm: float = 0.12   # skip this much at the very edge (anti-aliased background)
    depth_mm: float = 0.6    # perimeter band examined for whitening
    delta: float = 32.0      # whiteness rise over local border colour (Lab units)
    min_l: float = 150.0     # whitening must actually be light
    max_chroma: float = 30.0  # ... and close to neutral (white / silver)
    min_area_mm2: float = 0.02
    texture_z: float = 3.0       # on speckled borders, must beat the local speckle by this many SDs
    edge_touch_mm: float = 0.15  # real wear reaches the edge; spots must come this close
    textured_level: float = 12.0  # local SD above which a border counts as textured
    min_px_per_mm: float = 20.0   # below this, spots are shown for checking but not counted


def card_mask(shape: Tuple[int, int], outer: Box, radius_px: float) -> np.ndarray:
    """Filled rounded rectangle at the card outline."""
    m = np.zeros(shape[:2], np.uint8)
    l, t = int(round(outer.left)), int(round(outer.top))
    r, b = int(round(outer.right)), int(round(outer.bottom))
    rad = int(round(max(1.0, min(radius_px, (r - l) / 2, (b - t) / 2))))
    cv2.rectangle(m, (l + rad, t), (r - rad, b), 255, -1)
    cv2.rectangle(m, (l, t + rad), (r, b - rad), 255, -1)
    for cx, cy in ((l + rad, t + rad), (r - rad, t + rad), (r - rad, b - rad), (l + rad, b - rad)):
        cv2.circle(m, (cx, cy), rad, 255, -1, lineType=cv2.LINE_AA)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def _locate(cx: float, cy: float, outer: Box, zone: float) -> str:
    dl, dr = cx - outer.left, outer.right - cx
    dt, db = cy - outer.top, outer.bottom - cy
    v = "top" if dt < zone else "bottom" if db < zone else None
    h = "left" if dl < zone else "right" if dr < zone else None
    if v and h:
        return f"{v}-{h} corner"
    return f"{min((dl, 'left'), (dr, 'right'), (dt, 'top'), (db, 'bottom'))[1]} edge"


def _severity(spots: List[Spot]) -> str:
    real = [s for s in spots if s.kind == "whitening"]
    if not real:
        return "none"
    total = sum(s.area_mm2 for s in real)
    biggest = max(s.area_mm2 for s in real)
    if len(real) <= 2 and total < 0.25 and biggest < 0.15:
        return "minor"
    if len(real) <= 6 and total < 1.5 and biggest < 0.6:
        return "moderate"
    return "heavy"


def scan_edges(img: np.ndarray, outer: Box, px_per_mm: float, sensitivity: float = 1.0,
               params: ScanParams = ScanParams()) -> DefectScan:
    """Flag pixels along the perimeter that are much whiter than the nearby border ink.

    The reference colour is a local average taken a little further in from the edge,
    so this works on any border colour (black, blue, yellow, holo...) but cannot see
    whitening on white or silver borders, which are reported as unreliable."""
    H, W = img.shape[:2]
    ppm = px_per_mm
    radius = CORNER_RADIUS_MM * ppm
    inset = max(1.5, params.inset_mm * ppm)
    depth = max(3.0, params.depth_mm * ppm)
    mask = card_mask((H, W), outer, radius)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    band = (dist > inset) & (dist <= inset + depth)
    ref = (dist > inset + 1.5 * depth) & (dist <= inset + 4 * depth)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    C = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)
    k = int(6 * depth) | 1
    rf = ref.astype(np.float32)
    wsum = cv2.blur(rf, (k, k)) + 1e-6
    refL = cv2.blur(L * rf, (k, k)) / wsum
    refC = cv2.blur(C * rf, (k, k)) / wsum
    has_ref = wsum > 0.02
    # Local speckle level of the border (glitter, foil, holo, print texture).
    varL = np.maximum(cv2.blur(L * L * rf, (k, k)) / wsum - refL ** 2, 0)
    varC = np.maximum(cv2.blur(C * C * rf, (k, k)) / wsum - refC ** 2, 0)
    texture = np.sqrt(varL + 0.64 * varC)

    whitish_ref = (refL > 185) & (refC < 22)  # white / silver border: can't judge
    thr = np.maximum(params.delta, params.texture_z * texture) / max(sensitivity, 0.05)
    whiteness = (L - refL) + 0.8 * (refC - C)
    flag = (band & has_ref & ~whitish_ref & (L > params.min_l)
            & (C < params.max_chroma * max(1.0, sensitivity)) & (whiteness > thr))
    flag = cv2.morphologyEx(flag.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Sides where most of the band can't be judged.
    unreliable, textured = [], []
    cw, ch = outer.right - outer.left, outer.bottom - outer.top
    yy, xx = np.nonzero(band)
    side_of = np.argmin(np.stack([xx - outer.left, outer.right - xx, yy - outer.top, outer.bottom - yy]), axis=0)
    wr = whitish_ref[yy, xx]
    tx = texture[yy, xx]
    for i, name in enumerate(("left", "right", "top", "bottom")):
        sel = side_of == i
        if sel.any() and wr[sel].mean() > 0.5:
            unreliable.append(name)
        elif sel.any() and float(np.median(tx[sel])) > params.textured_level:
            textured.append(name)

    # Group nearby flagged pixels into spots.
    grow = max(2, int(round(0.15 * ppm)))
    grouped = cv2.dilate(flag, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    min_px = params.min_area_mm2 * ppm * ppm
    zone = radius + 2.0 * ppm
    spots: List[Spot] = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        member = (labels[y:y + h, x:x + w] == i) & (flag[y:y + h, x:x + w] > 0)
        cnt = int(member.sum())
        if cnt < min_px:
            continue
        # Edge wear starts at the edge and is concentrated there; glitter and print
        # speckle sit anywhere in the band. Require both.
        dd = dist[y:y + h, x:x + w][member]
        if dd.min() > inset + max(2.0, params.edge_touch_mm * ppm):
            continue
        if np.mean(dd <= inset + depth / 2) < 0.5:
            continue
        contrast = float(whiteness[y:y + h, x:x + w][member].mean())
        cx, cy = x + w / 2, y + h / 2
        loc = _locate(cx, cy, outer, zone)
        long_side = cw if loc.split()[0] in ("top", "bottom") else ch
        sides = [p for p in loc.replace("-", " ").split()[:2] if p in ("left", "right", "top", "bottom")]
        if max(w, h) > 0.35 * long_side:
            kind = "check-line"
        elif ppm < params.min_px_per_mm or any(sd in textured for sd in sides):
            kind = "verify"
        else:
            kind = "whitening"
        spots.append(Spot(int(x), int(y), int(w), int(h), loc, cnt / (ppm * ppm), contrast, kind))
    spots.sort(key=lambda s: -s.area_mm2)
    return DefectScan(spots, _severity(spots), unreliable, textured, ppm < params.min_px_per_mm)


def annotate(img: np.ndarray, scan: DefectScan, px_per_mm: float) -> np.ndarray:
    """Copy of img with defect boxes drawn on it."""
    out = img.copy()
    t = max(2, int(round(0.12 * px_per_mm)))
    pad = int(round(0.3 * px_per_mm))
    for s in scan.spots:
        color = {"whitening": (0, 0, 255), "verify": (0, 200, 255)}.get(s.kind, (0, 120, 255))
        cv2.rectangle(out, (s.x - pad, s.y - pad), (s.x + s.w + pad, s.y + s.h + pad), color, t)
    return out


def corner_crops(img: np.ndarray, outer: Box, px_per_mm: float, size_mm: float = 6.0) -> Dict[str, np.ndarray]:
    """Square crops around each corner; the card corner sits ~1 mm in from the crop edge."""
    s = int(round(size_mm * px_per_mm))
    m = int(round(1.0 * px_per_mm))
    H, W = img.shape[:2]
    pts = {
        "top-left": (outer.left - m, outer.top - m),
        "top-right": (outer.right + m - s, outer.top - m),
        "bottom-left": (outer.left - m, outer.bottom + m - s),
        "bottom-right": (outer.right + m - s, outer.bottom + m - s),
    }
    out = {}
    for name, (x, y) in pts.items():
        x0 = int(np.clip(round(x), 0, W - s)); y0 = int(np.clip(round(y), 0, H - s))
        out[name] = img[y0:y0 + s, x0:x0 + s].copy()
    return out


def edge_strips(img: np.ndarray, outer: Box, px_per_mm: float, depth_mm: float = 2.5) -> Dict[str, np.ndarray]:
    """Full-length strips along each edge, rotated so the card edge is always at the top."""
    d = int(round(depth_mm * px_per_mm))
    m = int(round(0.6 * px_per_mm))
    l, t, r, b = (int(round(v)) for v in (outer.left, outer.top, outer.right, outer.bottom))
    H, W = img.shape[:2]
    cl = lambda v, hi: int(np.clip(v, 0, hi))  # noqa: E731
    top = img[cl(t - m, H):cl(t + d, H), cl(l, W):cl(r, W)]
    bottom = cv2.rotate(img[cl(b - d, H):cl(b + m, H), cl(l, W):cl(r, W)], cv2.ROTATE_180)
    left = cv2.rotate(img[cl(t, H):cl(b, H), cl(l - m, W):cl(l + d, W)], cv2.ROTATE_90_CLOCKWISE)
    right = cv2.rotate(img[cl(t, H):cl(b, H), cl(r - d, W):cl(r + m, W)], cv2.ROTATE_90_COUNTERCLOCKWISE)
    return {"top": top, "right": right, "bottom": bottom, "left": left}
