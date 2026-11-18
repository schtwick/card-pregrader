"""Border lines, automatic inner-border detection and centering ratios."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SIDES = ("left", "right", "top", "bottom")


@dataclass
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def get(self, side: str) -> float:
        return getattr(self, side)

    def set(self, side: str, value: float) -> None:
        setattr(self, side, float(value))

    def copy(self) -> "Box":
        return replace(self)

    def rotated_cw(self, img_height: int) -> "Box":
        """This box after the image is rotated 90 degrees clockwise."""
        h1 = img_height - 1
        return Box(left=h1 - self.bottom, top=self.left, right=h1 - self.top, bottom=self.right)


@dataclass
class Lines:
    outer: Box  # physical card edge
    inner: Box  # where the border meets the artwork / print frame
    confidence: Dict[str, str] = field(default_factory=dict)  # per inner side: high|low|none|manual

    def copy(self) -> "Lines":
        return Lines(self.outer.copy(), self.inner.copy(), dict(self.confidence))

    def rotated_cw(self, img_height: int) -> "Lines":
        c = self.confidence
        conf = {"left": c.get("bottom", "none"), "top": c.get("left", "none"),
                "right": c.get("top", "none"), "bottom": c.get("right", "none")}
        return Lines(self.outer.rotated_cw(img_height), self.inner.rotated_cw(img_height), conf)


def split(a: float, b: float) -> Tuple[float, float]:
    """Percent split of two border widths, e.g. (52.3, 47.7)."""
    t = a + b
    if t <= 0 or not math.isfinite(t):
        return (float("nan"), float("nan"))
    return (100.0 * a / t, 100.0 * b / t)


def ratio_text(pct: Tuple[float, float]) -> str:
    if not math.isfinite(pct[0]):
        return "--/--"
    a = int(round(pct[0]))
    return f"{a}/{100 - a}"


@dataclass
class Centering:
    left: float    # border widths in px
    right: float
    top: float
    bottom: float
    px_per_mm: float
    # Spread (SD, percentage points) of the ratio measured at several positions along
    # the card. Captures print tilt, warp residue and detection noise.
    spread_lr: float = 0.0
    spread_tb: float = 0.0
    line_error_px: float = 0.75  # assumed placement error of each line

    def _sigma(self, a: float, b: float, spread: float) -> float:
        t = a + b
        if t <= 0:
            return float("nan")
        base = 100.0 * self.line_error_px * math.hypot(a, b) / (t * t)
        return math.hypot(base, spread if math.isfinite(spread) else 3.0)

    @property
    def sigma_lr(self) -> float:
        """1-sigma uncertainty of the L/R ratio, percentage points."""
        return self._sigma(self.left, self.right, self.spread_lr)

    @property
    def sigma_tb(self) -> float:
        return self._sigma(self.top, self.bottom, self.spread_tb)

    @property
    def lr(self) -> Tuple[float, float]:
        return split(self.left, self.right)

    @property
    def tb(self) -> Tuple[float, float]:
        return split(self.top, self.bottom)

    @property
    def worst_lr(self) -> float:
        return max(self.lr)

    @property
    def worst_tb(self) -> float:
        return max(self.tb)

    @property
    def worst(self) -> float:
        """Larger share of the worse axis, e.g. 58.0 for 58/42."""
        return max(self.worst_lr, self.worst_tb)

    def mm(self, side: str) -> float:
        return getattr(self, side) / self.px_per_mm

    def summary(self) -> str:
        return f"L/R {ratio_text(self.lr)}  T/B {ratio_text(self.tb)}"


def measure(lines: Lines, px_per_mm: float, spread: Optional[Dict[str, float]] = None) -> Centering:
    o, i = lines.outer, lines.inner
    spread = spread or {}
    return Centering(
        left=max(0.0, i.left - o.left), right=max(0.0, o.right - i.right),
        top=max(0.0, i.top - o.top), bottom=max(0.0, o.bottom - i.bottom),
        px_per_mm=px_per_mm,
        spread_lr=spread.get("lr", 0.0), spread_tb=spread.get("tb", 0.0),
    )


def _first_strong_step(prof: np.ndarray, span: float) -> Tuple[float, float, float]:
    """Find the first strong colour step in a 1-D Lab profile that starts at the card
    edge and runs inward. Returns (boundary index, strength, contrast ratio)."""
    n = len(prof)
    k = max(2, int(round(0.0012 * span)))
    start = max(k, int(round(0.008 * span)))  # skip the physical edge / any whitening (~0.7 mm)
    if n < start + 2 * k + 3:
        return float("nan"), 0.0, 0.0
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(prof, axis=0)])
    i = np.arange(start, n - k - 1)
    before = (cs[i + 1] - cs[i + 1 - k]) / k
    after = (cs[i + 1 + k] - cs[i + 1]) / k
    step = np.sqrt(((after - before) ** 2).sum(axis=1))
    mx = float(step.max())
    if mx < 6.0:
        return float("nan"), mx, 0.0
    # The border ends where the colour first departs from the border's own colour
    # (taken just inside the edge) and stays different. Then snap to the sharpest
    # colour step around that point.
    m = max(3, int(round(0.004 * span)))
    ref = np.median(prof[start:start + m], axis=0)
    dep = np.sqrt(((prof - ref) ** 2).sum(axis=1))
    thr = max(12.0, 0.3 * float(dep[start:].max()))
    run_len = max(2, k)
    peak = None
    for x in range(start + m, n - run_len):
        if np.all(dep[x:x + run_len] > thr):
            # x is the first pixel off the border colour, so the boundary lies at or
            # just before it; never look further in (the next edge may be stronger).
            lo = max(0, x - k - 2 - start)
            hi = min(len(step), x + 1 - start)
            peak = lo + int(np.argmax(step[lo:hi]))
            break
    if peak is None:
        peak = int(np.argmax(step))
    a = step[peak - 1] if peak > 0 else step[peak]
    b = step[peak]
    c = step[peak + 1] if peak < len(step) - 1 else step[peak]
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    contrast = float(b / (np.median(step) + 1e-6))
    return float(i[peak]) + 0.5 + float(np.clip(delta, -0.5, 0.5)), float(b), contrast


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


def _inner_at(lab: np.ndarray, outer: Box, side: str, f0: float, f1: float,
              depth: float = 0.3) -> Tuple[float, float, float]:
    """Inner border position on one side, using the stretch from fraction f0 to f1
    along that side. Returns (coordinate, strength, contrast); NaN if not found."""
    H, W = lab.shape[:2]
    cw, ch = outer.right - outer.left, outer.bottom - outer.top
    xl = int(math.floor(outer.left + 0.5)); xr = int(math.floor(outer.right + 0.5))
    yt = int(math.floor(outer.top + 0.5)); yb = int(math.floor(outer.bottom + 0.5))
    nx, ny = int(depth * cw), int(depth * ch)
    if side in ("left", "right"):
        a0, a1 = int(outer.top + f0 * ch), int(outer.bottom - (1 - f1) * ch)
        span = cw
        if side == "left":
            prof, origin, sign = np.median(lab[a0:a1, xl:min(W, xl + nx)], axis=0), xl, +1
        else:
            prof, origin, sign = np.median(lab[a0:a1, max(0, xr - nx + 1):xr + 1][:, ::-1], axis=0), xr, -1
    else:
        a0, a1 = int(outer.left + f0 * cw), int(outer.right - (1 - f1) * cw)
        span = ch
        if side == "top":
            prof, origin, sign = np.median(lab[yt:min(H, yt + ny), a0:a1], axis=1), yt, +1
        else:
            prof, origin, sign = np.median(lab[max(0, yb - ny + 1):yb + 1, a0:a1][::-1], axis=1), yb, -1
    idx, strength, contrast = _first_strong_step(prof, span)
    return origin + sign * idx, strength, contrast


def detect_inner(img: np.ndarray, outer: Box, depth: float = 0.3) -> Tuple[Box, Dict[str, str]]:
    """Auto-detect the inner (artwork) border on all four sides.

    For each side, takes the per-column (or per-row) median colour across the middle
    60% of the card, and finds where the colour first departs from the border's own
    colour. The median means more than half the rows must agree, so artwork that
    touches the border in a few places, glitter or text doesn't fool it."""
    lab = _lab(img)
    inner = outer.copy()
    conf: Dict[str, str] = {}
    for side in SIDES:
        v, strength, contrast = _inner_at(lab, outer, side, 0.2, 0.8, depth)
        if not math.isfinite(v):
            span = outer.right - outer.left if side in ("left", "right") else outer.bottom - outer.top
            sign = 1 if side in ("left", "top") else -1
            inner.set(side, outer.get(side) + sign * 0.05 * span)
            conf[side] = "none"
        else:
            inner.set(side, v)
            conf[side] = "high" if (strength > 20 and contrast > 4) else "low"
    return inner, conf


def measure_spread(img: np.ndarray, outer: Box, n: int = 6) -> Dict[str, float]:
    """How much each centering ratio changes along the card: measures both opposite
    borders at n matching positions and returns the robust SD of the ratios (points).
    NaN when too few positions could be measured."""
    lab = _lab(img)
    edges = np.linspace(0.1, 0.9, n + 1)
    out: Dict[str, float] = {}
    for axis, (a, b) in (("lr", ("left", "right")), ("tb", ("top", "bottom"))):
        ratios: List[float] = []
        for f0, f1 in zip(edges[:-1], edges[1:]):
            va = _inner_at(lab, outer, a, f0, f1)[0]
            vb = _inner_at(lab, outer, b, f0, f1)[0]
            wa, wb = va - outer.get(a), outer.get(b) - vb
            if math.isfinite(wa) and math.isfinite(wb) and wa > 0 and wb > 0:
                ratios.append(100.0 * wa / (wa + wb))
        if len(ratios) < 3:
            out[axis] = float("nan")
            continue
        r = np.array(ratios)
        med = np.median(r)
        mad = 1.4826 * np.median(np.abs(r - med)) + 1e-6
        r = r[np.abs(r - med) <= max(3 * mad, 1.0)]
        out[axis] = float(np.std(r)) if len(r) >= 3 else float("nan")
    return out


@dataclass
class SideSquareness:
    widths_mm: List[float]      # border width at evenly spaced points along the side
    change_mm: float            # fitted change from one end of the side to the other
    significant: bool           # bigger than the measurement noise / resolution

    @property
    def wider_end(self) -> Optional[str]:
        return None if not self.significant else ("start" if self.change_mm < 0 else "end")


def measure_squareness(img: np.ndarray, outer: Box, px_per_mm: float, n: int = 8) -> Dict[str, SideSquareness]:
    """Is the printed frame parallel to each card edge? Measures each border at n
    points along its length and fits a line. 'start' is the top (left/right sides)
    or the left end (top/bottom sides)."""
    lab = _lab(img)
    edges = np.linspace(0.08, 0.92, n + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    out: Dict[str, SideSquareness] = {}
    for side in SIDES:
        span = (outer.bottom - outer.top) if side in ("left", "right") else (outer.right - outer.left)
        w = np.array([abs(_inner_at(lab, outer, side, f0, f1)[0] - outer.get(side))
                      for f0, f1 in zip(edges[:-1], edges[1:])]) / px_per_mm
        x = mids * span / px_per_mm
        ok = np.isfinite(w)
        if ok.sum() < 4:
            out[side] = SideSquareness(w.tolist(), float("nan"), False)
            continue
        xs, ws = x[ok], w[ok]
        med = np.median(ws)
        keep = np.abs(ws - med) <= max(3 * 1.4826 * np.median(np.abs(ws - med)), 0.15)
        if keep.sum() >= 4:
            xs, ws = xs[keep], ws[keep]
        k, b0 = np.polyfit(xs, ws, 1)
        change = float(k * (mids[-1] - mids[0]) * span / px_per_mm)
        resid = float(np.std(ws - (k * xs + b0)))
        floor = max(0.3, 3.0 / px_per_mm, 2.5 * resid)
        out[side] = SideSquareness(np.round(w, 3).tolist(), change, abs(change) > floor)
    return out
