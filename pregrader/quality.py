"""Photo-quality checks: tell the user when a photo is too poor to trust."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from .centering import Box
from .geometry import BUSY_BACKGROUND, background_busyness


@dataclass
class Quality:
    issues: List[tuple] = field(default_factory=list)   # (level "warn"|"bad", message)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def level(self) -> str:
        levels = [lv for lv, _ in self.issues]
        return "bad" if "bad" in levels else "warn" if "warn" in levels else "ok"


def _side_profiles(lab: np.ndarray, outer: Box, ppm: float, side: str, n: int = 40,
                   out_mm: float = 1.5, in_mm: float = 1.0) -> np.ndarray:
    """Profiles (n x m x 3) across one outer edge, running from outside to inside."""
    H, W = lab.shape[:2]
    o_px, i_px = int(round(out_mm * ppm)), int(round(in_mm * ppm))
    if side in ("left", "right"):
        ys = np.linspace(outer.top + 0.15 * (outer.bottom - outer.top),
                         outer.bottom - 0.15 * (outer.bottom - outer.top), n).astype(int)
        e = int(round(outer.get(side)))
        if side == "left":
            xs = np.arange(e - o_px, e + i_px)
        else:
            xs = np.arange(e + o_px, e - i_px, -1)
        xs = np.clip(xs, 0, W - 1)
        return lab[ys[:, None], xs[None, :]]
    xs = np.linspace(outer.left + 0.15 * (outer.right - outer.left),
                     outer.right - 0.15 * (outer.right - outer.left), n).astype(int)
    e = int(round(outer.get(side)))
    ys = np.arange(e - o_px, e + i_px) if side == "top" else np.arange(e + o_px, e - i_px, -1)
    ys = np.clip(ys, 0, H - 1)
    return lab[ys[None, :], xs[:, None]]


def edge_metrics(lab: np.ndarray, outer: Box, ppm: float) -> Dict[str, float]:
    """Edge blur (equivalent transition width, mm) and background contrast (Lab dE)."""
    widths, contrasts = [], []
    o_px = int(round(1.5 * ppm))
    for side in ("left", "right", "top", "bottom"):
        prof = _side_profiles(lab, outer, ppm, side)
        bg = np.median(prof[:, : int(0.6 * ppm)], axis=1)                       # outside
        fg = np.median(prof[:, o_px + int(0.3 * ppm): o_px + int(0.8 * ppm)], axis=1)  # border
        d = fg - bg
        norm = np.sqrt((d ** 2).sum(axis=1))
        contrasts.extend(norm.tolist())
        ok = norm > 15
        if not ok.any():
            continue
        # Equivalent edge width = step size / steepest slope. A shadow line or a bright
        # cut edge only adds slope, so they can't make a sharp photo look blurry.
        sm = cv2.blur(prof.astype(np.float32), (3, 1))
        win = sm[:, int(0.6 * ppm): o_px + int(0.6 * ppm)]
        slope = np.sqrt(((win[:, 2:] - win[:, :-2]) ** 2).sum(axis=2)) / 2
        w = norm[ok] / (slope[ok].max(axis=1) + 1e-6) / ppm
        widths.extend(w.tolist())
    return {
        "edge_blur_mm": float(np.median(widths)) if widths else float("nan"),
        "background_contrast": float(np.median(contrasts)) if contrasts else float("nan"),
    }


def assess(img: np.ndarray, outer: Box, ppm: float, original: np.ndarray = None,
           inner_confidence: dict = None) -> Quality:
    q = Quality()
    if original is not None:
        busy = background_busyness(original)
        q.metrics["background_busyness"] = busy
        if busy > BUSY_BACKGROUND:
            q.issues.append(("warn", "Patterned background (cloth/carpet): the card edges were found using "
                                     "texture, which works but is less exact. A plain, dark, smooth mat is best."))
    if inner_confidence:
        weak = [k for k, v in inner_confidence.items() if v in ("low", "none")]
        if len(weak) >= 3:
            q.issues.append(("bad", "Most border lines couldn't be found - the card outline may be wrong. "
                                    "Check it in 'Card corners' mode before trusting the results."))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    C = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)
    m = int(round(1.0 * ppm))
    l, t, r, b = (int(round(v)) for v in (outer.left, outer.top, outer.right, outer.bottom))
    card_L, card_C = L[t + m:b - m, l + m:r - m], C[t + m:b - m, l + m:r - m]

    q.metrics["px_per_mm"] = ppm
    if ppm < 20:
        q.issues.append(("bad", f"Low resolution ({ppm:.0f} px/mm): move closer or use the full-resolution "
                                "photo. Whitening can't be judged and centering is less precise."))
    elif ppm < 28:
        q.issues.append(("warn", f"Modest resolution ({ppm:.0f} px/mm): fine edge wear may be missed."))

    med = float(np.median(card_L))
    q.metrics["median_brightness"] = med
    if med < 75:
        q.issues.append(("bad", "Photo is too dark: shoot in brighter, even light (daylight near a window)."))
    elif med < 100:
        q.issues.append(("warn", "Photo is a little dark: more light will make edges and whitening clearer."))

    # Border brightness per side (a ring 0.4-1.2 mm in): big differences = shadow / uneven light.
    a, z = int(0.4 * ppm), int(1.2 * ppm)
    sides = {"left": L[t + m:b - m, l + a:l + z], "right": L[t + m:b - m, r - z:r - a],
             "top": L[t + a:t + z, l + m:r - m], "bottom": L[b - z:b - a, l + m:r - m]}
    lv = {k: float(np.median(v)) for k, v in sides.items() if v.size}
    if lv:
        hi, lo = max(lv.values()), min(lv.values())
        q.metrics["light_ratio"] = hi / max(lo, 1.0)
        if hi / max(lo, 1.0) > 1.45:
            dark = min(lv, key=lv.get)
            q.issues.append(("warn", f"Uneven lighting: the {dark} of the card is noticeably darker "
                                     "(shadow?). Light the card evenly from above."))

    glare = float(np.mean((card_L > 250) & (card_C < 12)))
    q.metrics["glare_fraction"] = glare
    if glare > 0.01:
        q.issues.append(("bad", "Strong glare on the card: tilt the light or card so no reflections show."))
    elif glare > 0.002:
        q.issues.append(("warn", "Some glare highlights: may hide or fake whitening."))

    em = edge_metrics(lab, outer, ppm)
    q.metrics.update(em)
    blur = em["edge_blur_mm"]
    if blur == blur:
        if blur > 0.3:
            q.issues.append(("bad", f"Edges are blurry ({blur:.2f} mm): hold steady / tap to focus on the card."))
        elif blur > 0.18:
            q.issues.append(("warn", f"Edges are slightly soft ({blur:.2f} mm): a sharper photo helps."))
    bc = em["background_contrast"]
    if bc == bc and bc < 30:
        q.issues.append(("warn", "Low contrast between the card edge and background: use a darker "
                                 "background (or lighter, for black-bordered cards)."))
    return q
