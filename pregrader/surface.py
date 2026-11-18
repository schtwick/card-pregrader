"""Surface check from angled-light photos.

A scratch, dent or crease is a change in the card's *shape*, so it shows up when light
skims the card from a low angle. The printed artwork does not change with the light.
Each angled-light photo is aligned pixel-for-pixel with the main (evenly lit) photo and
divided by it: the artwork cancels, and what remains is how the surface catches the
light. Thin lines in that "relief" image are scratches, round spots are dents or pits,
long lines are creases. Foil/holo areas change colour with the angle, so they're
marked as unjudgeable instead of producing false defects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .centering import Box
from .geometry import detect_card, order_quad


@dataclass
class SurfaceDefect:
    x: int
    y: int
    w: int
    h: int
    kind: str          # scratch | crease | dent | mark (small: scratch or dent, can't tell)
    size_mm: float     # length for lines, diameter for dents
    contrast: float    # strength vs. the photo's own noise
    location: str
    ends: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # line end points (x0, y0, x1, y1)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class SurfaceScan:
    defects: List[SurfaceDefect]
    severity: str                       # none | minor | moderate | heavy
    photos_used: int
    photos_rejected: int
    foil_fraction: float                # share of the card that couldn't be judged
    relief: Optional[np.ndarray] = None  # uint8 visualisation, main-photo frame
    notes: List[str] = field(default_factory=list)


@dataclass
class SurfaceParams:
    margin_mm: float = 1.2      # skip the perimeter (edges are checked separately)
    ridge_sigma_mm: float = 0.06  # scratch width scale
    blob_sigma_mm: float = 0.45   # dent size scale
    line_z: float = 6.0         # ridge strength vs. robust noise
    blob_z: float = 4.5
    min_scratch_mm: float = 1.5
    crease_mm: float = 20.0
    short_line_mm: float = 4.5    # shorter "lines" next to a dent are the dent's own shading
    min_line_strength: float = 9.5
    foil_level: float = 2.2     # local relief noise (x global) that marks foil/holo
    min_align_corr: float = 0.65   # same card 0.86-0.97 in tests (even with a crease); different card ~0.35-0.45


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; very wide ones are done on a shrunk copy (same result, much faster,
    which matters in the phone version)."""
    if sigma <= 6:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = sigma / 3.0
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- align
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _highpass(g: np.ndarray, s: float) -> np.ndarray:
    hp = g - cv2.GaussianBlur(g, (0, 0), s)
    return hp / (np.std(hp) + 1e-6)


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    # clipped, so a strong scratch/crease shadow can't drag down the match of the print
    a, b = np.clip(a[mask], -2.5, 2.5), np.clip(b[mask], -2.5, 2.5)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def align(main: np.ndarray, outer: Box, other: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Warp `other` (a raw photo of the same card side) into the main photo's frame.
    Returns (aligned image or None, correlation of fine detail after alignment)."""
    quad = detect_card(other)
    if quad is None:
        return None, 0.0
    H, W = main.shape[:2]
    dst = np.float32([[outer.left, outer.top], [outer.right, outer.top],
                      [outer.right, outer.bottom], [outer.left, outer.bottom]])
    q = order_quad(quad).astype(np.float32)
    s = 900.0 / max(H, W)
    small_main = cv2.resize(main, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hm = _highpass(_gray(small_main), 3)
    mask = np.zeros(hm.shape, bool)
    l, t, r, b = (int(v * s) for v in (outer.left, outer.top, outer.right, outer.bottom))
    mask[t + 8:b - 8, l + 8:r - 8] = True
    S = np.diag([s, s, 1.0])
    small_other = cv2.resize(other, None, fx=min(1.0, 1500 / max(other.shape[:2])),
                             fy=min(1.0, 1500 / max(other.shape[:2])), interpolation=cv2.INTER_AREA)
    so = small_other.shape[1] / other.shape[1]
    So = np.diag([so, so, 1.0])
    sh, sw = hm.shape
    best_c, best_M = -1.0, None
    for k in range(4):  # the card may be turned in the other photo: try all four on small copies
        M = cv2.getPerspectiveTransform(np.roll(q, -k, axis=0), dst).astype(np.float64)
        Ms = S @ M @ np.linalg.inv(So)
        ho = _highpass(_gray(cv2.warpPerspective(small_other, Ms, (sw, sh), flags=cv2.INTER_LINEAR)), 3)
        c = _corr(hm, ho, mask)
        if c > best_c:
            best_c, best_M, best_ho = c, M, ho
    # Fine alignment on fine detail (the print), the same in both photos however the
    # light falls. Done on small copies; the result is scaled up.
    M_final = best_M
    try:
        warp = np.eye(3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        _, warp = cv2.findTransformECC(hm.astype(np.float32), best_ho.astype(np.float32), warp,
                                       cv2.MOTION_HOMOGRAPHY, crit, mask.astype(np.uint8), 5)
        Wf = np.linalg.inv(S) @ warp.astype(np.float64) @ S
        M2 = np.linalg.inv(Wf) @ best_M     # ECC maps template->input; invert to refine
        hr = _highpass(_gray(cv2.warpPerspective(small_other, S @ M2 @ np.linalg.inv(So), (sw, sh),
                                                 flags=cv2.INTER_LINEAR)), 3)
        c2 = _corr(hm, hr, mask)
        if c2 >= best_c:
            M_final, best_c = M2, c2
    except cv2.error:
        pass
    best = cv2.warpPerspective(other, M_final, (W, H), flags=cv2.INTER_LINEAR)
    return best, best_c


# ---------------------------------------------------------------------------- relief
def relief_map(main: np.ndarray, aligned: np.ndarray, ppm: float) -> Tuple[np.ndarray, np.ndarray]:
    """log(angled / even) with the broad light falloff removed. Returns (relief, valid)."""
    gm = cv2.GaussianBlur(_gray(main), (0, 0), 0.8)
    go = cv2.GaussianBlur(_gray(aligned), (0, 0), 0.8)
    # Very dark ink reflects almost nothing, so the ratio there is noise: don't judge it.
    # (dark *shadows* in the angled photo are the evidence, so only the main photo's
    # ink darkness and either photo's blown highlights limit what can be judged)
    # Glare is a broad blown-out patch; a thin blown-out line is the lit side of a
    # crease or scratch (evidence), so only broad saturated areas are excluded.
    k = max(3, int(0.8 * ppm) | 1)
    glare = cv2.morphologyEx((go >= 252).astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    valid = (gm > 40) & (gm < 250) & ~glare
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    r = np.log(go + 4.0) - np.log(gm + 4.0)
    r = r - _blur(r, 1.5 * ppm)
    # Print edges leak into the ratio when the photos are a fraction of a pixel apart
    # or differently sharp; damp the relief right on strong print edges.
    gx = cv2.Sobel(gm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gm, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_scale = np.percentile(edge[valid], 90) + 1e-6 if valid.any() else 1.0
    r = r / (1.0 + (edge / edge_scale) ** 2)
    return r.astype(np.float32), valid


def print_edges(main: np.ndarray, ppm: float) -> np.ndarray:
    """Strong edges in the printed artwork (where alignment leaks are likely)."""
    gm = cv2.GaussianBlur(_gray(main), (0, 0), 0.8)
    gx = cv2.Sobel(gm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gm, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    strong = (e > np.percentile(e, 93)).astype(np.uint8)
    k = max(3, int(0.15 * ppm) | 1)
    return cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def _locate(cx: float, cy: float, outer: Box) -> str:
    fx = (cx - outer.left) / max(outer.right - outer.left, 1)
    fy = (cy - outer.top) / max(outer.bottom - outer.top, 1)
    v = "top" if fy < 1 / 3 else "bottom" if fy > 2 / 3 else "middle"
    h = "left" if fx < 1 / 3 else "right" if fx > 2 / 3 else "centre"
    return "centre" if (v, h) == ("middle", "centre") else f"{v}-{h}" if v != "middle" else f"middle-{h}"


def _severity(defects: List[SurfaceDefect]) -> str:
    if not defects:
        return "none"
    if any(d.kind == "crease" for d in defects):
        return "severe"
    big = sum(1 for d in defects if (d.kind == "scratch" and d.size_mm > 6) or (d.kind in ("dent", "mark") and d.size_mm > 3))
    if len(defects) == 1 and not big:
        return "minor"
    if len(defects) <= 3 and big <= 1:
        return "moderate"
    return "heavy"


def _line_defects(lines: np.ndarray, ridge: np.ndarray, judge: np.ndarray, outer: Box, ppm: float,
                  params: SurfaceParams, relief: Optional[np.ndarray] = None) -> List[SurfaceDefect]:
    """Straight scratches/creases: join collinear pieces (they break where they cross
    dark ink or print edges), then require the path to really show a line."""
    segs = cv2.HoughLinesP(lines, 1, np.pi / 360, threshold=max(8, int(0.6 * ppm)),
                           minLineLength=params.min_scratch_mm * ppm * 0.6, maxLineGap=3.0 * ppm)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(np.float64)
    used = np.zeros(len(segs), bool)
    out: List[SurfaceDefect] = []
    order = np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
    for i in order:
        if used[i]:
            continue
        x0, y0, x1, y1 = segs[i]
        d = np.array([x1 - x0, y1 - y0])
        d /= np.hypot(*d) + 1e-9
        nrm = np.array([-d[1], d[0]])
        p0 = np.array([x0, y0])
        group = [i]
        for j in order:
            if used[j] or j == i:
                continue
            a, b = segs[j, :2], segs[j, 2:]
            dj = (b - a) / (np.hypot(*(b - a)) + 1e-9)
            if abs(float(np.dot(dj, d))) < np.cos(np.radians(8)):
                continue
            if max(abs(np.dot(a - p0, nrm)), abs(np.dot(b - p0, nrm))) > 0.8 * ppm:
                continue
            group.append(j)
        pts = np.concatenate([segs[group, :2], segs[group, 2:]])
        if len(group) > 1:  # fit through all joined pieces, not just the first one
            vx, vy, fx, fy = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            d = np.array([vx, vy], float)
            nrm = np.array([-d[1], d[0]])
            p0 = np.array([fx, fy], float)
        t = (pts - p0) @ d
        t0, t1 = float(t.min()), float(t.max())
        tt = np.arange(t0, t1, 1.0)
        if len(tt) < 3:
            continue
        # walk the path: how much of it shows the line?
        path = p0 + tt[:, None] * d
        hits, inside, peaks = 0, 0, []
        for px, py in path.astype(int):
            if not (0 <= py < lines.shape[0] and 0 <= px < lines.shape[1]):
                continue
            if not judge[py, px]:
                continue
            inside += 1
            hw = max(2, int(0.25 * ppm))
            r0, r1 = max(0, py - hw), py + hw + 1
            c0, c1 = max(0, px - hw), px + hw + 1
            hits += int(lines[r0:r1, c0:c1].any())
            peaks.append(float(ridge[r0:r1, c0:c1].max()))
        used[group] = True
        length = (t1 - t0) / ppm
        if length < params.min_scratch_mm or inside < 0.5 * len(tt) or hits < 0.45 * inside:
            continue
        kind = "crease" if length >= params.crease_mm else "scratch" if length >= 4.0 else "mark"
        xs, ys = path[:, 0], path[:, 1]
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        # the signal peaks on the line's shaded edges, a few pixels either side of its centre
        strength = float(np.median(peaks)) if peaks else 0.0
        cand = SurfaceDefect(x, y, w, h, kind, length, strength, _locate(x + w / 2, y + h / 2, outer),
                             (float(path[0, 0]), float(path[0, 1]), float(path[-1, 0]), float(path[-1, 1])))
        _merge(out, cand, ppm)
    return out


def _is_line(d: SurfaceDefect) -> bool:
    # short marks are compared by position, not direction (a dent's shading pieces point
    # every which way)
    return d.kind in ("scratch", "crease") and d.ends != (0.0, 0.0, 0.0, 0.0)


def _same(a: SurfaceDefect, b: SurfaceDefect, ppm: float) -> bool:
    """Do two findings describe the same physical defect?"""
    if _is_line(a) and _is_line(b):
        long_, short = (a, b) if a.size_mm >= b.size_mm else (b, a)
        p0 = np.array(long_.ends[:2]); p1 = np.array(long_.ends[2:])
        d = (p1 - p0) / (np.hypot(*(p1 - p0)) + 1e-9)
        q0 = np.array(short.ends[:2]); q1 = np.array(short.ends[2:])
        e = (q1 - q0) / (np.hypot(*(q1 - q0)) + 1e-9)
        if abs(float(np.dot(d, e))) < np.cos(np.radians(12)):
            return False
        nrm = np.array([-d[1], d[0]])
        if max(abs(np.dot(q0 - p0, nrm)), abs(np.dot(q1 - p0, nrm))) > 1.2 * ppm:
            return False
        t = sorted([np.dot(q0 - p0, d), np.dot(q1 - p0, d)])
        L = np.hypot(*(p1 - p0))
        overlap = min(t[1], L) - max(t[0], 0.0)
        return overlap > 0.4 * max(t[1] - t[0], 1.0)
    if _is_line(a) or _is_line(b):
        # a point-like finding (dent/mark) that lies on a scratch/crease is part of it
        line, pt = (a, b) if _is_line(a) else (b, a)
        p0 = np.array(line.ends[:2]); p1 = np.array(line.ends[2:])
        L = np.hypot(*(p1 - p0)) + 1e-9
        d = (p1 - p0) / L
        c = np.array(pt.center)
        t = float(np.clip(np.dot(c - p0, d), 0, L))
        return float(np.hypot(*(c - (p0 + t * d)))) < (pt.size_mm / 2 + 1.0) * ppm
    reach = (max(a.size_mm, b.size_mm) + 1.0) * ppm
    return abs(a.center[0] - b.center[0]) < reach and abs(a.center[1] - b.center[1]) < reach


def _merge(found: List[SurfaceDefect], d: SurfaceDefect, ppm: float) -> None:
    """Add d unless it duplicates something already found; for lines keep the longer
    one (a whole crease beats a fragment of it), otherwise the stronger one."""
    for i, e in enumerate(found):
        if _same(e, d, ppm):
            if _is_line(d) != _is_line(e):
                better = _is_line(d)          # the line explains the point, not the other way round
            elif _is_line(d):
                better = d.size_mm > e.size_mm
            else:
                better = d.contrast > e.contrast
            if _is_line(d) and _is_line(e):
                keep = d if better else e         # longer outline...
                keep.contrast = max(d.contrast, e.contrast)   # ...with the clearer of the two views
                found[i] = keep
            elif better:
                found[i] = d
            return
    found.append(d)


def detect(relief: np.ndarray, valid: np.ndarray, outer: Box, ppm: float,
           params: SurfaceParams = SurfaceParams(),
           edges: Optional[np.ndarray] = None, debug: Optional[dict] = None) -> Tuple[List[SurfaceDefect], np.ndarray]:
    """Find scratches, creases and dents in a relief map. Returns (defects, foil mask)."""
    H, W = relief.shape
    from .defects import card_mask
    from .geometry import CORNER_RADIUS_MM
    dist = cv2.distanceTransform(card_mask((H, W), outer, CORNER_RADIUS_MM * ppm), cv2.DIST_L2, 5)
    inside = dist > params.margin_mm * ppm      # rounded card shape, minus the perimeter
    use = inside & valid
    if use.sum() < 1000:
        return [], np.zeros((H, W), bool)
    sub = (slice(None, None, 3), slice(None, None, 3))   # statistics from every 3rd pixel
    use_s = use[sub]
    med = float(np.median(relief[sub][use_s]))
    dev = np.abs(relief - med)
    g = 1.4826 * float(np.median(dev[sub][use_s])) + 1e-6
    # Local noise level from *clipped* deviations, so one strong scratch or dent doesn't
    # make its neighbourhood look noisy; whole areas that are busy (foil, holo, dark
    # ink with little light to work with) do.
    clipped = np.minimum(dev, 4 * g) * valid
    wsum = lambda k: cv2.blur(valid.astype(np.float32), (k, k)) + 1e-6  # noqa: E731
    k5 = int(5.0 * ppm) | 1
    local5 = 1.25 * cv2.blur(clipped, (k5, k5)) / wsum(k5)
    # Foil/holo glitter flickers from pixel to pixel; a dent or crease is a smooth shape
    # and a scratch is a thin line. So "foil" = busy at the finest scale over an area.
    fine = relief - cv2.GaussianBlur(relief, (0, 0), max(1.0, 0.12 * ppm))
    fdev = np.abs(fine - np.median(fine[sub][use_s]))
    fg = 1.4826 * float(np.median(fdev[sub][use_s])) + 1e-6
    fclip = np.minimum(fdev, 4 * fg) * valid
    # Foil areas are judged at the millimetre scale, so do that on a ~6 px/mm copy:
    # same answer, and big-shape morphology stays fast in the phone version.
    f = max(1.0, ppm / 6.0)
    sw, sh = max(1, int(W / f)), max(1, int(H / f))
    small = lambda a: cv2.resize(a.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)  # noqa: E731
    sppm = ppm / f
    fclip_s, valid_s, inside_s = small(fclip), small(valid), small(inside) > 0.5
    k2s = int(2.0 * sppm) | 1
    wsum2 = cv2.blur(valid_s, (k2s, k2s)) + 1e-6
    local2 = 1.25 * cv2.blur(fclip_s, (k2s, k2s)) / wsum2
    # (only where most of the window can be judged: beside dark ink there are too few
    # pixels to tell foil from noise)
    foil_s = (local2 > params.foil_level * fg) & inside_s & (wsum2 > 0.6)
    # Real foil/holo areas are big (a holo window, a foil border); a dent or a cluster
    # of scratches is small. Keep only big busy patches.
    n_f, lab_f, st_f, _ = cv2.connectedComponentsWithStats(foil_s.astype(np.uint8), connectivity=8)
    keep = np.zeros(n_f, bool)
    keep[1:] = st_f[1:, cv2.CC_STAT_AREA] >= 20.0 * sppm * sppm
    foil_s = keep[lab_f].astype(np.uint8)
    foil_s = cv2.morphologyEx(foil_s, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2s, k2s)))
    k3s = int(3.0 * sppm) | 1
    foil_s = cv2.morphologyEx(foil_s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3s, k3s)))
    foil_s = cv2.dilate(foil_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k3s, k3s)))
    foil_wide = cv2.resize(foil_s, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    foil = foil_wide & inside
    judge = use & ~foil_wide
    if edges is None:
        edges = np.zeros((H, W), bool)
    relief = (relief - med) / np.maximum(local5, g)   # in units of local noise

    defects: List[SurfaceDefect] = []
    # Lines: Hessian ridge strength at a scratch scale and a (wider) crease scale.
    ridge = np.zeros_like(relief)
    for smm in (params.ridge_sigma_mm, 0.22):
        sg = max(0.8, smm * ppm)
        sm = cv2.GaussianBlur(relief, (0, 0), sg)
        dxx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3)
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy * dxy)
        l1 = 0.5 * (dxx + dyy + tmp)
        l2 = 0.5 * (dxx + dyy - tmp)
        big = np.where(np.abs(l1) > np.abs(l2), l1, l2)
        small = np.where(np.abs(l1) > np.abs(l2), l2, l1)
        rs = np.abs(big) * (1 - np.clip(np.abs(small) / (np.abs(big) + 1e-9), 0, 1)) * sg * sg
        rs = rs / (1.4826 * float(np.median(rs[sub][judge[sub]])) + 1e-9) if judge[sub].any() else rs
        ridge = np.maximum(ridge, rs)
    lines = ((ridge > params.line_z) & judge).astype(np.uint8)
    if debug is not None:
        debug.update(relief=relief, judge=judge, ridge=ridge, foil=foil, edges=edges)
    defects.extend(_line_defects(lines, ridge, judge, outer, ppm, params, relief))

    # Spots (dents, pits): difference of Gaussians at the dent scale.
    sb = max(1.5, params.blob_sigma_mm * ppm)
    dog = _blur(relief, sb) - _blur(relief, 2.5 * sb)
    dnoise = 1.4826 * float(np.median(np.abs(dog[sub][judge[sub]]))) + 1e-9 if judge[sub].any() else 1.0
    z = dog / dnoise
    if debug is not None:
        debug.update(dog=np.abs(z))
    # A dent lit from the side is a light half next to a dark half (one slope faces the
    # light, the other faces away). Noise specks come one at a time, so only report
    # spots that have an opposite-signed partner right beside them.
    ok = judge & ~edges
    comps = []
    for sign in (1, -1):
        m = ((sign * z) > params.blob_z) & ok
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < (0.15 * ppm) ** 2 or max(w, h) > 3 * min(w, h) + 2:
                continue
            comps.append((sign, cents[i], np.sqrt(area / np.pi), stats[i],
                          float(np.abs(z[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]).mean())))
    pos = [c for c in comps if c[0] > 0]
    neg = [c for c in comps if c[0] < 0]
    taken = set()
    for p_ in pos:
        best, best_d = None, 1e9
        for k, n_ in enumerate(neg):
            if k in taken:
                continue
            dd = float(np.hypot(*(p_[1] - n_[1])))
            # side by side (a printed dot's leak is a ring: both halves share one centre)
            if 0.6 * (p_[2] + n_[2]) <= dd < 2.2 * (p_[2] + n_[2]) and 0.25 < (p_[2] / n_[2]) ** 2 < 4 \
                    and dd < best_d:
                best, best_d = k, dd
        if best is None:
            continue
        taken.add(best)
        n_ = neg[best]
        x0 = min(p_[3][0], n_[3][0]); y0 = min(p_[3][1], n_[3][1])
        x1 = max(p_[3][0] + p_[3][2], n_[3][0] + n_[3][2]); y1 = max(p_[3][1] + p_[3][3], n_[3][1] + n_[3][3])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(d.size_mm >= params.short_line_mm and abs(d.center[0] - cx) < d.w / 2 + 0.5 * ppm
               and abs(d.center[1] - cy) < d.h / 2 + 0.5 * ppm for d in defects):
            continue  # part of a longer scratch/crease already reported
        pad = int(0.3 * ppm)
        if edges[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad].mean() > 0.03:
            continue  # sits on printed detail: a fraction-of-a-pixel misalignment looks just like this
        # the light and dark halves sit about one dent-width apart
        diam = float(best_d / ppm)
        if diam > 4.0:
            continue  # broader than any dent: lighting or print, not a dent
        defects.append(SurfaceDefect(int(x0), int(y0), int(x1 - x0), int(y1 - y0), "dent", diam,
                                     (p_[4] + n_[4]) / 2, _locate(cx, cy, outer)))
    # A dent lit from the side also looks like one or two very short lines: where a short
    # "scratch" coincides with a dent, it's the dent.
    dents = [d for d in defects if d.kind == "dent"]
    defects = [d for d in defects if d.kind == "dent" or d.size_mm >= params.short_line_mm
               or not any(abs(d.center[0] - e.center[0]) < (d.w + e.w) / 2 + 0.5 * ppm
                          and abs(d.center[1] - e.center[1]) < (d.h + e.h) / 2 + 0.5 * ppm for e in dents)]
    defects = [d for d in defects if d.kind == "dent" or d.contrast >= params.min_line_strength]
    # Nothing touching a foil/holo area is reported: those areas are "can't judge".
    if foil.any():
        pads = int(2.0 * sppm)
        near_foil = cv2.resize(cv2.dilate(foil_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pads + 1,) * 2)),
                               (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) & inside

        def touches(d: SurfaceDefect) -> bool:
            if _is_line(d):   # a long line: is much of the line itself next to foil?
                p0, p1 = np.array(d.ends[:2]), np.array(d.ends[2:])
                pts = (p0 + np.linspace(0, 1, 50)[:, None] * (p1 - p0)).astype(int)
                pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                return near_foil[pts[:, 1], pts[:, 0]].mean() > 0.3
            return near_foil[max(0, d.y):d.y + d.h + 1, max(0, d.x):d.x + d.w + 1].any()
        defects = [d for d in defects if not touches(d)]
    defects.sort(key=lambda d: -d.contrast)
    return defects, foil


def scan(main: np.ndarray, outer: Box, ppm: float, photos: List[np.ndarray],
         params: SurfaceParams = SurfaceParams(), cache: Optional[dict] = None) -> SurfaceScan:
    """Surface check for one card side from one or more angled-light photos. `cache`
    (any dict kept by the caller) remembers each photo's alignment so adding a second
    photo doesn't redo the first."""
    reliefs, valids, rejected = [], [], 0
    frame = (main.shape, round(outer.left, 1), round(outer.top, 1), round(outer.right, 1), round(outer.bottom, 1))
    for p in photos:
        key = (id(p), frame)
        if cache is not None and key in cache:
            r, v = cache[key]
        else:
            aligned, corr = align(main, outer, p)
            r, v = (None, None) if aligned is None or corr < params.min_align_corr else relief_map(main, aligned, ppm)
            if cache is not None:
                cache[key] = (r, v)
        if r is None:
            rejected += 1
            continue
        reliefs.append(r)
        valids.append(v)
    notes = []
    if rejected:
        notes.append(f"{rejected} angled-light photo(s) couldn't be lined up with the main photo and were "
                     "skipped - keep the whole card in frame and hold the phone in the same place.")
    if not reliefs:
        return SurfaceScan([], "none", 0, rejected, 0.0, None, notes)
    # Each light direction shows different defects (a crease only shows when the light
    # crosses it; a dent's light/dark halves face the light), so analyse each angled
    # photo on its own and combine the findings.
    edges = print_edges(main, ppm)
    found: List[SurfaceDefect] = []
    foil_frac = 0.0
    inside = np.zeros(reliefs[0].shape, bool)
    inside[int(outer.top):int(outer.bottom), int(outer.left):int(outer.right)] = True
    for r, v in zip(reliefs, valids):
        ds, foil = detect(r, v, outer, ppm, params, edges)
        foil_frac = max(foil_frac, float(foil[inside].mean()) if inside.any() else 0.0)
        for d in ds:
            _merge(found, d, ppm)
    defects = sorted(found, key=lambda d: -d.contrast)
    stack = np.stack(reliefs)
    pick = np.argmax(np.abs(stack), axis=0)
    relief = np.take_along_axis(stack, pick[None], axis=0)[0]
    valid = np.logical_or.reduce(valids)
    vis = np.clip(128 + relief / (np.std(relief[valid]) + 1e-6) * 32, 0, 255).astype(np.uint8) if valid.any() \
        else np.full(relief.shape, 128, np.uint8)
    if foil_frac > 0.25:
        notes.append(f"About {foil_frac:.0%} of the card is foil/holo, where the surface can't be judged "
                     "from photos - check those areas by tilting the card under a light.")
    return SurfaceScan(defects, _severity(defects), len(reliefs), rejected, foil_frac, vis, notes)


def annotate(img: np.ndarray, sc: SurfaceScan, ppm: float) -> np.ndarray:
    out = img.copy()
    t = max(2, int(round(0.12 * ppm)))
    pad = int(round(0.5 * ppm))
    colors = {"scratch": (255, 120, 0), "crease": (255, 0, 160), "dent": (0, 200, 255), "mark": (0, 200, 255)}
    for d in sc.defects:
        cv2.rectangle(out, (d.x - pad, d.y - pad), (d.x + d.w + pad, d.y + d.h + pad), colors[d.kind], t)
    return out
