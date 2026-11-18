"""Card detection and perspective correction.

Coordinates use OpenCV's pixel-centre convention: pixel (i, j) has its centre at
(i, j), so an edge that lies between two pixels sits on a half-integer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

CARD_W_MM = 63.5
CARD_H_MM = 88.9
CORNER_RADIUS_MM = 3.18
CARD_ASPECT = CARD_W_MM / CARD_H_MM
BUSY_BACKGROUND = 8.0  # border texture SD; plain mats/tables measure ~1-3, cloth 15-30

Line = Tuple[np.ndarray, np.ndarray]  # (point, unit direction)


@dataclass
class Warp:
    image: np.ndarray        # warped BGR image, card plus a background margin
    quad: np.ndarray         # source corners TL, TR, BR, BL (4x2 float)
    homography: np.ndarray   # source -> warped
    px_per_mm: float
    card_box: Tuple[float, float, float, float]  # left, top, right, bottom


def _odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 else n + 1


def order_quad(pts) -> np.ndarray:
    """Order 4 points clockwise (image coordinates) starting at top-left."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def _border_median_lab(lab: np.ndarray) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def _candidate_masks(small: np.ndarray) -> List[np.ndarray]:
    """Several independent foreground segmentations; the best contour wins."""
    h, w = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _border_median_lab(lab)
    dist = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist8 = cv2.GaussianBlur(dist8, (5, 5), 0)
    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)

    masks = []
    otsu = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    masks.append(cv2.threshold(dist8, 0, 255, otsu)[1])
    m = cv2.threshold(gray, 0, 255, otsu)[1]
    masks += [m, cv2.bitwise_not(m)]
    masks.append(cv2.threshold(sat, 0, 255, otsu)[1])

    # Texture: a woven/patterned background (cloth, carpet, wood grain) is busy while
    # a card's print and foil are mostly smooth, even where their colours match
    # (silver foil on grey cloth). Only used when the photo's border is actually busy.
    L = lab[..., 0]
    mu = cv2.blur(L, (5, 5))
    sd = cv2.blur(np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mu * mu, 0)), (7, 7))
    b = max(4, int(0.04 * min(h, w)))
    frame = np.concatenate([sd[:b].ravel(), sd[-b:].ravel(), sd[:, :b].ravel(), sd[:, -b:].ravel()])
    bg_busy = float(np.median(frame))
    if bg_busy > 15:
        smooth = (sd < 0.7 * bg_busy).astype(np.uint8) * 255
        smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        filled = np.zeros_like(smooth)
        cnts, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, 255, -1)   # text and logos inside the card
        masks.append(filled)

    edges = cv2.dilate(cv2.Canny(gray, 20, 60), np.ones((3, 3), np.uint8), iterations=2)
    filled = np.zeros_like(gray)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    masks.append(filled)

    k = _odd(max(3, 0.008 * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    for m in masks:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        out.append(m)
    return out


def _score_contour(cnt: np.ndarray, shape: Tuple[int, int]) -> float:
    H, W = shape
    img_area = float(H * W)
    area = cv2.contourArea(cnt)
    if area < 0.04 * img_area or area > 0.97 * img_area:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
    if rw < 2 or rh < 2:
        return 0.0
    rectangularity = hull_area / (rw * rh)
    solidity = area / max(hull_area, 1.0)
    aspect = min(rw, rh) / max(rw, rh)
    aspect_term = float(np.exp(-abs(aspect - CARD_ASPECT) * 5))  # tolerant of perspective
    x, y, w, h = cv2.boundingRect(cnt)
    touches = int(x <= 1) + int(y <= 1) + int(x + w >= W - 1) + int(y + h >= H - 1)
    return (rectangularity ** 4) * (solidity ** 2) * aspect_term * (0.3 ** touches) * (area / img_area) ** 0.3


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    q = None
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            q = order_quad(approx)
            break
    if q is None:
        q = order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))
    # approxPolyDP vertices sit on the rounded corners; fit straight lines to the
    # middle of each side and intersect them to get the virtual sharp corners.
    pts = cnt.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        d = p1 - p0
        length = float(np.hypot(*d))
        d = d / length
        rel = pts - p0
        t = rel[:, 0] * d[0] + rel[:, 1] * d[1]
        dist = np.abs(rel[:, 0] * -d[1] + rel[:, 1] * d[0])
        sel = pts[(t > 0.15 * length) & (t < 0.85 * length) & (dist < 0.03 * length)]
        if len(sel) < 5:
            return q
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    try:
        return np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    except np.linalg.LinAlgError:
        return q


def _fit_side(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, search: float,
              n: int = 90) -> Optional[Line]:
    """Locate the card edge along one side with sub-pixel profiles, then fit a line.

    Samples stay in the middle 80% of the side so rounded corners don't bias the fit.
    """
    d = p1 - p0
    length = float(np.hypot(*d))
    d = d / length
    nrm = np.array([-d[1], d[0]])
    step = 0.5
    offs = np.arange(-search, search + step / 2, step)
    base = p0 + np.linspace(0.1, 0.9, n)[:, None] * length * d
    pts = base[:, None, :] + offs[None, :, None] * nrm
    samp = cv2.remap(img, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lab = cv2.cvtColor(samp, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (5, 1), 0)
    g = np.sqrt(((lab[:, 4:] - lab[:, :-4]) ** 2).sum(axis=2))  # colour change over 2 px
    j = np.argmax(g, axis=1)
    rows = np.arange(n)
    strength = g[rows, j]
    jl, jr = np.clip(j - 1, 0, g.shape[1] - 1), np.clip(j + 1, 0, g.shape[1] - 1)
    a, b, c = g[rows, jl], strength, g[rows, jr]
    denom = a - 2 * b + c
    delta = np.where((j > 0) & (j < g.shape[1] - 1) & (np.abs(denom) > 1e-6), 0.5 * (a - c) / denom, 0.0)
    off = offs[0] + (j + 2 + np.clip(delta, -1, 1)) * step

    keep = (strength > 0.5 * np.median(strength)) & (strength > 8)
    if keep.sum() < 10:
        return None
    edge = (base[keep] + off[keep, None] * nrm).astype(np.float32)

    def fit(pts_):
        vx, vy, x0, y0 = cv2.fitLine(pts_, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array([x0, y0], float), np.array([vx, vy], float)

    p, v = fit(edge)
    rel = edge - p
    res = np.abs(rel[:, 0] * -v[1] + rel[:, 1] * v[0])
    inl = res < max(1.0, 3 * 1.4826 * float(np.median(res)))
    if inl.sum() >= 10:
        p, v = fit(edge[inl])
    return p, v


def _intersect(l1: Line, l2: Line) -> np.ndarray:
    (p, d), (q, e) = l1, l2
    A = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    t, _ = np.linalg.solve(A, q - p)
    return p + t * d


def refine_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Snap each side to the true edge at full resolution; corners become the
    intersections of the fitted sides (the 'virtual' sharp corners)."""
    q = order_quad(quad)
    for frac in (0.012, 0.005):
        lines = []
        for i in range(4):
            p0, p1 = q[i], q[(i + 1) % 4]
            search = max(6.0, frac * float(np.hypot(*(p1 - p0))))
            line = _fit_side(img, p0, p1, search)
            if line is None:
                d = p1 - p0
                line = (p0, d / np.hypot(*d))
            lines.append(line)
        q = np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    return q


def background_busyness(img: np.ndarray) -> float:
    """Median local brightness SD (5x5) around the photo's border: high for cloth,
    carpet or wood grain, low for a plain table or mat."""
    h, w = img.shape[:2]
    s = min(1.0, 1000 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    mu = cv2.blur(L, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mu * mu, 0))
    b = max(4, int(0.04 * min(L.shape)))
    return float(np.median(np.concatenate([sd[:b].ravel(), sd[-b:].ravel(),
                                           sd[:, :b].ravel(), sd[:, -b:].ravel()])))


def _texture_side(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, search: float,
                  n: int = 60) -> Optional[Line]:
    """On a busy background: along many lines across the side, coming in from
    outside, the card edge is where the busy texture stops and stays stopped."""
    d = p1 - p0
    length = float(np.hypot(*d))
    d = d / length
    nrm = np.array([-d[1], d[0]])
    offs = np.arange(-search, search + 1.0, 1.0)
    base = p0 + np.linspace(0.1, 0.9, n)[:, None] * length * d
    grid = base[:, None, :] + offs[None, :, None] * nrm
    samp = cv2.remap(img, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    L = cv2.cvtColor(samp, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    # 2-D local SD (offsets x neighbouring positions) so a thread running parallel to
    # the edge still reads as texture
    win = max(5, int(round(0.006 * length)) | 1)
    mu = cv2.blur(L, (win, 3))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (win, 3)) - mu * mu, 0))
    run = max(6, int(0.004 * length))
    pts = []
    for r in range(n):
        row = sd[r]
        outside = float(np.median(row[: int(0.25 * len(row))]))
        if outside < 10:
            continue
        calm = row < 0.45 * outside
        for j in range(len(row) - run):
            if calm[j:j + run].all():
                pts.append(base[r] + (offs[j] - win / 2 + 0.5) * nrm)
                break
    if len(pts) < 12:
        return None
    pts = np.array(pts, np.float32)

    def fit(p_):
        vx, vy, x0, y0 = cv2.fitLine(p_, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array([x0, y0], float), np.array([vx, vy], float)

    p, v = fit(pts)
    for _ in range(2):
        rel = pts - p
        res = np.abs(rel[:, 0] * -v[1] + rel[:, 1] * v[0])
        inl = res < max(3.0, 2.5 * 1.4826 * float(np.median(res)))
        if inl.sum() < 12:
            break
        p, v = fit(pts[inl])
    return p, v


def texture_snap(img: np.ndarray, quad: np.ndarray, search_frac: float = 0.1) -> Optional[np.ndarray]:
    """Re-find all four sides by where a busy background turns into smooth card."""
    q = order_quad(quad)
    short = min(np.hypot(*(q[1] - q[0])), np.hypot(*(q[3] - q[0])))
    lines = []
    for i in range(4):
        line = _texture_side(img, q[i], q[(i + 1) % 4], search_frac * short)
        if line is None:
            return None
        lines.append(line)
    try:
        return np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    except np.linalg.LinAlgError:
        return None


def _quad_inconsistency(q: np.ndarray) -> float:
    """0 for a perfect card shape; grows with non-parallel opposite sides and a
    width/height ratio away from 63.5 x 88.9 mm."""
    def ang(a, b):
        d = b - a
        return np.arctan2(d[1], d[0])
    def diff(a1, a2):  # angle between two lines, ignoring direction
        x = abs((a1 - a2 + np.pi / 2) % np.pi - np.pi / 2)
        return np.degrees(x)
    par = diff(ang(q[0], q[1]), ang(q[3], q[2])) + diff(ang(q[0], q[3]), ang(q[1], q[2]))
    top, bottom = np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3]))
    left, right = np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1]))
    w, h = (top + bottom) / 2, (left + right) / 2
    aspect = min(w, h) / max(w, h)
    return par / 2 + 100 * abs(aspect - CARD_ASPECT) / CARD_ASPECT / 2


def _edge_support(img: np.ndarray, p: np.ndarray, d: np.ndarray, length: float, n: int = 160) -> float:
    """Mean colour-change across a straight line (point p, unit direction d)."""
    nrm = np.array([-d[1], d[0]])
    ts = np.linspace(0.1, 0.9, n) * length
    base = p + ts[:, None] * d
    pts = base[:, None, :] + np.array([-2.0, 2.0])[None, :, None] * nrm
    samp = cv2.remap(img, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lab = cv2.cvtColor(samp, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.sqrt(((lab[:, 1] - lab[:, 0]) ** 2).sum(axis=1))))


def _refind_side(img: np.ndarray, q: np.ndarray, i: int) -> Optional[np.ndarray]:
    """Re-find side i (q[i] -> q[i+1]) as a line parallel to the opposite side, at the
    distance a real card would have, picking the strongest full-length edge there."""
    j0, j1 = (i + 2) % 4, (i + 3) % 4          # opposite side, running the same way
    o0, o1 = q[j1], q[j0]
    d = o1 - o0
    length = float(np.hypot(*d))
    d = d / length
    nrm = np.array([-d[1], d[0]])
    # which way is the card from the opposite side?
    centre = q.mean(axis=0)
    if np.dot(centre - o0, nrm) < 0:
        nrm = -nrm
    # expected distance: opposite side length x card aspect (or its inverse)
    long_side = length > 0.5 * (np.hypot(*(q[(i + 1) % 4] - q[(i + 2) % 4])) + np.hypot(*(q[i] - q[(i + 3) % 4])))
    expect = length * (CARD_ASPECT if long_side else 1 / CARD_ASPECT)
    best, best_s = None, 0.0
    for dist in np.arange(0.9 * expect, 1.1 * expect, 1.0):
        s = _edge_support(img, o0 + dist * nrm, d, length)
        if s > best_s:
            best, best_s = dist, s
    if best is None:
        return None
    line_new = (o0 + best * nrm, d)
    lines = []
    for k in range(4):
        if k == i:
            lines.append(line_new)
        else:
            a, b = q[k], q[(k + 1) % 4]
            dd = b - a
            lines.append((a, dd / np.hypot(*dd)))
    try:
        return np.array([_intersect(lines[k - 1], lines[k]) for k in range(4)])
    except np.linalg.LinAlgError:
        return None


def fix_inconsistent(img: np.ndarray, quad: np.ndarray, tol: float = 6.0) -> np.ndarray:
    """If the outline is clearly not card-shaped (one side lost on a low-contrast edge),
    re-find the one side whose replacement makes it most card-shaped. Normal camera
    perspective scores ~1-3, so only scores above `tol` are touched, and a fix must
    at least halve the score."""
    q = order_quad(quad)
    bad = _quad_inconsistency(q)
    if bad <= tol:
        return q
    best, best_bad = q, bad
    for i in range(4):
        cand = _refind_side(img, q, i)
        if cand is None or not cv2.isContourConvex(cand.astype(np.float32).reshape(-1, 1, 2)):
            continue
        cb = _quad_inconsistency(order_quad(cand))
        if cb < best_bad and cb < 0.5 * bad:
            best, best_bad = order_quad(cand), cb
    return best


def detect_card(img: np.ndarray, work_size: int = 1000) -> Optional[np.ndarray]:
    """Return the card's corners (TL, TR, BR, BL) in source pixels, or None."""
    h, w = img.shape[:2]
    s = min(1.0, work_size / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    best, best_score = None, 0.0
    for m in _candidate_masks(small):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            sc = _score_contour(c, small.shape[:2])
            if sc > best_score:
                best, best_score = c, sc
    if best is None or best_score < 0.05:
        return None
    coarse = _quad_from_contour(best) / s
    # Busy background (cloth, carpet): colour-based outlines can lose a border that
    # matches it, and thread edges confuse gradient fitting. Find the sides by where
    # the texture stops instead - twice, the second time with a tighter search.
    if background_busyness(img) > BUSY_BACKGROUND:
        snapped = texture_snap(img, coarse)
        if snapped is not None and cv2.isContourConvex(snapped.astype(np.float32).reshape(-1, 1, 2)):
            again = texture_snap(img, snapped, search_frac=0.03)
            coarse = again if again is not None else snapped
            return coarse
    coarse = fix_inconsistent(img, coarse)
    try:
        fine = refine_quad(img, coarse)
    except np.linalg.LinAlgError:
        return coarse
    diag = float(np.hypot(*(coarse[2] - coarse[0])))
    if (not np.all(np.isfinite(fine))
            or np.max(np.hypot(*(fine - coarse).T)) > 0.05 * diag
            or not cv2.isContourConvex(fine.astype(np.float32).reshape(-1, 1, 2))):
        return coarse
    return fine


def warp_card(img: np.ndarray, quad: np.ndarray, long_px: Optional[int] = None,
              margin_mm: float = 3.0) -> Warp:
    """Warp the card flat to the standard 63.5 x 88.9 mm aspect, keeping a small
    background margin around it so the outer edge stays visible and adjustable.

    The output keeps the photo's orientation (portrait or landscape)."""
    q = order_quad(quad)
    top, bottom = np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3]))
    left, right = np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1]))
    w_m, h_m = (top + bottom) / 2, (left + right) / 2
    landscape = w_m > h_m
    if long_px is None:  # keep roughly the native resolution
        long_px = int(np.clip(max(w_m, h_m), 1000, 3000))
    short_px = int(round(long_px * CARD_ASPECT))
    cw, ch = (long_px, short_px) if landscape else (short_px, long_px)
    ppm = long_px / CARD_H_MM
    pad = int(round(margin_mm * ppm))
    dst = np.float32([[pad, pad], [pad + cw, pad], [pad + cw, pad + ch], [pad, pad + ch]])
    H = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    out = cv2.warpPerspective(img, H, (cw + 2 * pad, ch + 2 * pad), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return Warp(out, q, H, ppm, (float(pad), float(pad), float(pad + cw), float(pad + ch)))


def default_quad(img: np.ndarray) -> np.ndarray:
    """Placeholder corners for manual placement when detection fails."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    hh = 0.4 * h
    hw = min(0.4 * w, hh * CARD_ASPECT)
    return np.array([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
