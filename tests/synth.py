"""Synthetic card photos with known ground truth, for tests and demos."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from pregrader.geometry import CARD_H_MM, CARD_W_MM, CORNER_RADIUS_MM


def make_card(ppm: float = 14.0, borders_mm: Sequence[float] = (3.6, 2.4, 3.0, 3.0),
              border_bgr=(160, 70, 20), frame_bgr=(20, 20, 20), sparkle: int = 0, art_tilt_deg: float = 0.0, nicks: Sequence[Tuple[str, float, float]] = (),
              seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """borders_mm = (left, right, top, bottom). nicks = (side, position 0..1, radius mm).
    Returns (card BGR, alpha mask)."""
    rng = np.random.default_rng(seed)
    W, H = int(round(CARD_W_MM * ppm)), int(round(CARD_H_MM * ppm))
    card = np.zeros((H, W, 3), np.uint8)
    card[:] = border_bgr
    l, r, t, b = (int(round(v * ppm)) for v in borders_mm)
    aw, ah = W - l - r, H - t - b
    yy, xx = np.mgrid[0:ah, 0:aw]
    art = np.stack([90 + 120 * xx / aw, 60 + 150 * yy / ah, 200 - 120 * xx / aw], axis=2)
    art = art.astype(np.uint8)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(art, (int(rng.integers(0, aw)), int(rng.integers(0, ah))), int(rng.integers(5, aw // 5)), c, -1)
    cv2.rectangle(art, (0, 0), (aw - 1, ah - 1), frame_bgr, max(2, int(0.3 * ppm)))
    card[t:t + ah, l:l + aw] = art
    if art_tilt_deg:  # print rotated relative to the cut (tilted / diamond cut)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), art_tilt_deg, 1.0)
        card = cv2.warpAffine(card, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderValue=tuple(int(v) for v in border_bgr))
    if sparkle:  # glitter flakes all over the border (Celebrations-style gold borders)
        border = np.ones((H, W), bool)
        border[t:t + ah, l:l + aw] = False
        ys, xs = np.nonzero(border)
        pick = rng.choice(len(ys), sparkle, replace=False)
        for y, x in zip(ys[pick], xs[pick]):
            v = int(rng.integers(200, 250))
            cv2.circle(card, (int(x), int(y)), int(rng.integers(1, 3)), (v, v, v - 10), -1)
    for side, pos, rad_mm in nicks:
        rad = rad_mm * ppm
        if side == "top":
            c = (pos * W, 0)
        elif side == "bottom":
            c = (pos * W, H - 1)
        elif side == "left":
            c = (0, pos * H)
        else:
            c = (W - 1, pos * H)
        cv2.ellipse(card, (int(c[0]), int(c[1])), (int(rad * 1.6), int(rad)) if side in ("top", "bottom")
                    else (int(rad), int(rad * 1.6)), 0, 0, 360, (245, 245, 245), -1)
    mask = np.zeros((H, W), np.uint8)
    rad = int(round(CORNER_RADIUS_MM * ppm))
    cv2.rectangle(mask, (rad, 0), (W - 1 - rad, H - 1), 255, -1)
    cv2.rectangle(mask, (0, rad), (W - 1, H - 1 - rad), 255, -1)
    for cx, cy in ((rad, rad), (W - 1 - rad, rad), (W - 1 - rad, H - 1 - rad), (rad, H - 1 - rad)):
        cv2.circle(mask, (cx, cy), rad, 255, -1, cv2.LINE_AA)
    return card, mask


def weave(shape, period: int = 16, seed: int = 3) -> np.ndarray:
    """Grey woven-cloth texture: alternating light/dark threads with random shading."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cell = ((yy // period) + (xx // period)) % 2
    thread_h = 0.5 + 0.5 * np.cos(2 * np.pi * yy / (period / 2))
    thread_v = 0.5 + 0.5 * np.cos(2 * np.pi * xx / (period / 2))
    v = np.where(cell == 0, thread_h, thread_v)
    shade = cv2.resize(rng.uniform(0.75, 1.15, (h // 200 + 2, w // 200 + 2)), (w, h))
    L = (45 + 120 * v) * shade
    return np.repeat(np.clip(L, 0, 255)[..., None], 3, axis=2).astype(np.float32)


def place(card: np.ndarray, mask: np.ndarray, dst_quad, canvas=(3000, 4000),
          bg_bgr=(30, 30, 30), noise: float = 3.0, seed: int = 1, bg_image=None) -> np.ndarray:
    """Composite the card onto a background so its edges land exactly on dst_quad."""
    rng = np.random.default_rng(seed)
    H, W = card.shape[:2]
    src = np.float32([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]])
    M = cv2.getPerspectiveTransform(src, np.float32(dst_quad))
    ch, cw = canvas
    warped = cv2.warpPerspective(card, M, (cw, ch), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(mask, M, (cw, ch), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    if bg_image is not None:
        bg = bg_image.astype(np.float32)
    else:
        bg = np.empty((ch, cw, 3), np.float32)
        bg[:] = bg_bgr
    out = warped.astype(np.float32) * alpha + bg * (1 - alpha)
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def tilted_quad(canvas=(3000, 4000), height=2300.0, angle_deg=7.0, keystone=0.04) -> np.ndarray:
    ch, cw = canvas
    w = height * CARD_W_MM / CARD_H_MM
    pts = np.array([[-w / 2, -height / 2], [w / 2, -height / 2], [w / 2, height / 2], [-w / 2, height / 2]])
    pts[0, 0] += keystone * w; pts[1, 0] -= keystone * w  # top narrower: camera tilt
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = np.einsum("ij,nj->ni", R, pts)
    return rot + np.array([cw / 2 + 60, ch / 2 - 40])


def photo(borders_mm=(3.6, 2.4, 3.0, 3.0), nicks: List = (), **kw):
    card, mask = make_card(borders_mm=borders_mm, nicks=nicks, **kw)
    q = tilted_quad()
    return place(card, mask, q), q


def surface_photos(defects=(), foil_box=None, ppm: float = 20.0, seed: int = 5,
                   lights=((0.94, 0.0, 0.34),)):
    """A card photographed once with even overhead light and once per entry in `lights`
    with light skimming in from a low angle. `defects`: ("scratch", (x0,y0,x1,y1) mm,
    depth px) | ("dent", (cx,cy) mm, radius mm) | ("crease", (x0,y0,x1,y1) mm).
    Returns (main photo, [angled photos], main quad)."""
    rng = np.random.default_rng(seed)
    card, mask = make_card(ppm=ppm, seed=seed)
    H, W = card.shape[:2]
    h = np.zeros((H, W), np.float32)
    for d in defects:
        if d[0] in ("scratch", "crease"):
            x0, y0, x1, y1 = (int(v * ppm) for v in d[1])
            depth = d[2] if len(d) > 2 else (2.0 if d[0] == "scratch" else 6.0)
            big = np.zeros((H * 4, W * 4), np.float32)
            cv2.line(big, (x0 * 4, y0 * 4), (x1 * 4, y1 * 4), 1.0, 6 if d[0] == "scratch" else 12, cv2.LINE_AA)
            line = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
            h -= depth * cv2.GaussianBlur(line, (0, 0), 0.8 if d[0] == "scratch" else 2.0)
        elif d[0] == "dent":
            cx, cy = d[1][0] * ppm, d[1][1] * ppm
            r = d[2] * ppm
            yy, xx = np.mgrid[0:H, 0:W]
            h -= 4.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r / 2) ** 2))
    gy, gx = np.gradient(h)
    n = np.dstack([-gx, -gy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)

    def shade(L, falloff):
        L = np.array(L, np.float32) / np.linalg.norm(L)
        s = np.clip(np.einsum("ijk,k->ij", n, L), 0, None) / L[2]
        if falloff:
            ramp = np.linspace(1.25, 0.75, W)[None, :]
            s = s * ramp
        return np.clip(0.15 + 0.85 * s, 0, 3)

    def photo(L, falloff, foil, exposure, quad_angle, noise_seed):
        img = card.astype(np.float32) * shade(L, falloff)[..., None] * exposure
        if foil and foil_box is not None:
            x0, y0, x1, y1 = (int(v * ppm) for v in foil_box)
            field = cv2.resize(rng.uniform(0.5, 1.6, (12, 12, 3)).astype(np.float32), (x1 - x0, y1 - y0),
                               interpolation=cv2.INTER_CUBIC)
            fine = rng.uniform(0.7, 1.3, (y1 - y0, x1 - x0, 3)).astype(np.float32)
            img[y0:y1, x0:x1] *= field * cv2.GaussianBlur(fine, (0, 0), 1.2)
        img = np.clip(img, 0, 255).astype(np.uint8)
        q = tilted_quad(angle_deg=quad_angle, height=2300.0)
        return place(img, mask, q, seed=noise_seed), q

    main, q = photo((0.0, 0.0, 1.0), False, False, 1.0, 7.0, 1)
    angled = [photo(L, True, True, 0.9, 7.0 + 1.5 * (i + 1), 10 + i)[0] for i, L in enumerate(lights)]
    return main, angled, q
