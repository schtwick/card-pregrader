"""Bridge for the browser version (runs inside Pyodide). Everything in and out is
plain JSON-able data so the page's JavaScript can use it directly."""
from __future__ import annotations

import base64
import io
from typing import Dict, Optional

import cv2
import numpy as np

from . import centering, findings, pipeline, report, standards

_sides: Dict[str, Optional[pipeline.Side]] = {"front": None, "back": None}
_sensitivity = 1.0
DISPLAY_LONG = 1100  # px on the long side of the image sent back for display


def _jpeg_b64(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def load(name: str, data: bytes) -> dict:
    """Decode a photo, find + flatten the card, measure it. Returns the side's view."""
    img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Couldn't read that image. Try a JPG or PNG.")
    side = pipeline.new_side(name, img)
    pipeline.scan_side(side, _sensitivity)
    _sides[name] = side
    return side_view(name)


def clear(name: str) -> None:
    _sides[name] = None


def side_view(name: str) -> dict:
    s = _sides[name]
    H, W = s.img.shape[:2]
    scale = DISPLAY_LONG / max(H, W)
    small = cv2.resize(s.img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    c = s.centering
    return {
        "name": name,
        "image": _jpeg_b64(small),
        "width": W, "height": H, "display_scale": scale,
        "px_per_mm": s.px_per_mm,
        "detected": s.detected,
        "borderless": s.borderless,
        "outer": vars(s.lines.outer), "inner": vars(s.lines.inner),
        "confidence": s.lines.confidence,
        "spots": [vars(sp) for sp in s.scan.spots],
        "centering": {
            "lr": c.lr, "tb": c.tb, "sigma_lr": c.sigma_lr, "sigma_tb": c.sigma_tb,
            "mm": {k: c.mm(k) for k in centering.SIDES},
            "lr_text": centering.ratio_text(c.lr), "tb_text": centering.ratio_text(c.tb),
        },
        "quality": [{"level": lv, "text": t} for lv, t in (s.quality.issues if s.quality else [])],
    }


def set_lines(name: str, outer: dict, inner: dict, borderless: bool, rescan: bool) -> dict:
    """Apply the user's line edits. rescan=True after an outer line moved (re-checks
    edges, photo quality and the spread)."""
    s = _sides[name]
    outer, inner = dict(outer), dict(inner)
    for k in centering.SIDES:
        s.lines.outer.set(k, outer[k])
        if abs(s.lines.inner.get(k) - inner[k]) > 1e-6:
            s.lines.inner.set(k, inner[k])
            s.lines.confidence[k] = "manual"
    s.borderless = bool(borderless)
    if rescan:
        pipeline.reassess(s)
        pipeline.scan_side(s, _sensitivity)
    return side_view(name)


def rotate(name: str) -> dict:
    s = _sides[name]
    pipeline.rotate_side(s)
    pipeline.scan_side(s, _sensitivity)
    return side_view(name)


def auto_borders(name: str) -> dict:
    s = _sides[name]
    pipeline.auto_inner(s)
    return side_view(name)


def corners(name: str, size: int = 360) -> dict:
    s = _sides[name]
    from . import defects
    img = defects.annotate(s.img, s.scan, s.px_per_mm) if s.scan else s.img
    crops = defects.corner_crops(img, s.lines.outer, s.px_per_mm, size_mm=5.0)
    return {k: _jpeg_b64(cv2.resize(v, (size, size), interpolation=cv2.INTER_CUBIC), 90)
            for k, v in crops.items()}


def summary() -> dict:
    """Verdict, predictions, target checks and plain-English findings for all sides."""
    present = {k: v for k, v in _sides.items() if v is not None}
    if not present:
        return {}
    a = pipeline.analyze(_sides, _sensitivity)
    items = findings.build(a)
    return {
        "verdict": vars(a.verdict) if a.verdict else None,
        "predictions": [{
            "company": p.company, "label": p.label, "probability": p.probability,
            "distribution": p.distribution, "ceiling": p.ceiling, "if_marks_real": p.if_marks_real,
            "confidence": p.confidence, "confidence_why": p.confidence_why, "basis": p.basis,
            "available": p.available, "has_odds": p.has_odds,
        } for p in a.predictions],
        "checks": [{"name": ch.name, "status": ch.status} for ch in a.checks],
        "ceilings": [{"company": c.company, "label": c.label} for c in a.ceilings],
        "findings": [{"kind": f.kind, "label": findings.LABEL[f.kind], "topic": f.topic, "text": f.text}
                     for f in items],
        "text": findings.as_text(a, items),
    }


def report_png() -> str:
    a = pipeline.analyze(_sides, _sensitivity)
    img = report.render(a)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
