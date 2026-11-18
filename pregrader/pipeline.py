"""Glue between the vision steps, shared by the GUI and the command line."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import centering, defects, geometry, quality, standards, surface
from .centering import Centering, Lines


def _load_with_pillow(path: str) -> Optional[np.ndarray]:
    """HEIC/HEIF (iPhone) and anything else Pillow can read, with EXIF rotation."""
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001 - fall through to the error below
        return None


def load_image(path: str) -> np.ndarray:
    img = None
    if not path.lower().endswith((".heic", ".heif")):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # applies EXIF orientation
        if img is None and os.path.exists(path):
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = _load_with_pillow(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}\n"
                         "(For HEIC support: pip install pillow-heif, or export the photo as JPG.)")
    return img


@dataclass
class Side:
    name: str                       # "front" or "back"
    original: np.ndarray
    path: Optional[str] = None
    quad: Optional[np.ndarray] = None
    detected: bool = False
    rotation: int = 0               # extra 90-degree clockwise turns after warping
    img: Optional[np.ndarray] = None
    px_per_mm: float = 0.0
    lines: Optional[Lines] = None
    scan: Optional[defects.DefectScan] = None
    spread: Dict[str, float] = field(default_factory=dict)
    squareness: Dict[str, centering.SideSquareness] = field(default_factory=dict)
    borderless: bool = False        # full-art card: centering is judged by eye, not measured
    surface_photos: List[np.ndarray] = field(default_factory=list)   # angled-light shots
    surface: Optional[surface.SurfaceScan] = None
    surface_cache: dict = field(default_factory=dict, repr=False)
    quality: Optional[quality.Quality] = None

    @property
    def centering(self) -> Optional[Centering]:
        return centering.measure(self.lines, self.px_per_mm, self.spread) if self.lines else None

    @property
    def graded_centering(self) -> Optional[Centering]:
        """Centering that counts toward grades (None for a borderless card)."""
        return None if self.borderless else self.centering


def border_hint(side: Side) -> Optional[str]:
    """Suggest the 'no border' option when the measured borders look implausible."""
    if side.borderless or side.lines is None:
        return None
    c = side.centering
    mm = {k: c.mm(k) for k in centering.SIDES}
    ratios = [max(mm["left"], mm["right"]) / max(min(mm["left"], mm["right"]), 0.05),
              max(mm["top"], mm["bottom"]) / max(min(mm["top"], mm["bottom"]), 0.05)]
    weak = sum(v in ("low", "none") for v in side.lines.confidence.values())
    if min(mm.values()) < 1.0 or max(mm.values()) > 7.0 or max(ratios) > 2.5 or weak >= 2:
        return (f"{side.name.title()}: if this card has no printed border (full-art / borderless design), tick "
                "'No border' - the lines may be sitting on artwork, not a border.")
    return None


def new_side(name: str, original: np.ndarray, path: Optional[str] = None) -> Side:
    side = Side(name, original, path)
    quad = geometry.detect_card(original)
    side.detected = quad is not None
    side.quad = quad if quad is not None else geometry.default_quad(original)
    warp_side(side)
    return side


def open_side(name: str, path: str) -> Side:
    return new_side(name, load_image(path), path)


def warp_side(side: Side) -> None:
    w = geometry.warp_card(side.original, side.quad)
    img, outer = w.image, centering.Box(*w.card_box)
    for _ in range(side.rotation):
        outer = outer.rotated_cw(img.shape[0])
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    side.img, side.px_per_mm = img, w.px_per_mm
    inner, conf = centering.detect_inner(img, outer)
    side.lines = Lines(outer, inner, conf)
    side.scan = None
    reassess(side)


def reassess(side: Side) -> None:
    """Recompute what depends on the outer edge: centering spread and photo quality."""
    side.spread = centering.measure_spread(side.img, side.lines.outer)
    side.quality = quality.assess(side.img, side.lines.outer, side.px_per_mm, side.original,
                                  side.lines.confidence)
    side.squareness = centering.measure_squareness(side.img, side.lines.outer, side.px_per_mm)


def rotate_side(side: Side) -> None:
    side.lines = side.lines.rotated_cw(side.img.shape[0])
    side.img = cv2.rotate(side.img, cv2.ROTATE_90_CLOCKWISE)
    side.rotation = (side.rotation + 1) % 4
    side.scan = None
    side.spread = {"lr": side.spread.get("tb", 0.0), "tb": side.spread.get("lr", 0.0)}
    side.squareness = centering.measure_squareness(side.img, side.lines.outer, side.px_per_mm)


def auto_inner(side: Side) -> None:
    inner, conf = centering.detect_inner(side.img, side.lines.outer)
    side.lines = Lines(side.lines.outer, inner, conf)


def scan_side(side: Side, sensitivity: float = 1.0) -> defects.DefectScan:
    side.scan = defects.scan_edges(side.img, side.lines.outer, side.px_per_mm, sensitivity)
    side.surface = (surface.scan(side.img, side.lines.outer, side.px_per_mm, side.surface_photos,
                                 cache=side.surface_cache) if side.surface_photos else None)
    return side.scan


def add_surface_photo(side: Side, img: np.ndarray, sensitivity: float = 1.0) -> None:
    side.surface_photos.append(img)
    scan_side(side, sensitivity)


def severity(side: Side) -> str:
    """Worst condition problem on a side: edge whitening or surface damage."""
    return standards.worst_severity(side.scan.severity if side.scan else None,
                                    side.surface.severity if side.surface else None)


@dataclass
class Analysis:
    sides: Dict[str, Side]
    checks: List[standards.Check]
    ceilings: List[standards.Ceiling]
    severity: str
    notes: List[str] = field(default_factory=list)
    verdict: Optional["Verdict"] = None
    predictions: List[standards.Prediction] = field(default_factory=list)


@dataclass
class Verdict:
    level: str        # "send" | "maybe" | "skip"
    headline: str
    reasons: List[str]


def make_verdict(sides: Dict[str, Side], checks: List[standards.Check], severity: str) -> Verdict:
    """Should this card go in for a PSA 10? Conservative by design: anything the photo
    can't settle makes it 'maybe', never 'send'."""
    psa = next(c for c in checks if c.name == "PSA 10")
    skip, maybe, good = [], [], []
    if any(s.borderless for s in sides.values()):
        maybe.append("No printed border on " + " and ".join(s.name for s in sides.values() if s.borderless)
                     + " - centering has to be judged by eye against the design.")
    if psa.status is None:
        pass
    elif psa.status == "fail":
        skip.append("Centering misses PSA 10 limits even allowing for measurement error.")
    elif psa.status == "borderline":
        maybe.append("Centering is within measurement error of the PSA 10 limit.")
    else:
        good.append("Centering clears PSA 10 with margin.")
    w_sev = standards.worst_severity(*(s.scan.severity for s in sides.values()))
    if w_sev in ("moderate", "heavy"):
        skip.append(f"Edge/corner whitening detected ({w_sev}).")
    elif w_sev == "minor":
        maybe.append("Minor edge/corner whitening detected - check it under a light.")
    checked = [s for s in sides.values() if s.surface and s.surface.photos_used]
    s_sev = standards.worst_severity(*(s.surface.severity for s in checked))
    found = [d for s in checked for d in s.surface.defects]
    if s_sev in ("moderate", "heavy", "severe"):
        kinds = sorted({d.kind for d in found})
        skip.append(f"Surface damage found ({', '.join(kinds)}).")
    elif s_sev == "minor":
        maybe.append("A small surface mark was found - check it under a light.")
    elif checked:
        good.append("No scratches, dents or creases found on the surface.")
    verify = sum(1 for s in sides.values() for sp in s.scan.spots if sp.kind == "verify")
    if verify:
        why = "low-resolution photo" if any(s.scan.low_res for s in sides.values()) else "glitter/foil border"
        maybe.append(f"{verify} possible edge mark{'s' if verify != 1 else ''} to check by eye ({why}).")
    if any(s.scan.low_res for s in sides.values()):
        maybe.append("Photo resolution is too low to judge edge whitening - retake closer.")
    elif (not w_sev or w_sev == "none") and not verify:
        good.append("No edge whitening found.")
    if "back" not in sides:
        maybe.append("Back not photographed - PSA 10 also needs back centering (75/25) and clean back edges.")
    for s in sides.values():
        if s.quality and s.quality.level == "bad":
            maybe.append(f"{s.name.title()} photo quality is poor - retake before trusting this.")
        weak = [k for k, v in s.lines.confidence.items() if v in ("low", "none")]
        if weak:
            maybe.append(f"{s.name.title()}: confirm the {', '.join(weak)} inner line(s) by eye.")
    if skip:
        return Verdict("skip", "Unlikely to get a PSA 10", skip + maybe)
    if maybe:
        return Verdict("maybe", "Borderline - check the flagged items before sending", maybe + good)
    if not checked:
        good.append("Surface not checked - add two angled-light photos for a fuller answer.")
        return Verdict("send", "Good PSA 10 candidate (centering & edges)", good)
    return Verdict("send", "Good PSA 10 candidate (centering, edges & surface)", good)


def analyze(sides: Dict[str, Optional[Side]], sensitivity: float = 1.0) -> Analysis:
    present = {k: v for k, v in sides.items() if v is not None}
    for s in present.values():
        if s.scan is None:
            scan_side(s, sensitivity)
    f = present["front"].graded_centering if "front" in present else None
    b = present["back"].graded_centering if "back" in present else None
    sev = standards.worst_severity(*(severity(s) for s in present.values()))
    notes = []
    for s in present.values():
        if s.borderless:
            notes.append(f"{s.name.title()}: no printed border - centering isn't measured; judge it by eye.")
        hint = border_hint(s)
        if hint:
            notes.append(hint)
    if b is None:
        notes.append("Back not measured - ceilings use front centering only.")
    if f is None:
        notes.append("Front not measured - ceilings use back centering only.")
    for s in present.values():
        if not s.detected:
            notes.append(f"{s.name.title()}: card outline was placed manually.")
        weak = [k for k, v in s.lines.confidence.items() if v in ("low", "none")]
        if weak:
            notes.append(f"{s.name.title()}: weak inner-border detection on {', '.join(weak)} - check those lines.")
        if s.scan.unreliable_sides:
            notes.append(f"{s.name.title()}: white/silver border on {', '.join(s.scan.unreliable_sides)} - "
                         "whitening can't be judged there.")
        if s.scan.textured_sides:
            notes.append(f"{s.name.title()}: glitter/foil texture on {', '.join(s.scan.textured_sides)} - "
                         "edge spots there are marked 'verify by eye' and don't lower the estimate.")
        if s.quality:
            notes.extend(f"{s.name.title()} photo: {msg}" for _, msg in s.quality.issues)
        if any(sp.kind == "check-line" for sp in s.scan.spots):
            notes.append(f"{s.name.title()}: a long bright strip along an edge usually means the outer "
                         "line sits on the background - nudge it inward.")
    ceilings = [standards.grade_ceiling(c, f, b, sev) for c in standards.LADDERS]
    checks = standards.target_checks(f, b)
    return Analysis(present, checks, ceilings, sev, notes, make_verdict(present, checks, sev),
                    predictions(present, sev))


def predictions(sides: Dict[str, Side], severity: str) -> List[standards.Prediction]:
    f = sides["front"].graded_centering if "front" in sides else None
    b = sides["back"].graded_centering if "back" in sides else None
    unknown = any(sp.kind == "verify" for s in sides.values() for sp in s.scan.spots)
    why = []
    no_border = [s.name for s in sides.values() if s.borderless]
    if no_border:
        why.append(f"centering not included ({' and '.join(no_border)} has no border)")
    if "back" not in sides:
        why.append("back not photographed")
    if unknown:
        why.append("possible edge marks need checking by eye")
    if any(v in ("low", "none") for s in sides.values() for v in s.lines.confidence.values()):
        why.append("some border lines need confirming")
    if any(s.quality and s.quality.level == "warn" for s in sides.values()):
        why.append("photo has minor quality issues")
    bad = any((s.quality and s.quality.level == "bad") or s.scan.low_res for s in sides.values()) \
        or (f is None and b is None)
    if bad:
        why.insert(0, "photo quality is poor")
    has_c = f is not None or b is not None
    has_e = not all(s.scan.low_res for s in sides.values())
    has_s = any(s.surface and s.surface.photos_used for s in sides.values())
    basis = {(True, True): "centering and edges", (False, True): "edges only",
             (True, False): "centering only", (False, False): "nothing"}[(has_c, has_e)]
    if has_s and basis != "nothing":
        basis = basis.replace(" and ", ", ").replace(" only", "") + " and surface"
    elif has_s:
        basis = "surface only"
    return [standards.predict(c, f, b, severity, unknown, why, bad, basis) for c in standards.LADDERS]


def to_json(a: Analysis) -> dict:
    out: dict = {"sides": {}, "checks": [], "ceilings": [], "whitening_severity": a.severity, "notes": a.notes}
    for name, s in a.sides.items():
        c = s.centering
        out["sides"][name] = {
            "source": s.path,
            "card_corners_px": None if s.quad is None else np.round(s.quad, 2).tolist(),
            "auto_detected": s.detected,
            "px_per_mm": round(s.px_per_mm, 3),
            "outer": vars(s.lines.outer), "inner": vars(s.lines.inner),
            "inner_confidence": s.lines.confidence,
            "border_mm": {k: round(c.mm(k), 3) for k in centering.SIDES},
            "left_right": centering.ratio_text(c.lr), "top_bottom": centering.ratio_text(c.tb),
            "left_right_exact": [round(v, 2) for v in c.lr], "top_bottom_exact": [round(v, 2) for v in c.tb],
            "uncertainty_1sd_points": {"left_right": round(c.sigma_lr, 2), "top_bottom": round(c.sigma_tb, 2)},
            "photo_quality": None if s.quality is None else {
                "level": s.quality.level, "issues": [m for _, m in s.quality.issues],
                "metrics": {k: round(v, 3) for k, v in s.quality.metrics.items()}},
            "whitening": {
                "severity": s.scan.severity,
                "unreliable_sides": s.scan.unreliable_sides,
                "textured_sides": s.scan.textured_sides,
                "spots": [vars(sp) for sp in s.scan.spots],
            },
            "surface": None if not s.surface else {
                "photos_used": s.surface.photos_used, "photos_rejected": s.surface.photos_rejected,
                "severity": s.surface.severity, "foil_fraction": round(s.surface.foil_fraction, 3),
                "defects": [vars(d) for d in s.surface.defects], "notes": s.surface.notes},
        }
    for ch in a.checks:
        out["checks"].append({"target": ch.name, "front_limit": ch.front_limit, "back_limit": ch.back_limit,
                              "front_worst": ch.front_value, "back_worst": ch.back_value,
                              "status": ch.status})
    if a.verdict:
        out["verdict"] = vars(a.verdict)
    for ce in a.ceilings:
        out["ceilings"].append(vars(ce))
    out["predictions"] = [vars(p) for p in a.predictions]
    return out
