"""Surface check (scratches, dents, creases) from simulated angled-light photos."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_card, place, surface_photos, tilted_quad  # noqa: E402

from pregrader import findings, pipeline, surface  # noqa: E402

TWO_LIGHTS = ((0.94, 0.0, 0.34), (0.0, 0.94, 0.34))   # from the side, then from the top


def _scan(defects=(), foil=None, seed=5, lights=TWO_LIGHTS):
    main, angled, _ = surface_photos(defects, foil, lights=lights, seed=seed)
    side = pipeline.new_side("front", main)
    for p in angled:
        side.surface_photos.append(p)
    pipeline.scan_side(side)
    return side


def _near(d, side, x_mm, y_mm, tol_mm=4.0):
    o = side.lines.outer
    cx = o.left + x_mm * (o.right - o.left) / 63.5
    cy = o.top + y_mm * (o.bottom - o.top) / 88.9
    return abs(d.center[0] - cx) < d.w / 2 + tol_mm * side.px_per_mm and \
        abs(d.center[1] - cy) < d.h / 2 + tol_mm * side.px_per_mm


@pytest.mark.parametrize("seed", [5, 9, 13])
def test_clean_card_has_no_surface_defects(seed):
    side = _scan(seed=seed)
    assert side.surface.photos_used == 2
    assert side.surface.defects == [] and side.surface.severity == "none"


def test_scratch_found_with_length_and_place():
    side = _scan((("scratch", (15, 30, 23, 36)),))
    scratches = [d for d in side.surface.defects if d.kind == "scratch"]
    assert len(scratches) == 1
    assert 8 < scratches[0].size_mm < 13            # true length 10 mm
    assert _near(scratches[0], side, 19, 33)


def test_crease_needs_light_across_it():
    # A left-to-right crease barely shows with light from the side only...
    side_only = _scan((("crease", (5, 70, 58, 72)),), lights=((0.94, 0.0, 0.34),))
    assert not any(d.kind == "crease" for d in side_only.surface.defects)
    # ...and is found once a photo has the light coming from the top.
    both = _scan((("crease", (5, 70, 58, 72)),))
    creases = [d for d in both.surface.defects if d.kind == "crease"]
    assert len(creases) == 1 and creases[0].size_mm > 45
    assert both.surface.severity == "severe"


def test_dent_is_flagged():
    side = _scan((("dent", (40, 60), 1.0),))
    assert side.surface.defects and all(d.kind in ("dent", "mark") for d in side.surface.defects)
    assert any(_near(d, side, 40, 60) for d in side.surface.defects)


def test_foil_area_is_not_judged_and_raises_no_false_defects():
    side = _scan(foil=(12, 15, 52, 45))
    assert side.surface.defects == []
    assert side.surface.foil_fraction > 0.15


def test_unrelated_photo_is_rejected():
    main, angled, _ = surface_photos(seed=5)
    side = pipeline.new_side("front", main)
    other, mask = make_card(seed=99, border_bgr=(30, 160, 30))
    side.surface_photos.append(place(other, mask, tilted_quad(angle_deg=-3), bg_bgr=(30, 30, 30)))
    pipeline.scan_side(side)
    assert side.surface.photos_used == 0 and side.surface.photos_rejected == 1
    assert any("couldn't be lined up" in n for n in side.surface.notes)


def test_surface_feeds_verdict_prediction_and_findings():
    side = _scan((("crease", (5, 70, 58, 72)),))
    a = pipeline.analyze({"front": side, "back": None})
    assert a.verdict.level == "skip" and any("Surface damage" in r for r in a.verdict.reasons)
    psa = next(p for p in a.predictions if p.company == "PSA")
    assert "surface" in psa.basis and psa.label != "10"
    texts = [f.text for f in findings.build(a) if f.topic == "Surface"]
    assert any("crease" in t for t in texts)


def test_no_surface_photos_says_not_checked():
    side = pipeline.new_side("front", surface_photos(seed=5)[0])
    a = pipeline.analyze({"front": side, "back": None})
    assert any(f.topic == "Surface" and "not checked" in f.text for f in findings.build(a))
    assert all("surface" not in p.basis for p in a.predictions)


def test_crease_caps_grades_hard():
    from pregrader import standards
    assert standards.grade_ceiling("PSA", None, None, "severe").label == "5"
    assert standards.grade_ceiling("BGS", None, None, "severe").label.startswith("below")
    side = _scan((("crease", (5, 70, 58, 72)),))
    psa = next(p for p in pipeline.analyze({"front": side, "back": None}).predictions if p.company == "PSA")
    assert psa.label in ("5", "3") or psa.label.startswith("below")
