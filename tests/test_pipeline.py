import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_card, place, tilted_quad  # noqa: E402

from pregrader import centering, defects, geometry, pipeline, report, standards  # noqa: E402

CASES = {
    "blue-on-dark": dict(border_bgr=(160, 70, 20), bg=(30, 30, 30)),
    "yellow-on-dark": dict(border_bgr=(20, 210, 245), bg=(40, 35, 30)),
    "black-on-grey": dict(border_bgr=(25, 25, 25), bg=(150, 150, 150), frame=(235, 235, 235)),
    "red-on-light": dict(border_bgr=(40, 40, 200), bg=(225, 225, 225)),
}


def _photo(border_bgr, bg, borders=(3.6, 2.4, 3.0, 3.0), nicks=(), angle=7.0, frame=(20, 20, 20)):
    card, mask = make_card(borders_mm=borders, border_bgr=border_bgr, frame_bgr=frame, nicks=nicks)
    q = tilted_quad(angle_deg=angle)
    return place(card, mask, q, bg_bgr=bg), q


def _true_split(a_mm, b_mm, ppm=14.0):
    a, b = round(a_mm * ppm), round(b_mm * ppm)
    return 100 * a / (a + b)


@pytest.mark.parametrize("name", CASES)
def test_corners_and_centering(name):
    case = CASES[name]
    img, q = _photo(case["border_bgr"], case["bg"], borders=(3.6, 2.4, 2.6, 3.4),
                    frame=case.get("frame", (20, 20, 20)))
    quad = geometry.detect_card(img)
    assert quad is not None
    assert np.max(np.hypot(*(quad - q).T)) < 1.5  # px on a ~2300 px tall card
    side = pipeline.new_side("front", img)
    c = side.centering
    assert abs(c.lr[0] - _true_split(3.6, 2.4)) < 0.5
    assert abs(c.tb[0] - _true_split(2.6, 3.4)) < 0.5
    assert set(side.lines.confidence.values()) == {"high"}


@pytest.mark.parametrize("angle", [-25.0, 0.0, 40.0])
def test_rotation_angles(angle):
    img, q = _photo((160, 70, 20), (30, 30, 30), angle=angle)
    side = pipeline.new_side("front", img)
    c = side.centering
    # 40 degrees tips the card closer to landscape ordering; centering must still hold
    # for whichever axis ends up horizontal.
    lr = c.lr[0] if side.img.shape[0] > side.img.shape[1] else c.tb[0]
    assert abs(min(lr, 100 - lr) - (100 - _true_split(3.6, 2.4))) < 0.6


def test_clean_card_has_no_whitening():
    img, _ = _photo((160, 70, 20), (30, 30, 30))
    side = pipeline.new_side("front", img)
    scan = pipeline.scan_side(side)
    assert scan.whitening == [] and scan.severity == "none"


def test_nicks_are_found_and_located():
    nicks = [("top", 0.3, 0.35), ("right", 0.6, 0.25), ("bottom", 0.7, 0.2)]
    img, _ = _photo((160, 70, 20), (30, 30, 30), nicks=nicks)
    side = pipeline.new_side("front", img)
    scan = pipeline.scan_side(side)
    locs = sorted(s.location for s in scan.whitening)
    assert locs == ["bottom edge", "right edge", "top edge"]
    assert scan.severity in ("moderate", "heavy")


def test_nicks_on_yellow_border():
    img, _ = _photo((20, 210, 245), (40, 35, 30), nicks=[("left", 0.5, 0.3)])
    side = pipeline.new_side("front", img)
    scan = pipeline.scan_side(side)
    assert [s.location for s in scan.whitening] == ["left edge"]


def test_white_border_reported_unreliable():
    img, _ = _photo((240, 240, 240), (30, 30, 30))
    side = pipeline.new_side("front", img)
    scan = pipeline.scan_side(side)
    assert sorted(scan.unreliable_sides) == ["bottom", "left", "right", "top"]
    assert scan.whitening == []


def test_rotate_keeps_measurements():
    img, _ = _photo((160, 70, 20), (30, 30, 30))
    side = pipeline.new_side("front", img)
    before = side.centering
    pipeline.rotate_side(side)
    after = side.centering
    # 90 deg clockwise: old left -> new top, old bottom -> new left
    assert after.top == pytest.approx(before.left)
    assert after.left == pytest.approx(before.bottom)
    for _ in range(3):
        pipeline.rotate_side(side)
    assert side.centering.left == pytest.approx(before.left)


def test_detection_failure_falls_back():
    img = np.full((800, 1000, 3), 128, np.uint8)
    side = pipeline.new_side("front", img)
    assert not side.detected and side.img is not None


def _cent(lr, tb):
    # Build a Centering with the given larger shares.
    return centering.Centering(lr, 100 - lr, tb, 100 - tb, 1.0)


def test_standards_ladder():
    f, b = _cent(52, 50), _cent(70, 60)
    ce = {c.company: c for c in (standards.grade_ceiling(k, f, b) for k in standards.LADDERS)}
    assert ce["PSA"].label == "10"
    assert ce["BGS"].label == "9"          # back 70/30 misses 9.5's 60/40
    assert ce["CGC"].label == "9.5"
    checks = {c.name: c for c in standards.target_checks(f, b)}
    assert checks["PSA 10"].ok and not checks["BGS 9.5"].ok
    # rounding tolerance: 55.4/44.6 reads as 55/45
    assert standards.grade_ceiling("PSA", _cent(55.4, 50), None).label == "10"
    assert standards.grade_ceiling("PSA", _cent(55.6, 50), None).label == "9"


def test_defect_cap_limits_ceiling():
    f = _cent(50, 50)
    c = standards.grade_ceiling("PSA", f, f, "moderate")
    assert c.label == "8" and "whitening" in c.limited_by
    assert standards.grade_ceiling("BGS", _cent(90, 50), None, "minor").label.startswith("below")


def test_report_renders(tmp_path):
    img, _ = _photo((160, 70, 20), (30, 30, 30), nicks=[("top", 0.3, 0.35)])
    front = pipeline.new_side("front", img)
    back = pipeline.new_side("back", _photo((60, 60, 60), (200, 200, 200))[0])
    a = pipeline.analyze({"front": front, "back": back})
    out = tmp_path / "r.png"
    report.save(a, str(out))
    assert out.exists() and out.stat().st_size > 50_000
    assert (tmp_path / "r.json").exists()


def test_box_rotation_roundtrip():
    b = centering.Box(10, 20, 110, 220)
    r = b
    h, w = 300, 200
    for _ in range(4):
        r = r.rotated_cw(h)
        h, w = w, h
    assert vars(r) == pytest.approx(vars(b))


def test_card_mask_and_crops():
    img, _ = _photo((160, 70, 20), (30, 30, 30))
    side = pipeline.new_side("front", img)
    crops = defects.corner_crops(side.img, side.lines.outer, side.px_per_mm)
    assert len(crops) == 4 and all(c.shape[0] == c.shape[1] > 50 for c in crops.values())
    strips = defects.edge_strips(side.img, side.lines.outer, side.px_per_mm)
    assert all(s.shape[0] < s.shape[1] for s in strips.values())


@pytest.mark.parametrize("seed", range(6))
def test_border_widths_with_dark_frame_and_varied_art(seed):
    # Regression: a dark art frame followed by light artwork produces a bigger colour
    # step than border->frame; the detector must still stop at the border's end.
    borders = (3.9, 2.3, 2.8, 3.2)
    card, mask = make_card(borders_mm=borders, seed=seed)
    img = place(card, mask, tilted_quad(angle_deg=-6 + 2 * seed), bg_bgr=(30, 30, 30))
    c = pipeline.new_side("front", img).centering
    for side, mm in zip(("left", "right", "top", "bottom"), borders):
        assert abs(c.mm(side) - round(mm * 14) / 14) < 0.05, side


# ---- reliability features -------------------------------------------------------

def test_glitter_border_is_not_whitening():
    card, mask = make_card(border_bgr=(40, 150, 190), sparkle=6000, seed=2)
    img = place(card, mask, tilted_quad(), bg_bgr=(30, 30, 30))
    side = pipeline.new_side("front", img)
    scan = pipeline.scan_side(side)
    assert scan.severity == "none" and scan.whitening == []
    assert len(scan.textured_sides) >= 3
    assert all(s.kind == "verify" for s in scan.spots)
    # sparkle must not move the inner border either
    c = side.centering
    assert abs(c.lr[0] - _true_split(3.6, 2.4)) < 0.6


def test_nick_on_glitter_border_still_shown_for_verification():
    card, mask = make_card(border_bgr=(40, 150, 190), sparkle=6000, seed=2, nicks=[("right", 0.5, 0.35)])
    img = place(card, mask, tilted_quad(), bg_bgr=(30, 30, 30))
    scan = pipeline.scan_side(pipeline.new_side("front", img))
    assert any(s.location == "right edge" for s in scan.spots)


def test_skewed_print_increases_uncertainty():
    flat = pipeline.new_side("front", _photo((160, 70, 20), (30, 30, 30))[0]).centering
    card, mask = make_card(art_tilt_deg=0.6)
    skew = pipeline.new_side("front", place(card, mask, tilted_quad(), bg_bgr=(30, 30, 30))).centering
    assert skew.sigma_lr > flat.sigma_lr + 0.3


def test_borderline_status():
    # 55.4/44.6 with tight error -> pass; 55.4 with 1 pt SD -> borderline; 60 -> fail
    tight = centering.Centering(55.4, 44.6, 50, 50, 1.0, line_error_px=0.01)
    assert standards.axis_status(tight, 55) == "pass"
    loose = centering.Centering(55.4, 44.6, 50, 50, 1.0, spread_lr=1.0, line_error_px=0.01)
    assert standards.axis_status(loose, 55) == "borderline"
    off = centering.Centering(60, 40, 50, 50, 1.0, spread_lr=1.0, line_error_px=0.01)
    assert standards.axis_status(off, 55) == "fail"


def test_photo_quality_flags():
    import cv2
    img, _ = _photo((160, 70, 20), (30, 30, 30))
    good = pipeline.new_side("front", img).quality
    assert good.level != "bad"
    blurry = pipeline.new_side("front", cv2.GaussianBlur(img, (0, 0), 6)).quality
    assert any("blurry" in m for _, m in blurry.issues)
    dark = pipeline.new_side("front", (img * 0.3).astype(np.uint8)).quality
    assert any("dark" in m for _, m in dark.issues)


def test_verdicts():
    good_f = pipeline.new_side("front", _photo((160, 70, 20), (30, 30, 30), borders=(3.0, 3.0, 3.0, 3.0))[0])
    good_b = pipeline.new_side("back", _photo((160, 70, 20), (30, 30, 30), borders=(3.1, 2.9, 3.0, 3.0))[0])
    v = pipeline.analyze({"front": good_f, "back": good_b}).verdict
    assert v.level == "send", v
    off = pipeline.new_side("front", _photo((160, 70, 20), (30, 30, 30), borders=(4.0, 2.0, 3.0, 3.0))[0])
    assert pipeline.analyze({"front": off, "back": good_b}).verdict.level == "skip"
    assert pipeline.analyze({"front": good_f, "back": None}).verdict.level == "maybe"


def test_heic_loads(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    import cv2
    from PIL import Image
    img, _ = _photo((160, 70, 20), (30, 30, 30))
    pillow_heif.register_heif_opener()
    p = tmp_path / "card.heic"
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(p, quality=90)
    side = pipeline.open_side("front", str(p))
    assert side.detected and abs(side.centering.lr[0] - _true_split(3.6, 2.4)) < 0.6


def _findings(front_img, back_img=None):
    from pregrader import findings
    sides = {"front": pipeline.new_side("front", front_img),
             "back": pipeline.new_side("back", back_img) if back_img is not None else None}
    return findings.build(pipeline.analyze(sides))


def test_findings_clean_card_lists_good_things():
    f = _findings(_photo((160, 70, 20), (30, 30, 30), borders=(3.0, 3.0, 3.0, 3.0))[0],
                  _photo((160, 70, 20), (30, 30, 30), borders=(3.1, 2.9, 3.0, 3.0))[0])
    good = " ".join(x.text for x in f if x.kind == "good")
    assert "no whitening on any of the four corners" in good
    assert "parallel to all four card edges" in good
    assert "centering is excellent" in good
    # back is 51/49, which correctly misses BGS Black Label's perfect 50/50 - nothing else
    assert [x.text for x in f if x.kind == "problem"] == [
        "Centering misses BGS 10 Black Label (needs front 50/50, back 50/50)."]


def test_findings_name_each_problem():
    img = _photo((160, 70, 20), (30, 30, 30), borders=(4.2, 1.8, 3.0, 3.0),
                 nicks=[("top", 0.3, 0.35)])[0]
    f = _findings(img)
    probs = " ".join(x.text for x in f if x.kind == "problem")
    assert "left/right centering" in probs and "over the PSA 10 limit" in probs
    assert "top edge" in probs
    assert any("back wasn't photographed" in x.text for x in f)


def test_findings_detect_tilted_print():
    card, mask = make_card(art_tilt_deg=0.8)
    f = _findings(place(card, mask, tilted_quad(), bg_bgr=(30, 30, 30)))
    assert any("slightly rotated" in x.text for x in f if x.topic == "Squareness")


def test_report_writes_text_file(tmp_path):
    img, _ = _photo((160, 70, 20), (30, 30, 30))
    a = pipeline.analyze({"front": pipeline.new_side("front", img), "back": None})
    report.save(a, str(tmp_path / "r.png"))
    txt = (tmp_path / "r.txt").read_text()
    assert "== Centering ==" in txt and "[GOOD]" in txt and "== Next steps ==" in txt


def test_prediction_reflects_uncertainty():
    sure = centering.Centering(52, 48, 50, 50, 1.0, line_error_px=0.01)
    d = dict(standards.grade_distribution("PSA", sure, None))
    assert d == {"10": 1.0}
    edge = centering.Centering(55.5, 44.5, 50, 50, 1.0, spread_lr=1.0, line_error_px=0.01)
    d = dict(standards.grade_distribution("PSA", edge, None))
    assert 0.4 < d["10"] < 0.6 and 0.4 < d["9"] < 0.6   # right on the limit: a coin flip
    capped = dict(standards.grade_distribution("PSA", sure, None, "moderate"))
    assert capped == {"8": 1.0}


def test_predictions_in_analysis():
    good_f = pipeline.new_side("front", _photo((160, 70, 20), (30, 30, 30), borders=(3.0, 3.0, 3.0, 3.0))[0])
    a = pipeline.analyze({"front": good_f, "back": None})
    psa = next(p for p in a.predictions if p.company == "PSA")
    assert psa.label == "10" and psa.probability > 0.95
    assert psa.confidence == "medium" and "back not photographed" in psa.confidence_why


def test_silver_border_on_woven_cloth():
    # Silver border about as bright as the cloth: colour alone can't separate them.
    from synth import weave
    card, mask = make_card(border_bgr=(118, 114, 112), borders_mm=(3.6, 2.4, 3.0, 3.0))
    q = tilted_quad()
    img = place(card, mask, q, bg_image=weave((3000, 4000)))
    found = geometry.detect_card(img)
    assert found is not None
    assert np.max(np.hypot(*(found - q).T)) < 8  # px on a ~2300 px card (~0.3 mm)
    side = pipeline.new_side("front", img)
    assert abs(side.centering.lr[0] - _true_split(3.6, 2.4)) < 1.0
    assert any("Patterned background" in m for _, m in side.quality.issues)


def test_plain_background_not_flagged_as_patterned():
    side = pipeline.new_side("front", _photo((160, 70, 20), (30, 30, 30))[0])
    assert not any("Patterned" in m for _, m in side.quality.issues)


def test_edge_that_fades_into_background():
    # Left part of the card fades to the same light grey as the table (like a
    # silver-gradient design on a pale surface): one side is nearly invisible.
    card, mask = make_card(border_bgr=(40, 40, 40))
    H, W = card.shape[:2]
    ramp = np.clip(1 - np.arange(W) / (0.35 * W), 0, 1)[None, :, None]
    card = (card * (1 - ramp) + np.array([200, 200, 196]) * ramp).astype(np.uint8)
    q = tilted_quad(angle_deg=2.0)
    img = place(card, mask, q, bg_bgr=(205, 205, 200))
    found = geometry.detect_card(img)
    assert found is not None
    assert np.max(np.hypot(*(found - q).T)) < 10


def test_borderless_side_skips_centering():
    img = _photo((160, 70, 20), (30, 30, 30), borders=(4.5, 1.5, 3.0, 3.0))[0]  # badly off-centre
    side = pipeline.new_side("front", img)
    assert pipeline.analyze({"front": side, "back": None}).verdict.level == "skip"
    side.borderless = True
    a = pipeline.analyze({"front": side, "back": None})
    assert a.verdict.level == "maybe"
    psa = next(p for p in a.predictions if p.company == "PSA")
    assert psa.basis == "edges only" and not psa.has_odds and psa.confidence == "low"


def test_border_hint_for_implausible_borders():
    img = _photo((160, 70, 20), (30, 30, 30), borders=(3.0, 3.0, 4.2, 0.8))[0]
    assert pipeline.border_hint(pipeline.new_side("front", img)) is not None
    img = _photo((160, 70, 20), (30, 30, 30))[0]
    assert pipeline.border_hint(pipeline.new_side("front", img)) is None
