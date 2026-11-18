"""Plain-English findings: every good point and every problem, in words."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import standards
from .centering import Centering, SideSquareness, ratio_text

GOOD, PROBLEM, CHECK, INFO = "good", "problem", "check", "info"
LABEL = {GOOD: "GOOD", PROBLEM: "PROBLEM", CHECK: "CHECK", INFO: "NOTE"}


@dataclass
class Finding:
    kind: str     # good | problem | check | info
    topic: str
    text: str


def _n(k: int, word: str) -> str:
    return f"{k} {word}" if k == 1 else f"{k} {word}s"


def _sides(names: List[str], noun: str = "border") -> str:
    """['left'] -> 'left border is'; ['left', 'right'] -> 'left and right borders are'."""
    if len(names) == 1:
        return f"{names[0]} {noun} is"
    return f"{', '.join(names[:-1])} and {names[-1]} {noun}s are"


def _mm(v: float) -> str:
    return f"{v:.2f} mm"


def _quality_word(worst: float) -> str:
    if worst <= 52:
        return "excellent"
    if worst <= 55:
        return "good"
    if worst <= 60:
        return "a little off"
    if worst <= 65:
        return "noticeably off"
    return "badly off"


def _axis(face: str, axis: str, c: Centering, limit: float) -> Finding:
    a_name, b_name = ("left", "right") if axis == "lr" else ("top", "bottom")
    a, b = c.mm(a_name), c.mm(b_name)
    pct = c.lr if axis == "lr" else c.tb
    sig = c.sigma_lr if axis == "lr" else c.sigma_tb
    worst = max(pct)
    axis_txt = "Left/right" if axis == "lr" else "Top/bottom"
    if abs(a - b) < 0.03:
        wider = "The two borders are the same width."
    else:
        big = a_name if a > b else b_name
        wider = f"The {big} border is {abs(a - b):.2f} mm wider."
    margin = limit + standards.TOLERANCE - worst
    lim_txt = f"{limit:.0f}/{100 - limit:.0f}"
    if margin >= 2 * sig:
        kind, verdict = GOOD, f"That is {margin:.1f} points inside the PSA 10 limit ({lim_txt}), comfortably more than the ±{sig:.1f} measurement error."
    elif margin >= 0:
        kind = CHECK
        verdict = (f"That is {margin:.1f} points inside the PSA 10 limit ({lim_txt}), but this photo's "
                   f"measurement error is ±{sig:.1f}, so it isn't certain yet. A sharper, straight-on photo will settle it.")
    elif margin >= -2 * sig:
        kind = CHECK
        verdict = (f"That is {-margin:.1f} points over the PSA 10 limit ({lim_txt}), but within the ±{sig:.1f} "
                   "measurement error, so it's too close to call from this photo.")
    else:
        kind, verdict = PROBLEM, f"That is {-margin:.1f} points over the PSA 10 limit ({lim_txt}), so a PSA 10 is very unlikely."
    text = (f"{face} {axis_txt.lower()} centering is {_quality_word(worst)}: {ratio_text(pct)} "
            f"({a_name} {_mm(a)}, {b_name} {_mm(b)}). {wider} {verdict}")
    return Finding(kind, "Centering", text)


def _squareness(face: str, sq: Dict[str, SideSquareness]) -> List[Finding]:
    if not sq:
        return []
    sig = {k: v for k, v in sq.items() if v.significant}
    if not sig:
        return [Finding(GOOD, "Squareness", f"{face}: the printed frame runs parallel to all four card edges "
                                           "(no visible tilt or off-square cut).")]
    out = []
    for side, v in sig.items():
        ends = ("top", "bottom") if side in ("left", "right") else ("left", "right")
        wide_end = ends[0] if v.change_mm < 0 else ends[1]
        narrow_end = ends[1] if wide_end == ends[0] else ends[0]
        out.append(Finding(CHECK, "Squareness",
                           f"{face}: the {side} border is {abs(v.change_mm):.2f} mm wider at the {wide_end} end "
                           f"than at the {narrow_end} end, so the frame isn't quite parallel to that edge."))
    c = {k: v.change_mm for k, v in sq.items()}
    rotated = (len(sig) >= 3 and c["left"] * c["right"] < 0 and c["top"] * c["bottom"] < 0
               and c["left"] * c["top"] < 0)
    keystone = ("left" in sig and "right" in sig and c["left"] * c["right"] > 0) or \
               ("top" in sig and "bottom" in sig and c["top"] * c["bottom"] > 0)
    if rotated:
        out.append(Finding(PROBLEM, "Squareness", f"{face}: all four borders change in the pattern of a print that "
                           "is slightly rotated on the card (a tilted or 'diamond' cut). Graders notice this and "
                           "it can cost a 10 even when the centering numbers look fine."))
    elif keystone:
        out.append(Finding(CHECK, "Squareness", f"{face}: opposite borders both widen toward the same end. That "
                           "usually comes from the photo angle rather than the card - retake it straight on."))
    elif len(sig) == 1:
        side = next(iter(sig))
        out.append(Finding(CHECK, "Squareness", f"{face}: only the {side} edge is affected, which points to that "
                           "edge being cut slightly off-square (a mild miscut), or to the photo. Small amounts "
                           "like this rarely matter on their own; confirm with a sharper, straight-on photo."))
    return out


def _photo(face: str, q) -> List[Finding]:
    if q is None:
        return []
    out = [Finding(PROBLEM if lv == "bad" else CHECK, "Photo", f"{face} photo: {msg}") for lv, msg in q.issues]
    m = q.metrics
    texts = " ".join(msg for _, msg in q.issues)
    good = []
    if m.get("px_per_mm", 0) >= 28:
        good.append(f"good resolution ({m['px_per_mm']:.0f} px/mm)")
    if "dark" not in texts and m.get("median_brightness", 0) >= 100:
        good.append("well exposed")
    if "Uneven" not in texts and "light_ratio" in m:
        good.append("evenly lit")
    if "glare" not in texts.lower():
        good.append("no glare")
    blur = m.get("edge_blur_mm", float("nan"))
    if blur == blur and blur <= 0.18:
        good.append(f"sharp (edges {blur:.2f} mm wide)")
    if good:
        out.insert(0, Finding(GOOD, "Photo", f"{face} photo is " + ", ".join(good) + "."))
    return out


def _edges(face: str, s) -> List[Finding]:
    sc = s.scan
    if sc is None:
        return []
    out: List[Finding] = []
    real = sc.whitening
    corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
    edges = ["top", "bottom", "left", "right"]
    by_loc: Dict[str, list] = {}
    for sp in real:
        by_loc.setdefault(sp.location, []).append(sp)
    for name in corners:
        spots = by_loc.get(f"{name} corner", [])
        if spots:
            area = sum(x.area_mm2 for x in spots)
            out.append(Finding(PROBLEM, "Corners", f"{face} {name} corner: white wear found "
                               f"({_n(len(spots), 'spot')}, about {area:.2f} mm²). Corner whitening usually rules out a 10."))
    clean_c = [] if sc.low_res else [n for n in corners if f"{n} corner" not in by_loc
               and not any(p in sc.unreliable_sides for p in n.split("-"))]
    if len(clean_c) == 4:
        out.append(Finding(GOOD, "Corners", f"{face}: no whitening on any of the four corners."))
    elif clean_c:
        out.append(Finding(GOOD, "Corners", f"{face}: clean corners - {', '.join(clean_c)}."))
    for name in edges:
        spots = by_loc.get(f"{name} edge", [])
        if spots:
            area = sum(x.area_mm2 for x in spots)
            size = "a small nick" if area < 0.1 and len(spots) == 1 else "edge wear"
            out.append(Finding(PROBLEM, "Edges", f"{face} {name} edge: {size} found ({_n(len(spots), 'spot')}, "
                               f"about {area:.2f} mm² of white showing)."))
    clean_e = [] if sc.low_res else [n for n in edges if f"{n} edge" not in by_loc
                                     and n not in sc.unreliable_sides and n not in sc.textured_sides]
    if len(clean_e) == 4:
        out.append(Finding(GOOD, "Edges", f"{face}: no whitening along any of the four edges."))
    elif clean_e:
        out.append(Finding(GOOD, "Edges", f"{face}: clean edges - {', '.join(clean_e)}."))
    verify = [sp for sp in sc.spots if sp.kind == "verify"]
    if sc.low_res:
        out.append(Finding(CHECK, "Edges", f"{face}: the photo is too low-resolution to judge whitening"
                           + (f" - {_n(len(verify), 'possible mark')} boxed in yellow; check them by eye or "
                              "retake the photo closer." if verify else "; retake it closer to check the edges.")))
    elif sc.textured_sides:
        out.append(Finding(CHECK, "Edges", f"{face}: the {_sides(sc.textured_sides)} glitter/foil, "
                           "so tiny marks there can't be told apart from sparkle in a photo"
                           + (f" - {_n(len(verify), 'possible mark')} boxed in yellow; look at them under a "
                              "bright light or loupe." if verify else ".")))
    if sc.unreliable_sides:
        out.append(Finding(CHECK, "Edges", f"{face}: the {_sides(sc.unreliable_sides)} white/silver, which hides "
                           "whitening in photos - check those edges by eye."))
    if any(sp.kind == "check-line" for sp in sc.spots):
        out.append(Finding(CHECK, "Edges", f"{face}: a long bright strip along an edge was found - usually the "
                           "outer line is sitting on the background. Nudge it inward in the app."))
    return out


def build(a) -> List[Finding]:
    f: List[Finding] = []
    v = a.verdict
    if v:
        f.append(Finding({"send": GOOD, "maybe": CHECK, "skip": PROBLEM}[v.level], "Overall", v.headline + "."))

    for name, s in a.sides.items():
        face = name.title()
        f.append(Finding(GOOD if s.detected else INFO, "Photo",
                         f"{face}: card found and straightened automatically." if s.detected
                         else f"{face}: card outline was placed by hand."))
        f.extend(_photo(face, s.quality))

    for name, s in a.sides.items():
        face = name.title()
        if s.borderless:
            f.append(Finding(CHECK, "Centering", f"{face}: marked as having no printed border (full-art). Centering "
                             "isn't measured: graders judge it by how evenly the main design sits on the card, so "
                             "compare the margins of the design's frame by eye."))
            continue
        from .pipeline import border_hint
        hint = border_hint(s)
        if hint:
            f.append(Finding(CHECK, "Centering", hint))
        c = s.centering
        limit = standards.step("PSA", "10").front if name == "front" else standards.step("PSA", "10").back
        f.append(_axis(face, "lr", c, limit))
        f.append(_axis(face, "tb", c, limit))
        weak = [k for k, val in s.lines.confidence.items() if val in ("low", "none")]
        if weak:
            f.append(Finding(CHECK, "Centering", f"{face}: the artwork border on the {', '.join(weak)} was hard to "
                             "find automatically - check those lines in the app before trusting the numbers."))
        manual = [k for k, val in s.lines.confidence.items() if val == "manual"]
        if manual:
            f.append(Finding(INFO, "Centering", f"{face}: you placed the {', '.join(manual)} line(s) by hand."))
    if "back" not in a.sides:
        f.append(Finding(CHECK, "Centering", "The back wasn't photographed. PSA 10 also needs back centering of "
                         "75/25 or better and clean back edges, so add a back photo before deciding."))

    for ch in a.checks:
        need = f"front {ch.front_limit:.0f}/{100 - ch.front_limit:.0f}, back {ch.back_limit:.0f}/{100 - ch.back_limit:.0f}"
        st = ch.status
        if st == "pass":
            f.append(Finding(GOOD, "Grade targets", f"Centering meets {ch.name} (needs {need})."))
        elif st == "borderline":
            f.append(Finding(CHECK, "Grade targets", f"Centering is borderline for {ch.name} (needs {need})."))
        elif st == "fail":
            f.append(Finding(PROBLEM, "Grade targets", f"Centering misses {ch.name} (needs {need})."))

    for name, s in a.sides.items():
        if not s.borderless:
            f.extend(_squareness(name.title(), s.squareness))
    for name, s in a.sides.items():
        f.extend(_edges(name.title(), s))

    for pr in getattr(a, "predictions", []):
        if not pr.available:
            f.append(Finding(CHECK, "Grade estimate", f"{pr.company}: no prediction - neither centering nor the "
                             "edges could be measured from this photo."))
            continue
        if not pr.has_odds:
            f.append(Finding(CHECK, "Grade estimate", f"{pr.company}: nothing found on the edges that would stop "
                             f"a {pr.label}, but centering isn't included, so this is a best case, not a prediction."))
            continue
        others = [f"{lab} ({p:.0%})" for lab, p in pr.distribution if lab != pr.label]
        txt = (f"{pr.company}: predicted {pr.label} - {pr.probability:.0%} likely based on {pr.basis}"
               + (f", otherwise {', '.join(others)}" if others else "") + ".")
        if pr.if_marks_real:
            txt += f" If the yellow-boxed marks turn out to be real wear, expect {pr.if_marks_real} instead."
        kind = GOOD if pr.label == pr.ceiling and pr.probability >= 0.75 else CHECK
        f.append(Finding(kind, "Grade estimate", txt))
    if getattr(a, "predictions", None):
        pr = a.predictions[0]
        f.append(Finding({"high": GOOD, "medium": CHECK, "low": PROBLEM}[pr.confidence], "Grade estimate",
                         f"Confidence in these predictions: {pr.confidence}"
                         + (f" ({'; '.join(pr.confidence_why)})." if pr.confidence_why else ".")))
    for ce in a.ceilings:
        why = ("nothing measured holds it back" if ce.limited_by == "nothing measured limits it"
               else f"held back by {ce.limited_by}")
        f.append(Finding(INFO, "Grade estimate", f"{ce.company}: best possible grade (ceiling) {ce.label} ({why}; "
                         "surface not included)."))

    f.append(Finding(INFO, "Not checked", "Surface (scratches, print lines, dents, stains, fingerprints) and "
                     "how sharp the corners are under magnification. Tilt the card under a bright light and "
                     "look for these before submitting."))
    for step in next_steps(a):
        f.append(Finding(INFO, "Next steps", step))
    return f


def next_steps(a) -> List[str]:
    steps: List[str] = []
    if "back" not in a.sides:
        steps.append("Photograph the back the same way and add it.")
    for name, s in a.sides.items():
        if s.quality and s.quality.issues:
            steps.append(f"Retake the {name} photo: " + s.quality.issues[0][1])
            break
    if any(sp.kind == "verify" for s in a.sides.values() for sp in s.scan.spots):
        steps.append("Look at the yellow-boxed spots under a bright light or loupe.")
    if any(v in ("low", "none") for s in a.sides.values() for v in s.lines.confidence.values()):
        steps.append("Open the card in the app and confirm the flagged border lines sit where border meets artwork.")
    if any(c.status == "borderline" for c in a.checks if c.name == "PSA 10"):
        steps.append("Centering is close to the limit: a sharper, straight-on photo will narrow the ± and settle it.")
    if a.verdict and a.verdict.level == "send":
        steps.append("Check the surface under light; if it's clean, this is a reasonable submission.")
    if a.verdict and a.verdict.level == "skip":
        steps.append("Probably not worth a grading fee chasing a 10; a 9 may still be realistic (see grade estimate).")
    return steps


def as_text(a, items: Optional[List[Finding]] = None) -> str:
    items = items if items is not None else build(a)
    lines: List[str] = []
    topic = None
    for it in items:
        if it.topic != topic:
            topic = it.topic
            lines.append(f"\n== {topic} ==")
        lines.append(f"[{LABEL[it.kind]}] {it.text}")
    return "\n".join(lines).strip() + "\n"
