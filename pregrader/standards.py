"""Grading-company centering ladders and grade-ceiling logic.

Rows marked source="spec" are the thresholds you specified. Rows marked "approx"
are commonly cited values filled in so a ceiling below the top grade can be
estimated -- check them against each company's current published standards and
edit freely. Numbers are the maximum allowed larger share, e.g. 55 means 55/45.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .centering import Centering

# Ratios are compared after rounding to whole percent (how a 55.4/44.6 card is read).
TOLERANCE = 0.5


@dataclass(frozen=True)
class GradeStep:
    label: str
    front: float
    back: float
    source: str = "approx"


LADDERS: Dict[str, List[GradeStep]] = {
    "PSA": [
        GradeStep("10", 55, 75, "spec"),
        GradeStep("9", 60, 90),
        GradeStep("8", 65, 90),
        GradeStep("7", 70, 90),
        GradeStep("6", 80, 90),
        GradeStep("5", 85, 90),
        GradeStep("3", 90, 90),
    ],
    "BGS": [
        GradeStep("10 Black Label", 50, 50, "spec"),
        GradeStep("9.5", 55, 60, "spec"),
        GradeStep("9", 60, 70),
        GradeStep("8.5", 65, 80),
        GradeStep("8", 70, 90),
        GradeStep("7", 75, 95),
    ],
    "CGC": [
        GradeStep("Pristine 10", 55, 60, "spec"),
        GradeStep("Gem Mint 10", 60, 65),
        GradeStep("9.5", 60, 70),
        GradeStep("9", 65, 75),
        GradeStep("8.5", 70, 80),
        GradeStep("8", 75, 85),
        GradeStep("7", 80, 90),
    ],
}

# Highest grade still plausible given perimeter whitening severity (heuristic).
DEFECT_CAPS: Dict[str, Dict[str, str]] = {
    "PSA": {"minor": "9", "moderate": "8", "heavy": "7", "severe": "5"},
    "BGS": {"minor": "9", "moderate": "8.5", "heavy": "7", "severe": "below"},
    "CGC": {"minor": "9", "moderate": "8.5", "heavy": "7", "severe": "below"},
}

# The four headline targets checked individually in the UI and report.
TARGETS = [
    ("PSA 10", "PSA", "10"),
    ("BGS 10 Black Label", "BGS", "10 Black Label"),
    ("BGS 9.5", "BGS", "9.5"),
    ("CGC Pristine 10", "CGC", "Pristine 10"),
]

SEVERITY_ORDER = ["none", "minor", "moderate", "heavy", "severe"]   # severe = a crease


def _cap_index(company: str, severity: str) -> int:
    """Ladder position of the cap for this severity ("below" = under the lowest step)."""
    cap = DEFECT_CAPS[company].get(severity)
    ladder = LADDERS[company]
    if not cap:
        return 0
    if cap == "below":
        return len(ladder)
    return next(i for i, s in enumerate(ladder) if s.label == cap)


def step(company: str, label: str) -> GradeStep:
    return next(s for s in LADDERS[company] if s.label == label)


def passes(value: Optional[float], limit: float) -> Optional[bool]:
    if value is None or value != value:  # None or NaN
        return None
    return value <= limit + TOLERANCE


STATUS_ORDER = ["pass", "borderline", "fail"]


def axis_status(c: Centering, limit: float, z: float = 2.0) -> str:
    """pass: clears the limit even at the pessimistic end of the measurement
    uncertainty; fail: misses it even at the optimistic end; else borderline."""
    lim = limit + TOLERANCE
    worst = "pass"
    for pct, sig in ((c.lr, c.sigma_lr), (c.tb, c.sigma_tb)):
        v = max(pct)
        if v != v:
            return "borderline"
        st = "fail" if v - z * sig > lim else "borderline" if v + z * sig > lim else "pass"
        worst = max(worst, st, key=STATUS_ORDER.index)
    return worst


@dataclass
class Check:
    name: str
    front_limit: float
    back_limit: float
    front: Optional[Centering]
    back: Optional[Centering]

    @property
    def front_value(self) -> Optional[float]:
        return self.front.worst if self.front else None

    @property
    def back_value(self) -> Optional[float]:
        return self.back.worst if self.back else None

    @property
    def front_status(self) -> Optional[str]:
        return axis_status(self.front, self.front_limit) if self.front else None

    @property
    def back_status(self) -> Optional[str]:
        return axis_status(self.back, self.back_limit) if self.back else None

    @property
    def status(self) -> Optional[str]:
        parts = [p for p in (self.front_status, self.back_status) if p is not None]
        return max(parts, key=STATUS_ORDER.index) if parts else None

    @property
    def ok(self) -> Optional[bool]:
        """Point-estimate pass/fail (ignores uncertainty)."""
        parts = [p for p in (passes(self.front_value, self.front_limit),
                             passes(self.back_value, self.back_limit)) if p is not None]
        return None if not parts else all(parts)


def target_checks(front: Optional[Centering], back: Optional[Centering]) -> List[Check]:
    return [Check(name, step(company, label).front, step(company, label).back, front, back)
            for name, company, label in TARGETS]


@dataclass
class Ceiling:
    company: str
    label: str
    centering_label: str
    defect_cap: Optional[str]
    limited_by: str


def _centering_index(company: str, front: Optional[Centering], back: Optional[Centering]) -> int:
    ladder = LADDERS[company]
    for i, s in enumerate(ladder):
        ok_f = passes(front.worst, s.front) if front else True
        ok_b = passes(back.worst, s.back) if back else True
        if ok_f is not False and ok_b is not False:
            return i
    return len(ladder)


def grade_ceiling(company: str, front: Optional[Centering], back: Optional[Centering],
                  severity: str = "none") -> Ceiling:
    ladder = LADDERS[company]
    ic = _centering_index(company, front, back)
    cap = DEFECT_CAPS[company].get(severity)
    idd = _cap_index(company, severity)

    def label(i: int) -> str:
        return ladder[i].label if i < len(ladder) else f"below {ladder[-1].label}"

    idx = max(ic, idd)
    if idx == 0:
        why = "nothing measured limits it"
    elif ic >= idd:
        why = "centering"
    else:
        why = f"edge/corner whitening ({severity})"
    if ic == idd and ic > 0:
        why = f"centering and whitening ({severity})"
    return Ceiling(company, label(idx), label(ic), cap, why)


def worst_severity(*sev: Optional[str]) -> str:
    vals = [s for s in sev if s]
    return max(vals, key=SEVERITY_ORDER.index) if vals else "none"


@dataclass
class Prediction:
    company: str
    label: str                       # most likely grade
    probability: float               # chance of exactly that grade
    distribution: List[tuple]        # [(label, probability)], best grade first, >= 1% only
    ceiling: str
    if_marks_real: Optional[str]     # grade if 'verify' marks turn out to be real wear
    confidence: str                  # high | medium | low
    confidence_why: List[str]
    basis: str = "centering and edges"   # | "edges only" | "centering only" | "nothing"

    @property
    def available(self) -> bool:
        return self.basis != "nothing"

    @property
    def has_odds(self) -> bool:
        """Percentages only mean something when centering was measured."""
        return "centering" in self.basis

    def summary(self) -> str:
        if not self.available:
            return f"{self.company} n/a (nothing measurable)"
        if not self.has_odds:
            return f"{self.company} up to {self.label} ({self.basis})"
        others = [f"{lab} {p:.0%}" for lab, p in self.distribution if lab != self.label]
        tail = f"; otherwise {', '.join(others)}" if others else ""
        return f"{self.company} {self.label} ({self.probability:.0%} likely{tail}; {self.basis})"


def grade_distribution(company: str, front: Optional[Centering], back: Optional[Centering],
                       severity: str = "none", n: int = 4000, seed: int = 7) -> List[tuple]:
    """Probability of each grade, propagating centering measurement error (each ratio
    drawn from a normal with its ±) through the ladder, then applying the whitening cap."""
    ladder = LADDERS[company]
    rng = np.random.default_rng(seed)
    idx = np.zeros(n, dtype=int)

    def draws(c: Centering, axis: str) -> np.ndarray:
        pct, sig = (c.lr, c.sigma_lr) if axis == "lr" else (c.tb, c.sigma_tb)
        a = pct[0] + rng.normal(0.0, sig if sig == sig else 3.0, n)
        return np.maximum(a, 100 - a)

    faces = [(c, kind) for c, kind in ((front, "front"), (back, "back")) if c is not None]
    worst = {kind: np.maximum(draws(c, "lr"), draws(c, "tb")) for c, kind in faces}
    idx[:] = len(ladder)
    for i in range(len(ladder) - 1, -1, -1):
        s = ladder[i]
        passes_i = np.ones(n, dtype=bool)
        for kind, w in worst.items():
            passes_i &= w <= (s.front if kind == "front" else s.back) + TOLERANCE
        idx[passes_i] = i
    cap = DEFECT_CAPS[company].get(severity)
    if cap:
        ci = _cap_index(company, severity)
        idx = np.maximum(idx, ci)
    labels = [s.label for s in ladder] + [f"below {ladder[-1].label}"]
    counts = np.bincount(idx, minlength=len(labels)) / n
    return [(labels[i], float(p)) for i, p in enumerate(counts) if p >= 0.01]


def predict(company: str, front: Optional[Centering], back: Optional[Centering], severity: str,
            unknown_marks: bool, confidence_why: List[str], photo_bad: bool,
            basis: str = "centering and edges") -> Prediction:
    dist = grade_distribution(company, front, back, severity)
    # most likely grade; on a tie take the lower grade (conservative)
    label, prob = max(dist, key=lambda t: (round(t[1], 3), [d[0] for d in dist].index(t[0])))
    ceiling = grade_ceiling(company, front, back, severity).label
    alt = None
    if unknown_marks:
        worse = worst_severity(severity, "minor")
        d2 = grade_distribution(company, front, back, worse)
        alt = max(d2, key=lambda t: t[1])[0]
        if alt == label:
            alt = None
    why = list(confidence_why)
    if prob < 0.75:
        why.append("centering is close to a grade limit")
    conf = "low" if photo_bad else ("medium" if why else "high")
    return Prediction(company, label, prob, dist, ceiling, alt, conf, why, basis)
