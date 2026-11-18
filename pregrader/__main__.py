"""Command line entry point.

  python -m pregrader                         open the GUI
  python -m pregrader gui FRONT [--back BACK] open the GUI with images loaded
  python -m pregrader analyze FRONT [--back BACK] [-o report.png]   fully automatic, no GUI
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pregrader", description="Offline trading card pre-grader")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="interactive app (default)")
    g.add_argument("front", nargs="?")
    g.add_argument("--back")
    a = sub.add_parser("analyze", help="automatic analysis and report, no GUI")
    a.add_argument("front")
    a.add_argument("--back")
    a.add_argument("-o", "--out", help="report PNG path (default: <front>_pregrade.png)")
    a.add_argument("--sensitivity", type=float, default=1.0, help="whitening sensitivity, 0.5-2.0")
    a.add_argument("--save-warped", action="store_true", help="also save the flattened card image(s)")
    a.add_argument("--front-surface", nargs="+", default=[], metavar="PHOTO",
                   help="angled-light photos of the front (light from the side, then from the top)")
    a.add_argument("--back-surface", nargs="+", default=[], metavar="PHOTO",
                   help="angled-light photos of the back")
    a.add_argument("--no-border", choices=("front", "back", "both"),
                   help="full-art/borderless card: don't grade centering for that side")
    args = p.parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui import main as gui_main
        gui_main(getattr(args, "front", None), getattr(args, "back", None))
        return 0

    from . import pipeline, report
    sides = {"front": pipeline.open_side("front", args.front),
             "back": pipeline.open_side("back", args.back) if args.back else None}
    for name, s in sides.items():
        if s is not None and args.no_border in (name, "both"):
            s.borderless = True
        for p in (args.front_surface if name == "front" else args.back_surface):
            if s is not None:
                s.surface_photos.append(pipeline.load_image(p))
    for s in sides.values():
        if s is not None and not s.detected:
            print(f"warning: {s.name} card not detected automatically - use the GUI to place corners",
                  file=sys.stderr)
    an = pipeline.analyze(sides, args.sensitivity)
    out = args.out or os.path.splitext(args.front)[0] + "_pregrade.png"
    png, js = report.save(an, out)
    if args.save_warped:
        for s in an.sides.values():
            cv2.imwrite(os.path.splitext(out)[0] + f"_{s.name}_flat.png", s.img)
    for s in an.sides.values():
        c = s.centering
        print(f"{s.name:5s}  L/R {c.lr[0]:.1f}/{c.lr[1]:.1f} ±{c.sigma_lr:.1f}   "
              f"T/B {c.tb[0]:.1f}/{c.tb[1]:.1f} ±{c.sigma_tb:.1f}   "
              f"whitening: {s.scan.severity} ({len(s.scan.whitening)} spots)")
    for ch in an.checks:
        print(f"  {(ch.status or 'n/a').upper():<10} {ch.name}")
    print("predicted: " + "   ".join(p.summary() for p in an.predictions))
    if an.predictions:
        p0 = an.predictions[0]
        print(f"           confidence {p0.confidence}" + (f" ({'; '.join(p0.confidence_why)})" if p0.confidence_why else ""))
    print("ceiling:   " + "   ".join(f"{c.company} {c.label}" for c in an.ceilings))
    v = an.verdict
    print(f"\nVERDICT: {v.headline}")
    for r in v.reasons:
        print(f"  - {r}")
    print()
    from . import findings
    print(findings.as_text(an))
    print(f"report: {png}\ndata:   {js}\ntext:   {os.path.splitext(png)[0]}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
