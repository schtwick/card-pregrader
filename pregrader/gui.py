"""Tkinter desktop app: load front/back, fine-tune border lines, scan edges, export.

Centering mode
  click + drag a line          move it (nearest line within ~10 px is grabbed)
  Tab / Shift+Tab, keys 1-8    select a line
  arrow keys                   nudge the selected line by the step size
  Shift + arrow                nudge x10
Corners mode
  drag the 4 handles onto the card's corners (where the straight edges would
  meet, ignoring the rounding), then "Apply warp". Arrow keys nudge the selected
  handle.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import pipeline, report, standards
from .centering import ratio_text
from .pipeline import Side

LINE_ORDER = [("outer", "left"), ("inner", "left"), ("outer", "right"), ("inner", "right"),
              ("outer", "top"), ("inner", "top"), ("outer", "bottom"), ("inner", "bottom")]
COLORS = {"outer": "#2fdc2f", "inner": "#ff3cff"}
GRID = "#ffd400"
HANDLE = "#00c8ff"
LOUPE = 300


def _is_vertical(side: str) -> bool:
    return side in ("left", "right")


class App(tk.Tk):
    def __init__(self, front: Optional[str] = None, back: Optional[str] = None):
        super().__init__()
        self.title("Card Pre-Grader")
        self.geometry("1480x960")
        self.minsize(1100, 700)

        self.sides: Dict[str, Optional[Side]] = {"front": None, "back": None}
        self.side_var = tk.StringVar(value="front")
        self.mode_var = tk.StringVar(value="center")
        self.step_var = tk.StringVar(value="1")
        self.zoom_var = tk.StringVar(value="8")
        self.sens_var = tk.DoubleVar(value=1.0)
        self.show_defects = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.borderless_var = tk.BooleanVar(value=False)
        self.sel = 1            # index into LINE_ORDER
        self.sel_handle = 0     # corner handle index
        self.drag = False
        self.hover: Optional[Tuple[float, float]] = None
        self.edit_quad: Optional[np.ndarray] = None   # corners being edited (source px)
        self.scale, self.off = 1.0, (0.0, 0.0)
        self._photo = None
        self._photo_key = None
        self._loupe_photo = None
        self._rescan_job = None
        self._resize_job = None

        icon = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon):
            try:  # Dock / window icon
                self._icon = ImageTk.PhotoImage(Image.open(icon))
                self.iconphoto(True, self._icon)
            except tk.TclError:
                pass
        self._build_ui()
        self._bind_events()
        for name, path in (("front", front), ("back", back)):
            if path:
                self.load(name, path)

    # ------------------------------------------------------------------ layout
    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam" if sys.platform != "darwin" else "aqua")
        except tk.TclError:
            pass
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Open Front…", command=lambda: self.open_dialog("front")).pack(side="left")
        ttk.Button(bar, text="Open Back…", command=lambda: self.open_dialog("back")).pack(side="left", padx=(4, 14))
        ttk.Label(bar, text="Side:").pack(side="left")
        for v in ("front", "back"):
            ttk.Radiobutton(bar, text=v.title(), value=v, variable=self.side_var,
                            command=self.on_side_change).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(bar, text="Mode:").pack(side="left")
        ttk.Radiobutton(bar, text="Centering", value="center", variable=self.mode_var,
                        command=self.on_mode_change).pack(side="left")
        ttk.Radiobutton(bar, text="Card corners", value="corners", variable=self.mode_var,
                        command=self.on_mode_change).pack(side="left")
        bar2 = ttk.Frame(self, padding=(8, 0, 8, 6))
        bar2.pack(side="top", fill="x")
        self.apply_btn = ttk.Button(bar2, text="Apply warp", command=self.apply_warp)
        self.apply_btn.pack(side="left")
        ttk.Button(bar2, text="Re-detect card", command=self.redetect).pack(side="left", padx=4)
        ttk.Button(bar2, text="Auto inner borders", command=self.auto_borders).pack(side="left")
        ttk.Button(bar2, text="Rotate 90°", command=self.rotate).pack(side="left", padx=4)
        ttk.Button(bar2, text="Scan edges", command=self.scan).pack(side="left")
        ttk.Separator(bar2, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar2, text="Add surface photos…", command=self.add_surface).pack(side="left")
        ttk.Button(bar2, text="Clear surface photos", command=self.clear_surface).pack(side="left", padx=4)
        ttk.Button(bar, text="Export report…", command=self.export).pack(side="right")

        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg="#1b1b1d", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)

        panel = ttk.Frame(body, padding=8, width=390)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        ttk.Label(panel, text="Loupe", font=("Helvetica", 13, "bold")).pack(anchor="w")
        self.loupe = tk.Canvas(panel, width=LOUPE, height=LOUPE, bg="#000", highlightthickness=1,
                               highlightbackground="#555")
        self.loupe.pack(anchor="w", pady=(2, 4))
        row = ttk.Frame(panel)
        row.pack(anchor="w", fill="x")
        ttk.Label(row, text="Zoom").pack(side="left")
        ttk.Combobox(row, textvariable=self.zoom_var, values=("4", "8", "12", "16", "24"), width=4,
                     state="readonly").pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Nudge px").pack(side="left")
        ttk.Combobox(row, textvariable=self.step_var, values=("0.1", "0.25", "0.5", "1", "2", "5"), width=5,
                     state="readonly").pack(side="left", padx=4)
        self.zoom_var.trace_add("write", lambda *_: self.update_loupe())

        nud = ttk.Frame(panel)
        nud.pack(anchor="w", pady=4)
        ttk.Button(nud, text="◀ / ▲", width=7, command=lambda: self.nudge(-1)).pack(side="left")
        ttk.Button(nud, text="▶ / ▼", width=7, command=lambda: self.nudge(+1)).pack(side="left", padx=4)
        ttk.Button(nud, text="Next (Tab)", command=lambda: self.cycle(1)).pack(side="left")

        opts = ttk.Frame(panel)
        opts.pack(anchor="w", fill="x", pady=(2, 2))
        ttk.Checkbutton(opts, text="Defect boxes", variable=self.show_defects,
                        command=self.draw_overlays).pack(side="left")
        ttk.Checkbutton(opts, text="Centre grid", variable=self.show_grid,
                        command=self.draw_overlays).pack(side="left", padx=8)
        ttk.Checkbutton(opts, text="No border (full-art)", variable=self.borderless_var,
                        command=self.on_borderless).pack(side="left")
        sens = ttk.Frame(panel)
        sens.pack(anchor="w", fill="x")
        ttk.Label(sens, text="Whitening sensitivity").pack(side="left")
        self.sens_lbl = ttk.Label(sens, text="1.00", width=5)
        ttk.Scale(sens, from_=0.5, to=2.0, variable=self.sens_var, orient="horizontal", length=120,
                  command=self._on_sens).pack(side="left", padx=4)
        self.sens_lbl.pack(side="left")

        info_box = ttk.Frame(panel)
        info_box.pack(fill="both", expand=True, pady=(8, 0))
        self.info = tk.Text(info_box, width=44, height=24, wrap="word", relief="flat", bg="#fafafa",
                            font=("Helvetica", 12), padx=6, pady=6, cursor="arrow")
        sb = ttk.Scrollbar(info_box, orient="vertical", command=self.info.yview)
        self.info.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.info.pack(side="left", fill="both", expand=True)
        for tag, kw in {"h": dict(font=("Helvetica", 13, "bold")), "big": dict(font=("Helvetica", 15, "bold")),
                        "ok": dict(foreground="#178a46"), "bad": dict(foreground="#c62828"),
                        "warn": dict(foreground="#b26a00"), "muted": dict(foreground="#777"),
                        "sel": dict(background="#fff2a8"), "link": dict(foreground="#1565c0", underline=1)}.items():
            self.info.tag_configure(tag, **kw)
        self.info.configure(state="disabled")

        self.status = ttk.Label(self, anchor="w", padding=(8, 3))
        self.status.pack(side="bottom", fill="x")
        self.set_status("Open a front (and optionally back) image to begin.")

    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<Configure>", self._on_resize)
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Leave>", lambda e: self._set_hover(None))
        for key, d in (("Left", -1), ("Up", -1), ("Right", 1), ("Down", 1)):
            self.bind_all(f"<{key}>", lambda e, k=key, d=d: self._on_arrow(e, k, d, 1))
            self.bind_all(f"<Shift-{key}>", lambda e, k=key, d=d: self._on_arrow(e, k, d, 10))
        self.bind_all("<Tab>", lambda e: self._key_guard(e, lambda: self.cycle(1)))
        self.bind_all("<Shift-Tab>", lambda e: self._key_guard(e, lambda: self.cycle(-1)))
        try:
            self.bind_all("<ISO_Left_Tab>", lambda e: self._key_guard(e, lambda: self.cycle(-1)))
        except tk.TclError:
            pass
        for n in range(1, 9):
            self.bind_all(str(n), lambda e, n=n: self._key_guard(e, lambda: self.select(n - 1)))

    # ----------------------------------------------------------------- helpers
    @property
    def side(self) -> Optional[Side]:
        return self.sides[self.side_var.get()]

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def img2can(self, x: float, y: float) -> Tuple[float, float]:
        return self.off[0] + (x + 0.5) * self.scale, self.off[1] + (y + 0.5) * self.scale

    def can2img(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - self.off[0]) / self.scale - 0.5, (cy - self.off[1]) / self.scale - 0.5

    def _display_bgr(self) -> Optional[np.ndarray]:
        s = self.side
        if s is None:
            return None
        return s.original if self.mode_var.get() == "corners" else s.img

    def _key_guard(self, event, fn):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Spinbox)):
            return None
        fn()
        return "break"

    # ----------------------------------------------------------------- loading
    def open_dialog(self, name: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Open {name} image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.heic *.heif *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self.load(name, path)

    def load(self, name: str, path: str) -> None:
        self.set_status(f"Loading {os.path.basename(path)}…")
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            side = pipeline.open_side(name, path)
        except Exception as e:  # noqa: BLE001 - show any load failure to the user
            messagebox.showerror("Could not load image", str(e))
            self.set_status("Load failed.")
            return
        finally:
            self.config(cursor="")
        pipeline.scan_side(side, self.sens_var.get())
        self.sides[name] = side
        self.side_var.set(name)
        if side.detected:
            self.mode_var.set("center")
            self.set_status(f"{name.title()}: card detected and flattened. Check the lines, then nudge as needed.")
        else:
            self.mode_var.set("corners")
            self.set_status(f"{name.title()}: card NOT detected - drag the 4 corner handles onto the card, "
                            "then press Apply warp.")
        self.on_mode_change()

    # ------------------------------------------------------------------ events
    def on_side_change(self) -> None:
        if self.side is None:
            self.set_status(f"No {self.side_var.get()} image loaded - use Open {self.side_var.get().title()}…")
        self.on_mode_change()

    def on_borderless(self) -> None:
        s = self.side
        if s is not None:
            s.borderless = bool(self.borderless_var.get())
            self.refresh()

    def on_mode_change(self) -> None:
        s = self.side
        self.borderless_var.set(bool(s.borderless) if s is not None else False)
        self.edit_quad = s.quad.copy() if (s is not None and self.mode_var.get() == "corners") else None
        self.apply_btn.state(["!disabled"] if self.mode_var.get() == "corners" else ["disabled"])
        self.redraw()

    def _on_resize(self, _e) -> None:
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(60, self.redraw)

    def _on_sens(self, _v) -> None:
        self.sens_lbl.configure(text=f"{self.sens_var.get():.2f}")
        self._schedule_rescan()

    def _set_hover(self, p) -> None:
        self.hover = p
        self.update_loupe()

    def on_motion(self, e) -> None:
        if self._display_bgr() is not None and not self.drag:
            self._set_hover(self.can2img(e.x, e.y))

    def on_press(self, e) -> None:
        self.canvas.focus_set()
        s = self.side
        if s is None:
            return
        if self.mode_var.get() == "corners":
            pts = [self.img2can(*p) for p in self.edit_quad]
            d = [np.hypot(px - e.x, py - e.y) for px, py in pts]
            i = int(np.argmin(d))
            if d[i] < 25:
                self.sel_handle, self.drag = i, True
                self.draw_overlays()
            return
        best, bi = 1e9, None
        for i, (kind, side) in enumerate(LINE_ORDER):
            v = getattr(s.lines, kind).get(side)
            cx, cy = self.img2can(v, v)
            dist = abs(e.x - cx) if _is_vertical(side) else abs(e.y - cy)
            if dist < best:
                best, bi = dist, i
        if best <= 10:
            self.sel, self.drag = bi, True
            self.refresh()

    def on_drag(self, e) -> None:
        if not self.drag or self.side is None:
            return
        x, y = self.can2img(e.x, e.y)
        if self.mode_var.get() == "corners":
            self.edit_quad[self.sel_handle] = (x, y)
            self.hover = None
            self.draw_overlays()
            self.update_loupe(center=(x, y))
            return
        kind, side = LINE_ORDER[self.sel]
        v = x if _is_vertical(side) else y
        self._set_line(kind, side, round(v * 4) / 4)  # quarter-pixel snapping while dragging
        self.hover = None
        self.refresh(center=(x, y))

    def on_release(self, _e) -> None:
        if self.drag:
            self.drag = False
            if self.mode_var.get() == "center" and LINE_ORDER[self.sel][0] == "outer":
                self._schedule_rescan()

    def _on_arrow(self, e, key: str, direction: int, mult: float):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Spinbox, tk.Text)):
            return None
        s = self.side
        if s is None:
            return "break"
        step = float(self.step_var.get()) * mult
        if self.mode_var.get() == "corners":
            dx = direction * step if key in ("Left", "Right") else 0
            dy = direction * step if key in ("Up", "Down") else 0
            self.edit_quad[self.sel_handle] += (dx, dy)
            self.hover = None
            self.draw_overlays()
            self.update_loupe()
            return "break"
        kind, side = LINE_ORDER[self.sel]
        if _is_vertical(side) != (key in ("Left", "Right")):
            # wrong axis for this line: jump to the nearest line on the other axis
            self.select(self.sel + 4 if self.sel < 4 else self.sel - 4)
            return "break"
        self.nudge(direction, step)
        return "break"

    # ----------------------------------------------------------------- actions
    def _set_line(self, kind: str, side: str, v: float) -> None:
        s = self.side
        H, W = s.img.shape[:2]
        v = float(np.clip(v, 0, (W if _is_vertical(side) else H) - 1))
        getattr(s.lines, kind).set(side, v)
        if kind == "inner":
            s.lines.confidence[side] = "manual"

    def nudge(self, direction: int, step: Optional[float] = None) -> None:
        s = self.side
        if s is None or self.mode_var.get() != "center":
            return
        step = float(self.step_var.get()) if step is None else step
        kind, side = LINE_ORDER[self.sel]
        self._set_line(kind, side, getattr(s.lines, kind).get(side) + direction * step)
        self.hover = None
        self.refresh()
        if kind == "outer":
            self._schedule_rescan()

    def select(self, i: int) -> None:
        if self.mode_var.get() == "corners":
            self.sel_handle = i % 4
            self.hover = None
            self.draw_overlays()
            self.update_loupe()
            return
        self.sel = i % len(LINE_ORDER)
        self.hover = None
        self.refresh()

    def cycle(self, d: int) -> None:
        self.select((self.sel_handle if self.mode_var.get() == "corners" else self.sel) + d)

    def apply_warp(self) -> None:
        s = self.side
        if s is None or self.edit_quad is None:
            return
        if not np.allclose(s.quad, self.edit_quad):
            s.detected = False  # the report notes the outline was placed by hand
        s.quad = self.edit_quad.copy()
        pipeline.warp_side(s)
        pipeline.scan_side(s, self.sens_var.get())
        self.mode_var.set("center")
        self.on_mode_change()
        self.set_status("Warp applied with your corners; inner borders re-detected.")

    def redetect(self) -> None:
        s = self.side
        if s is None:
            return
        side = pipeline.new_side(s.name, s.original, s.path)
        side.rotation = s.rotation
        if side.rotation:
            pipeline.warp_side(side)
        pipeline.scan_side(side, self.sens_var.get())
        self.sides[s.name] = side
        self.mode_var.set("center" if side.detected else "corners")
        self.on_mode_change()
        self.set_status("Card re-detected." if side.detected else "Detection failed - place corners manually.")

    def auto_borders(self) -> None:
        s = self.side
        if s is None:
            return
        pipeline.auto_inner(s)
        self.refresh()
        weak = [k for k, v in s.lines.confidence.items() if v != "high"]
        self.set_status("Inner borders re-detected." + (f" Low confidence: {', '.join(weak)}." if weak else ""))

    def rotate(self) -> None:
        s = self.side
        if s is None:
            return
        pipeline.rotate_side(s)
        pipeline.scan_side(s, self.sens_var.get())
        if self.mode_var.get() == "corners":
            self.mode_var.set("center")
            self.on_mode_change()
        else:
            self.redraw()

    def scan(self) -> None:
        s = self.side
        if s is None:
            return
        sc = pipeline.scan_side(s, self.sens_var.get())
        self.show_defects.set(True)
        self.refresh()
        self.set_status(f"Edge scan: {sc.severity}, {len(sc.whitening)} whitening spot(s).")

    def add_surface(self) -> None:
        s = self.side
        if s is None:
            messagebox.showinfo("Load the card first", "Open the normal photo of this side first.")
            return
        paths = filedialog.askopenfilenames(
            title=f"Angled-light photos of the {s.name}",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.heic *.heif *.tif *.tiff *.webp"), ("All files", "*.*")])
        if not paths:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            for p in paths:
                s.surface_photos.append(pipeline.load_image(p))
            pipeline.scan_side(s, self.sens_var.get())
        except Exception as e:  # noqa: BLE001 - show any load failure
            messagebox.showerror("Could not add photo", str(e))
        finally:
            self.config(cursor="")
        sc = s.surface
        msg = f"Surface: {sc.photos_used} photo(s) used" if sc else "Surface: no photos"
        if sc and sc.photos_rejected:
            msg += f", {sc.photos_rejected} couldn't be lined up"
        if sc:
            msg += f" - {len(sc.defects)} defect(s) found."
        self.set_status(msg)
        self.refresh()

    def clear_surface(self) -> None:
        s = self.side
        if s is None:
            return
        s.surface_photos.clear()
        s.surface_cache.clear()
        s.surface = None
        self.refresh()
        self.set_status("Surface photos removed.")

    def _schedule_rescan(self) -> None:
        if self._rescan_job:
            self.after_cancel(self._rescan_job)

        def run():
            self._rescan_job = None
            for s in self.sides.values():
                if s is not None:
                    pipeline.reassess(s)
                    pipeline.scan_side(s, self.sens_var.get())
            self.refresh()
        self._rescan_job = self.after(350, run)

    def export(self) -> None:
        if not any(self.sides.values()):
            messagebox.showinfo("Nothing to export", "Load at least one image first.")
            return
        first = next(s for s in (self.sides["front"], self.sides["back"]) if s)
        stem = os.path.splitext(os.path.basename(first.path or "card"))[0]
        path = filedialog.asksaveasfilename(
            title="Save report", defaultextension=".png", initialfile=f"{stem}_pregrade.png",
            initialdir=os.path.dirname(first.path) if first.path else None,
            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            for s in self.sides.values():
                if s is not None:
                    pipeline.scan_side(s, self.sens_var.get())
            a = pipeline.analyze(self.sides, self.sens_var.get())
            png, js = report.save(a, path)
        finally:
            self.config(cursor="")
        self.set_status(f"Saved {png} and {os.path.basename(js)}")
        if sys.platform == "darwin":
            subprocess.Popen(["open", png])
        elif sys.platform.startswith("win"):
            os.startfile(png)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", png])

    # ---------------------------------------------------------------- drawing
    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        img = self._display_bgr()
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        if img is None:
            c.create_text(cw / 2, ch / 2, fill="#aaa", font=("Helvetica", 16),
                          text="Open Front… / Open Back… to load a card photo\n\n"
                               "Best results: whole card in frame, flat, on a plain dark background,\n"
                               "even lighting, no glare.", justify="center")
            self.update_info()
            self.update_loupe()
            return
        H, W = img.shape[:2]
        self.scale = min((cw - 24) / W, (ch - 24) / H)
        dw, dh = max(1, int(W * self.scale)), max(1, int(H * self.scale))
        self.off = ((cw - dw) / 2, (ch - dh) / 2)
        key = (id(img), dw, dh)
        if key != self._photo_key:
            small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self._photo_key = key
        c.create_image(self.off[0], self.off[1], anchor="nw", image=self._photo, tags="base")
        self.draw_overlays()
        self.update_info()
        self.update_loupe()

    def refresh(self, center=None) -> None:
        self.draw_overlays()
        self.update_info()
        self.update_loupe(center=center)

    def draw_overlays(self) -> None:
        c = self.canvas
        c.delete("ov")
        s = self.side
        img = self._display_bgr()
        if s is None or img is None:
            return
        H, W = img.shape[:2]
        x0, y0 = self.img2can(-0.5, -0.5)
        x1, y1 = self.img2can(W - 0.5, H - 0.5)
        if self.mode_var.get() == "corners":
            pts = [self.img2can(*p) for p in self.edit_quad]
            c.create_polygon(*[v for p in pts for v in p], outline=HANDLE, fill="", width=2, tags="ov")
            for i, (px, py) in enumerate(pts):
                r = 9 if i == self.sel_handle else 7
                c.create_oval(px - r, py - r, px + r, py + r, outline=HANDLE, width=2,
                              fill="#ffffff" if i == self.sel_handle else "", tags="ov")
                c.create_line(px - 14, py, px + 14, py, fill=HANDLE, tags="ov")
                c.create_line(px, py - 14, px, py + 14, fill=HANDLE, tags="ov")
            return
        L = s.lines
        if self.show_grid.get():
            o, i = L.outer, L.inner
            cx, cy = self.img2can((o.left + o.right) / 2, (o.top + o.bottom) / 2)
            ax, ay = self.img2can((i.left + i.right) / 2, (i.top + i.bottom) / 2)
            ot, ob = self.img2can(o.left, o.top), self.img2can(o.right, o.bottom)
            c.create_line(cx, ot[1], cx, ob[1], fill=GRID, dash=(6, 4), tags="ov")
            c.create_line(ot[0], cy, ob[0], cy, fill=GRID, dash=(6, 4), tags="ov")
            c.create_line(ax, ot[1], ax, ob[1], fill=COLORS["inner"], dash=(2, 4), tags="ov")
            c.create_line(ot[0], ay, ob[0], ay, fill=COLORS["inner"], dash=(2, 4), tags="ov")
            c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, outline=GRID, width=2, tags="ov")
            c.create_oval(ax - 4, ay - 4, ax + 4, ay + 4, fill=COLORS["inner"], outline="", tags="ov")
        if self.show_defects.get() and s.scan is not None:
            for sp in s.scan.spots:
                a = self.img2can(sp.x - 6, sp.y - 6)
                b = self.img2can(sp.x + sp.w + 6, sp.y + sp.h + 6)
                col = {"whitening": "#ff3030", "verify": "#ffd000"}.get(sp.kind, "#ffa000")
                c.create_rectangle(a[0] - 3, a[1] - 3, b[0] + 3, b[1] + 3, outline=col, width=2, tags="ov")
        if self.show_defects.get() and s.surface is not None:
            for d in s.surface.defects:
                a = self.img2can(d.x - 8, d.y - 8)
                b = self.img2can(d.x + d.w + 8, d.y + d.h + 8)
                col = {"scratch": "#00a0ff", "crease": "#ff00a0", "dent": "#00e0e0", "mark": "#00e0e0"}[d.kind]
                c.create_rectangle(a[0] - 3, a[1] - 3, b[0] + 3, b[1] + 3, outline=col, width=2, dash=(4, 2), tags="ov")
        for idx, (kind, side) in enumerate(LINE_ORDER):
            v = getattr(L, kind).get(side)
            selected = idx == self.sel
            w = 3 if selected else 1
            if _is_vertical(side):
                x, _ = self.img2can(v, 0)
                c.create_line(x, y0, x, y1, fill=COLORS[kind], width=w, tags="ov")
            else:
                _, y = self.img2can(0, v)
                c.create_line(x0, y, x1, y, fill=COLORS[kind], width=w, tags="ov")
            if selected:
                lx, ly = (self.img2can(v, (L.outer.top + L.outer.bottom) / 2) if _is_vertical(side)
                          else self.img2can((L.outer.left + L.outer.right) / 2, v))
                c.create_text(lx + 8, ly - 10, text=f"{idx + 1}: {kind} {side}", anchor="w",
                              fill="#ffffff", font=("Helvetica", 12, "bold"), tags="ov")

    def update_loupe(self, center=None) -> None:
        lp = self.loupe
        lp.delete("all")
        s = self.side
        img = self._display_bgr()
        if s is None or img is None:
            return
        z = int(self.zoom_var.get())
        H, W = img.shape[:2]
        corners = self.mode_var.get() == "corners"
        if center is None:
            center = self.hover
        if center is None:
            if corners:
                center = tuple(self.edit_quad[self.sel_handle])
            else:
                kind, side = LINE_ORDER[self.sel]
                v = getattr(s.lines, kind).get(side)
                o = s.lines.outer
                center = (v, (o.top + o.bottom) / 2) if _is_vertical(side) else ((o.left + o.right) / 2, v)
        n = LOUPE // z
        x0 = int(round(center[0])) - n // 2
        y0 = int(round(center[1])) - n // 2
        crop = np.zeros((n, n, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        big = cv2.resize(crop, (n * z, n * z), interpolation=cv2.INTER_NEAREST)
        self._loupe_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
        lp.create_image(0, 0, anchor="nw", image=self._loupe_photo)

        def m(v, origin):  # image coordinate -> loupe pixel
            return (v - origin + 0.5) * z

        if corners:
            for i, (px, py) in enumerate(self.edit_quad):
                lx, ly = m(px, x0), m(py, y0)
                col = "#ffffff" if i == self.sel_handle else HANDLE
                lp.create_line(lx - 20, ly, lx + 20, ly, fill=col)
                lp.create_line(lx, ly - 20, lx, ly + 20, fill=col)
            return
        for idx, (kind, side) in enumerate(LINE_ORDER):
            v = getattr(s.lines, kind).get(side)
            w = 2 if idx == self.sel else 1
            if _is_vertical(side):
                lp.create_line(m(v, x0), 0, m(v, x0), LOUPE, fill=COLORS[kind], width=w)
            else:
                lp.create_line(0, m(v, y0), LOUPE, m(v, y0), fill=COLORS[kind], width=w)
        lp.create_text(6, LOUPE - 8, anchor="w", fill="#fff", font=("Helvetica", 10),
                       text=f"x {center[0]:.1f}  y {center[1]:.1f}  ×{z}")

    def update_info(self) -> None:
        t = self.info
        t.configure(state="normal")
        t.delete("1.0", "end")
        s = self.side

        def w(text, *tags):
            t.insert("end", text, tags)

        if s is not None and self.mode_var.get() == "center":
            kind, side = LINE_ORDER[self.sel]
            v = getattr(s.lines, kind).get(side)
            w(f"Selected: {self.sel + 1} {kind} {side} = {v:.2f} px\n", "sel")
        elif s is not None:
            w(f"Corner handle {self.sel_handle + 1}: drag or use arrows, then Apply warp\n", "sel")

        f = self.sides["front"].graded_centering if self.sides["front"] else None
        b = self.sides["back"].graded_centering if self.sides["back"] else None
        present = {k: v for k, v in self.sides.items() if v is not None and v.scan is not None}
        if present:
            checks = standards.target_checks(f, b)
            sev = standards.worst_severity(*(pipeline.severity(sd) for sd in present.values()))
            v = pipeline.make_verdict(present, checks, sev)
            w("\n" + v.headline + "\n", {"send": "ok", "maybe": "warn", "skip": "bad"}[v.level], "big")
            for r in v.reasons:
                w(f"  • {r}\n", "muted")
            w("\nCentering targets\n", "h")
            for chk in checks:
                st = chk.status
                w(f"  {(st or 'n/a').upper():<10} ", {"pass": "ok", "borderline": "warn", "fail": "bad"}.get(st, "muted"))
                w(f"{chk.name}\n")
            preds = pipeline.predictions(present, sev)
            w(f"\nPredicted grade ({preds[0].basis if preds else 'centering and edges'})\n", "h")
            for pr in preds:
                if not pr.available or not pr.has_odds:
                    w(f"  {pr.summary()}\n", "warn")
                    continue
                w(f"  {pr.company} {pr.label}", "big")
                others = ", ".join(f"{lab} {p:.0%}" for lab, p in pr.distribution if lab != pr.label)
                w(f"  {pr.probability:.0%}" + (f"  (else {others})" if others else "") + "\n", "muted")
                if pr.if_marks_real:
                    w(f"     if yellow marks are real wear: {pr.if_marks_real}\n", "warn")
            if preds:
                w(f"  Confidence: {preds[0].confidence}\n",
                  {"high": "ok", "medium": "warn", "low": "bad"}[preds[0].confidence])
            w("\nBest possible (ceiling)\n", "h")
            for co in standards.LADDERS:
                ce = standards.grade_ceiling(co, f, b, sev)
                w(f"  {co} {ce.label}", "big")
                w(f"   {ce.limited_by}\n", "muted")
        for name in ("front", "back"):
            sd = self.sides[name]
            w(f"\n{name.title()}\n", "h")
            if sd is None:
                w("  not loaded\n", "muted")
                continue
            if sd.borderless:
                w("  No border (full-art): centering judged by eye, not measured\n", "warn")
            hint = pipeline.border_hint(sd)
            if hint:
                w("  Hint: tick 'No border' if this card is full-art - the lines may be on artwork\n", "warn")
            c = sd.centering
            w(f"  L/R  {ratio_text(c.lr)}", "big")
            w(f"   ({c.lr[0]:.1f}/{c.lr[1]:.1f} ±{c.sigma_lr:.1f})\n", "muted")
            w(f"       L {c.mm('left'):.2f} mm   R {c.mm('right'):.2f} mm\n", "muted")
            w(f"  T/B  {ratio_text(c.tb)}", "big")
            w(f"   ({c.tb[0]:.1f}/{c.tb[1]:.1f} ±{c.sigma_tb:.1f})\n", "muted")
            w(f"       T {c.mm('top'):.2f} mm   B {c.mm('bottom'):.2f} mm\n", "muted")
            weak = [k for k, v in sd.lines.confidence.items() if v in ("low", "none")]
            if weak:
                w(f"  check inner line(s): {', '.join(weak)}\n", "warn")
            if sd.scan is not None:
                sev = sd.scan.severity
                w(f"  Whitening: {sev} ({len(sd.scan.whitening)} spot(s))\n", "ok" if sev == "none" else "bad")
                for sp in sd.scan.spots[:6]:
                    extra = {"whitening": "", "verify": " - glitter border, verify by eye"}.get(
                        sp.kind, " - check outer line")
                    w(f"    • {sp.location} {sp.area_mm2:.2f} mm²{extra}\n", "muted")
                if sd.scan.unreliable_sides:
                    w(f"  White/silver border ({', '.join(sd.scan.unreliable_sides)}): can't judge\n", "warn")
                if sd.scan.textured_sides:
                    w(f"  Glitter/foil border ({', '.join(sd.scan.textured_sides)}): spots need a visual check\n",
                      "warn")
            sc = sd.surface
            if sc is None or not sc.photos_used:
                w("  Surface: not checked (Add surface photos…)\n", "muted")
            else:
                w(f"  Surface: {sc.severity} ({len(sc.defects)} found, {sc.photos_used} photo(s))\n",
                  "ok" if sc.severity == "none" else "bad")
                for d in sc.defects[:6]:
                    w(f"    • {d.kind} {d.size_mm:.1f} mm, {d.location}\n", "muted")
            if sc is not None:
                for n in sc.notes:
                    w(f"  {n}\n", "warn")
            if sd.quality is not None:
                for lv, msg in sd.quality.issues:
                    w(f"  Photo: {msg}\n", "bad" if lv == "bad" else "warn")

        t.configure(state="disabled")


def main(front: Optional[str] = None, back: Optional[str] = None) -> None:
    app = App(front, back)
    app.mainloop()
