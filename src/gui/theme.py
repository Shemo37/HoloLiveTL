"""
Shared dark theme, tooltip, and control helpers for the HoloLiveTL GUI.

One tk_setPalette call themes every classic tk widget; ttk widgets
(Notebook, Progressbar, Scale) are styled explicitly on a clam base.
"""
import tkinter as tk
from tkinter import ttk

# Palette
BG = "#1f1f23"          # window background
BG_CARD = "#27272d"     # card / labelframe background
BG_FIELD = "#323238"    # entries, menus
FG = "#e8e8ea"          # primary text
FG_DIM = "#9a9aa3"      # secondary text
FG_FAINT = "#6b6b74"    # hints
ACCENT = "#31c46e"      # start / success
ACCENT_ACTIVE = "#3ddc84"
DANGER = "#e5484d"      # stop / errors
WARN = "#f5a623"        # loading / warnings
INFO = "#4ea1ff"        # links / info
METER_BG = "#2a2a30"


def setup_theme(root):
    """Apply the dark palette to classic tk widgets and style ttk widgets."""
    root.tk_setPalette(
        background=BG,
        foreground=FG,
        activeBackground=BG_FIELD,
        activeForeground=FG,
        highlightBackground=BG,
        highlightColor=ACCENT,
        insertBackground=FG,
        selectBackground="#3a5f8a",
        selectForeground=FG,
        selectColor=BG_FIELD,      # checkbutton indicator fill
        troughColor=METER_BG,
        disabledForeground=FG_FAINT,
    )

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=FG, fieldbackground=BG_FIELD,
                    bordercolor="#3c3c44", lightcolor=BG_CARD, darkcolor=BG)
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG_CARD, foreground=FG_DIM,
                    padding=(14, 6), borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", BG_FIELD)],
              foreground=[("selected", FG)])
    style.configure("TProgressbar", background=ACCENT, troughcolor=METER_BG,
                    borderwidth=0, thickness=6)
    style.configure("Horizontal.TScale", background=BG_CARD, troughcolor=METER_BG)
    style.configure("TSpinbox", fieldbackground=BG_FIELD, background=BG_CARD,
                    foreground=FG, arrowcolor=FG)


class Tooltip:
    """Hover tooltip. Usage: Tooltip(widget, "explanation")."""

    DELAY_MS = 550

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self._tip, text=self.text, justify="left", wraplength=320,
                     bg="#101014", fg=FG, relief="solid", borderwidth=1,
                     font=("Helvetica", 9), padx=8, pady=5).pack()
        except tk.TclError:
            self._tip = None

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def labeled_scale(parent, label, from_, to, variable, command=None,
                  fmt="{:.2f}", resolution=None, tooltip=None, row=None):
    """A labeled ttk.Scale with a live value readout.

    `variable` is a DoubleVar/IntVar; `fmt` formats the readout. Returns the
    containing frame (packed by the caller unless `row` grid kwargs given).
    """
    frame = tk.Frame(parent, bg=parent.cget("bg") if "bg" in parent.keys() else BG_CARD)
    name = tk.Label(frame, text=label, width=18, anchor="w")
    name.pack(side="left")

    value_label = tk.Label(frame, width=7, anchor="e", fg=FG_DIM)

    def _update_label(*_args):
        try:
            value_label.config(text=fmt.format(variable.get()))
        except (tk.TclError, ValueError):
            pass

    def _on_move(raw):
        if resolution:
            try:
                snapped = round(float(raw) / resolution) * resolution
                if abs(snapped - variable.get()) > 1e-9:
                    variable.set(snapped)
            except (tk.TclError, ValueError):
                pass
        _update_label()
        if command is not None:
            command()

    scale = ttk.Scale(frame, from_=from_, to=to, variable=variable,
                      orient="horizontal", command=_on_move)
    scale.pack(side="left", fill="x", expand=True, padx=(4, 6))
    value_label.pack(side="left")
    _update_label()

    if tooltip:
        Tooltip(name, tooltip)
        Tooltip(scale, tooltip)
    return frame
