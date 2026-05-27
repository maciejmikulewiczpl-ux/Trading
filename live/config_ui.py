"""ORB parameter editor — a small Tkinter GUI (stdlib only, no extra deps).

View and edit every tunable the live runner uses for LONG and SHORT trades,
pre-filled with the project's validated defaults. Each field has an "ⓘ" button
explaining the BASIS for that setting (drawn from the backtest investigation).
Saving writes live/orb_config.json; the live runner reads it at the start of the
NEXT session (a running bot keeps the params it launched with).

Run:
    python live/config_ui.py
    .\\.venv\\Scripts\\python.exe live\\config_ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from live import config as orb_config  # noqa: E402
from live.config import SETTINGS, GROUPS, DEFAULTS, BY_KEY  # noqa: E402

ABOUT = (
    "These parameters drive the ORB live runner (live/paper_orb.py).\n\n"
    "Defaults are the values validated by this project's backtests:\n"
    " • Long side: 15-min opening range, 2.0R target, entries cut off at\n"
    "   11:30 ET, no break-even lift (all from compare_*.py / sweep_orb.py).\n"
    " • Short side is REGIME-GATED: a naive always-on short loses out-of-sample,\n"
    "   but shorting only when SPY has closed below its 20-day SMA for 3\n"
    "   straight days adds PnL and trims drawdown across 2021-2026 (including\n"
    "   the 2022 bear). TSLA is excluded — single-name squeezes were the loss.\n"
    " • Deployed flip-free (max flips = 0): higher total PnL than the flip\n"
    "   version in backtest, and simpler/safer live.\n\n"
    "Click ⓘ next to any field for that setting's specific basis.\n"
    "Changes apply on the NEXT session start, never to a running bot."
)


def _value_to_str(kind: str, val) -> str:
    if kind == "csv":
        return ", ".join(val or [])
    if kind == "optfloat":
        return "" if val in (None, "") else str(val)
    if kind == "bool":
        return val
    return "" if val is None else str(val)


def _str_to_value(kind: str, raw, label: str):
    """Convert a widget value to the typed config value. Raises ValueError with a
    friendly message on bad input."""
    if kind == "bool":
        return bool(raw)
    s = str(raw).strip()
    if kind == "csv":
        return [x.strip().upper() for x in s.split(",") if x.strip()]
    if kind == "str":
        return s
    if kind == "time":
        if not s:
            return ""
        try:
            hh, mm = s.split(":")
            h, m = int(hh), int(mm)
            assert 0 <= h <= 23 and 0 <= m <= 59
        except Exception:
            raise ValueError(f"{label}: use HH:MM (24h) or blank, got '{s}'")
        return f"{h:02d}:{m:02d}"
    if kind == "optfloat":
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"{label}: must be a number or blank, got '{s}'")
    if kind == "int":
        try:
            return int(float(s))
        except ValueError:
            raise ValueError(f"{label}: must be a whole number, got '{s}'")
    if kind == "float":
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"{label}: must be a number, got '{s}'")
    return s


def _validate_range(setting, value):
    if value is None or setting.kind not in ("int", "float", "optfloat"):
        return
    if setting.minv is not None and value < setting.minv:
        raise ValueError(f"{setting.label}: must be ≥ {setting.minv}, got {value}")
    if setting.maxv is not None and value > setting.maxv:
        raise ValueError(f"{setting.label}: must be ≤ {setting.maxv}, got {value}")


def main() -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as e:
        print(f"Tkinter not available ({e}). Edit live/orb_config.json by hand, "
              f"or run on a machine with a display.", file=sys.stderr)
        return 1

    cfg = orb_config.load_config()

    root = tk.Tk()
    root.title("ORB Parameters")
    root.geometry("760x720")

    # Header
    header = ttk.Frame(root, padding=(12, 10))
    header.pack(fill="x")
    ttk.Label(header, text="ORB Trading Parameters",
              font=("Segoe UI", 14, "bold")).pack(side="left")
    ttk.Button(header, text="ⓘ About these settings",
               command=lambda: messagebox.showinfo("About ORB parameters", ABOUT)
               ).pack(side="right")
    cfg_path_lbl = ttk.Label(root, text=f"File: {orb_config.CONFIG_PATH}",
                             foreground="#666")
    cfg_path_lbl.pack(fill="x", padx=12)

    # Scrollable body (in case the window is short).
    body = ttk.Frame(root)
    body.pack(fill="both", expand=True, padx=8, pady=6)
    canvas = tk.Canvas(body, highlightthickness=0)
    scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

    vars_by_key: dict[str, object] = {}

    def info_popup(s):
        rng = ""
        if s.minv is not None or s.maxv is not None:
            lo = "" if s.minv is None else f"min {s.minv}"
            hi = "" if s.maxv is None else f"max {s.maxv}"
            rng = f"\n\nAllowed: {', '.join(x for x in (lo, hi) if x)}"
        default = _value_to_str(s.kind, DEFAULTS[s.key])
        messagebox.showinfo(s.label, f"{s.help}{rng}\n\nDefault: {default!r}")

    for group in GROUPS:
        frame = ttk.LabelFrame(inner, text=group, padding=(10, 8))
        frame.pack(fill="x", expand=True, pady=6, padx=4)
        frame.columnconfigure(1, weight=1)
        row = 0
        for s in (x for x in SETTINGS if x.group == group):
            ttk.Label(frame, text=s.label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            if s.kind == "bool":
                var = tk.BooleanVar(value=bool(cfg[s.key]))
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, sticky="w")
            else:
                var = tk.StringVar(value=_value_to_str(s.kind, cfg[s.key]))
                ttk.Entry(frame, textvariable=var, width=42).grid(row=row, column=1, sticky="we")
            vars_by_key[s.key] = var
            ttk.Button(frame, text="ⓘ", width=3,
                       command=lambda s=s: info_popup(s)).grid(row=row, column=2, padx=(8, 0))
            row += 1

    # Footer buttons
    footer = ttk.Frame(root, padding=(12, 10))
    footer.pack(fill="x")
    status = ttk.Label(footer, text="", foreground="#2a7")
    status.pack(side="left")

    def collect_and_validate():
        values = {}
        for s in SETTINGS:
            raw = vars_by_key[s.key].get()
            val = _str_to_value(s.kind, raw, s.label)
            _validate_range(s, val)
            values[s.key] = val
        # Soft check: short symbols ⊆ watchlist.
        extra = set(values["short_symbols"]) - set(values["watchlist"])
        if extra:
            raise ValueError(f"Short-eligible symbols not in the watchlist: "
                             f"{', '.join(sorted(extra))}")
        return values

    def on_save():
        try:
            values = collect_and_validate()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))
            return
        path = orb_config.save_config(values)
        status.config(text="Saved ✓  (applies next session)")
        messagebox.showinfo("Saved",
                            f"Wrote {path.name}.\n\nThe live runner picks these up at the "
                            f"START of the next session. A bot already running keeps the "
                            f"parameters it launched with.")

    def on_reset():
        if not messagebox.askyesno("Reset to defaults",
                                   "Reset every field to the validated defaults? "
                                   "(does not save until you click Save)"):
            return
        for s in SETTINGS:
            v = vars_by_key[s.key]
            if s.kind == "bool":
                v.set(bool(DEFAULTS[s.key]))
            else:
                v.set(_value_to_str(s.kind, DEFAULTS[s.key]))
        status.config(text="Reset to defaults (not yet saved)")

    def on_reload():
        fresh = orb_config.load_config()
        for s in SETTINGS:
            v = vars_by_key[s.key]
            if s.kind == "bool":
                v.set(bool(fresh[s.key]))
            else:
                v.set(_value_to_str(s.kind, fresh[s.key]))
        status.config(text="Reloaded from file")

    ttk.Button(footer, text="Close", command=root.destroy).pack(side="right", padx=4)
    ttk.Button(footer, text="Save", command=on_save).pack(side="right", padx=4)
    ttk.Button(footer, text="Reset to defaults", command=on_reset).pack(side="right", padx=4)
    ttk.Button(footer, text="Reload", command=on_reload).pack(side="right", padx=4)

    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
