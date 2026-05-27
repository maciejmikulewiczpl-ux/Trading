"""Optional Windows system-tray status indicator for the ORB live runner.

Design rules:
  - Pure decoration. If anything in this module fails to import or run, the
    trading loop must continue unaffected. All entry points swallow errors.
  - Tray icon runs on pystray's detached thread.
  - Tk window runs on its OWN dedicated daemon thread, created at start()
    and kept alive for the process lifetime. The root window is hidden by
    default via withdraw(); pystray callbacks deiconify/withdraw it via
    root.after(0, ...), which marshals the call onto Tk's thread — the only
    thread-safe way to drive Tk from non-Tk threads on Windows.
  - No two-way control. The UI cannot stop the trader, cancel orders, or
    submit anything. It's read-only.

Usage from paper_orb.py:
    from live.status_ui import StatusController
    ui = StatusController(get_status=build_snapshot)
    ui.start()                 # non-blocking; returns False if unavailable
    ui.set_state("active")     # or "warning"/"halted"/"done"
    ui.stop()                  # on shutdown
"""
from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import datetime
from typing import Callable, Optional

log = logging.getLogger("orb_paper.status_ui")

try:
    import pystray
    from PIL import Image, ImageDraw
    _AVAILABLE = True
except Exception:
    _AVAILABLE = False


_STATE_COLORS = {
    "active":  "#21a300",   # green
    "warning": "#e07a00",   # orange
    "halted":  "#c83232",   # red
    "done":    "#666666",   # grey
}


def _make_icon(color_hex: str):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((6, 6, 58, 58), fill=color_hex, outline="#202020", width=3)
    return img


class StatusController:
    """Tray icon + an on-demand Tk status window."""

    def __init__(self, get_status: Callable[[], dict]):
        self._get_status = get_status
        self._icon: Optional["pystray.Icon"] = None
        self._state: str = "active"
        self._stopping = threading.Event()

        # Tk owns its own daemon thread. _root is set by that thread once
        # mainloop is up; _root_ready guards consumers.
        self._tk_thread: Optional[threading.Thread] = None
        self._root: Optional[tk.Tk] = None
        self._root_ready = threading.Event()
        self._text_widget: Optional[tk.Text] = None
        self._refresh_after_id: Optional[str] = None

    @property
    def available(self) -> bool:
        return _AVAILABLE

    # ----- lifecycle -----
    def start(self) -> bool:
        if not _AVAILABLE:
            log.info("Status UI not available (pystray/Pillow not installed). Skipping.")
            return False
        try:
            self._tk_thread = threading.Thread(
                target=self._tk_loop, name="orb-status-tk", daemon=True
            )
            self._tk_thread.start()
            if not self._root_ready.wait(timeout=5.0):
                log.warning("Status UI Tk thread did not become ready within 5s; aborting UI.")
                return False
            menu = pystray.Menu(
                pystray.MenuItem("Show status...", self._on_show, default=True),
                pystray.MenuItem("Hide tray icon (trader keeps running)", self._on_hide),
            )
            self._icon = pystray.Icon(
                "ORB",
                _make_icon(_STATE_COLORS["active"]),
                "ORB live (active)",
                menu,
            )
            self._icon.run_detached()
            log.info("Status UI tray icon started.")
            return True
        except Exception as e:
            log.warning(f"Status UI failed to start: {e}")
            return False

    def stop(self) -> None:
        self._stopping.set()
        try:
            if self._icon is not None:
                self._icon.stop()
                self._icon = None
        except Exception:
            pass
        # Destroying Tk must happen on Tk's own thread.
        try:
            if self._root is not None:
                self._root.after(0, self._root.destroy)
        except Exception:
            pass
        log.info("Status UI stopped.")

    def set_state(self, state: str) -> None:
        """Update the tray icon color/tooltip. Safe to call from any thread."""
        if not _AVAILABLE or self._icon is None:
            return
        if state == self._state:
            return
        self._state = state
        try:
            self._icon.icon = _make_icon(_STATE_COLORS.get(state, _STATE_COLORS["active"]))
            self._icon.title = f"ORB live ({state})"
        except Exception:
            pass

    # ----- pystray menu callbacks (run on pystray's thread) -----
    def _on_show(self, icon=None, item=None):
        if self._root is None:
            return
        try:
            # after(0, ...) marshals the call onto Tk's mainloop thread.
            self._root.after(0, self._show_window_safely)
        except Exception as e:
            log.warning(f"Show request failed: {e}")

    def _on_hide(self, icon=None, item=None):
        try:
            if self._icon is not None:
                self._icon.stop()
                self._icon = None
        except Exception:
            pass

    # ----- Tk thread (everything below runs on the Tk thread) -----
    def _tk_loop(self) -> None:
        """Owns the Tk root for the lifetime of the process. Runs mainloop.

        Window starts hidden. _show_window_safely reveals it, _hide_window
        (called by WM_DELETE_WINDOW) just withdraws so it can be re-shown.
        """
        try:
            root = tk.Tk()
            root.title("ORB Live - Status")
            root.geometry("620x440")
            root.minsize(520, 360)
            root.withdraw()
            root.protocol("WM_DELETE_WINDOW", self._hide_window)

            txt = tk.Text(root, font=("Consolas", 10), wrap="none",
                          bg="#0e1116", fg="#e6edf3")
            txt.pack(fill="both", expand=True, padx=6, pady=6)
            txt.config(state="disabled")

            footer = tk.Label(root, text="Auto-refreshes every 2 seconds.  Close to dismiss.",
                              anchor="w", padx=6, pady=2, fg="#8590a6")
            footer.pack(fill="x")

            self._root = root
            self._text_widget = txt
            self._root_ready.set()

            root.mainloop()
        except Exception as e:
            log.warning(f"Tk thread crashed: {e}")
        finally:
            self._root = None
            self._text_widget = None

    def _show_window_safely(self) -> None:
        if self._root is None:
            return
        try:
            self._root.deiconify()
            self._root.lift()
            self._root.focus_force()
            if self._refresh_after_id is None:
                self._refresh()
        except Exception as e:
            log.warning(f"Show window failed: {e}")

    def _hide_window(self) -> None:
        if self._root is None:
            return
        try:
            self._root.withdraw()
            if self._refresh_after_id is not None:
                try:
                    self._root.after_cancel(self._refresh_after_id)
                except Exception:
                    pass
                self._refresh_after_id = None
        except Exception:
            pass

    def _refresh(self) -> None:
        if self._root is None or self._text_widget is None:
            return
        if self._stopping.is_set():
            try:
                self._root.destroy()
            except Exception:
                pass
            return
        try:
            snap = self._get_status()
        except Exception as e:
            snap = {"error": f"snapshot failed: {e}"}
        content = self._render(snap)
        try:
            self._text_widget.config(state="normal")
            self._text_widget.delete("1.0", "end")
            self._text_widget.insert("end", content)
            self._text_widget.config(state="disabled")
        except Exception:
            pass
        try:
            self._refresh_after_id = self._root.after(2000, self._refresh)
        except Exception:
            self._refresh_after_id = None

    # ----- rendering -----
    @staticmethod
    def _render(snap: dict) -> str:
        if "error" in snap and len(snap) == 1:
            return f"(snapshot error) {snap['error']}"
        ts = datetime.now().strftime("%H:%M:%S")
        out = []
        out.append(f"ORB Live - refreshed {ts}")
        out.append("=" * 56)
        out.append(f"Phase    : {snap.get('phase', '?')}")
        eq = snap.get('equity')
        if eq is not None:
            out.append(f"Equity   : ${eq:,.2f}")
        pnl = snap.get('day_pnl')
        if pnl is not None:
            out.append(f"Day PnL  : ${pnl:+,.2f}")
        out.append(f"Halted   : {snap.get('halted', False)}  "
                   f"({snap.get('halt_reason', '')})" if snap.get('halted') else
                   f"Halted   : {snap.get('halted', False)}")
        last_update = snap.get('last_update')
        if last_update:
            out.append(f"Last poll: {last_update}")
        out.append("")
        out.append(f"{'Symbol':<7}{'OR high':>10}{'OR low':>10}  {'Status':<28}")
        out.append("-" * 56)
        for sym, st in snap.get('symbols', {}).items():
            orh = f"${st['or_high']:.2f}" if st.get('or_high') is not None else "    -   "
            orl = f"${st['or_low']:.2f}"  if st.get('or_low')  is not None else "    -   "
            status = st.get('status', '?')
            out.append(f"{sym:<7}{orh:>10}{orl:>10}  {status:<28}")
            if st.get('entry_price') is not None:
                out.append(f"         entry ${st['entry_price']:.2f}  "
                           f"stop ${st['stop_price']:.2f}  "
                           f"target ${st['target_price']:.2f}  "
                           f"qty {st.get('shares', '?')}")
        return "\n".join(out)
