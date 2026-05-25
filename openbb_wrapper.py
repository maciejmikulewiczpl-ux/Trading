"""Launch openbb-mcp-server in stdio mode on Windows.

Workaround for upstream bug: openbb_mcp_server.app.app.stdio_main calls
loop.add_signal_handler(SIGINT/SIGTERM), which raises NotImplementedError
on Windows asyncio event loops. We no-op those calls before importing.
"""
import asyncio
import sys


def _noop(*_a, **_kw):
    return None


for _loop_mod, _loop_cls in (
    ("asyncio.proactor_events", "BaseProactorEventLoop"),
    ("asyncio.selector_events", "BaseSelectorEventLoop"),
):
    try:
        mod = __import__(_loop_mod, fromlist=[_loop_cls])
        getattr(mod, _loop_cls).add_signal_handler = _noop
    except Exception:
        pass

sys.argv = ["openbb-mcp", "--transport", "stdio", "--tool-discovery"]
from openbb_mcp_server.app.app import main  # noqa: E402

if __name__ == "__main__":
    main()
