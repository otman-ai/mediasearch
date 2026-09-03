import time
from contextlib import contextmanager

import pytest

# nodeid -> {phase label: seconds}
_TIMINGS: dict[str, dict[str, float]] = {}


@pytest.fixture
def timer(request):
    """Context manager to time a phase of a test.

    Usage:
        with timer("insert"):
            ...
        with timer("search"):
            ...

    Recorded phases are printed in pytest's terminal summary (no ``-s`` needed).
    """
    records = _TIMINGS.setdefault(request.node.nodeid, {})

    @contextmanager
    def _time(label):
        start = time.perf_counter()
        try:
            yield
        finally:
            records[label] = time.perf_counter() - start

    return _time


def pytest_terminal_summary(terminalreporter):
    if not _TIMINGS:
        return
    terminalreporter.section("timings")
    for nodeid, records in _TIMINGS.items():
        parts = " | ".join(f"{label}: {secs:.3f}s" for label, secs in records.items())
        total = sum(records.values())
        terminalreporter.write_line(f"{nodeid}  {parts}  (total {total:.3f}s)")
