"""Record where a ScanService actually starts a scan; report at session end.

Printing inside a test is swallowed by pytest's capture, which is why the first
attempt looked like it never fired.
"""

import traceback

_current = {"nodeid": "?"}
_events = []


def pytest_configure(config):
    import sys

    sys.path.insert(0, "scripts")
    from ui.services.scan_service import ScanService

    original_init = ScanService.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._probe_born_in = _current["nodeid"]

    ScanService.__init__ = init

    original_start = ScanService._start

    def start(self, *args, **kwargs):
        started = original_start(self, *args, **kwargs)
        if started:
            _events.append(
                (
                    _current["nodeid"],
                    getattr(self, "_probe_born_in", "?"),
                    "".join(traceback.format_stack()[:-1][-12:]),
                )
            )
        return started

    ScanService._start = start


def pytest_runtest_setup(item):
    _current["nodeid"] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    print(f"\n[probe] scans actually started: {len(_events)}")
    for during, born, stack in _events:
        print(f"[probe] --- started during {during}")
        print(f"[probe]     service born in {born}")
        for line in stack.splitlines():
            print("[probe]    ", line)
