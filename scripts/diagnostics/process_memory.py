"""This process's working set and commit, from the OS, in MB.

One cheap syscall (``GetProcessMemoryInfo`` on Windows, ``/proc/self/statm``
on Linux). psutil is not a dependency. Import-light (no Qt), so any process
can log its own size; the desk's per-minute thread-CPU gauge uses it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_GET_PROCESS_MEMORY_INFO = None


def _get_process_memory_info():
    """A private psapi binding, so shared ``ctypes.windll`` prototypes stay untouched."""
    global _GET_PROCESS_MEMORY_INFO
    if _GET_PROCESS_MEMORY_INFO is None:
        function = ctypes.WinDLL("psapi").GetProcessMemoryInfo  # type: ignore[attr-defined]
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        function.restype = ctypes.c_int
        _GET_PROCESS_MEMORY_INFO = function
    return _GET_PROCESS_MEMORY_INFO


def process_memory() -> dict[str, float]:
    """This process's working set and commit in MB; {} where unsupported."""
    mb = 1024.0 * 1024.0
    try:
        if sys.platform.startswith("win"):
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ok = _get_process_memory_info()(
                ctypes.c_void_p(-1),  # GetCurrentProcess() pseudo-handle
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                return {}
            return {
                "rss_mb": round(counters.WorkingSetSize / mb, 1),
                "commit_mb": round(counters.PrivateUsage / mb, 1),
                "peak_commit_mb": round(counters.PeakPagefileUsage / mb, 1),
            }
        if sys.platform.startswith("linux"):
            pages = Path("/proc/self/statm").read_text(encoding="ascii").split()
            page = os.sysconf("SC_PAGE_SIZE")
            return {"rss_mb": round(int(pages[1]) * page / mb, 1)}
    except Exception:
        return {}
    return {}
