import ctypes
import gc
import os
import time
import tracemalloc

import psutil


class MemoryTracker:
    def __init__(self, log_path: str | None = None):
        self._log = open(log_path, "a") if log_path else None
        self._proc = psutil.Process(os.getpid())

    def snapshot(self, label: str) -> float:
        rss_mb = self._proc.memory_info().rss / 1024**2
        ts = time.strftime("%H:%M:%S", time.localtime())
        line = f"{ts}\t{label}\tRSS={rss_mb:.0f}MB\n"
        print(line, end="", flush=True)
        if self._log:
            self._log.write(line)
            self._log.flush()
        return rss_mb

    def collect(self, label: str = "gc_collect") -> float:
        gc.collect()
        return self.snapshot(label)

    def close(self):
        if self._log:
            self._log.close()


_tracker: MemoryTracker | None = None


def init_tracker(log_path: str | None = None):
    global _tracker
    if _tracker is not None:
        _tracker.close()
    tracemalloc.start()
    _tracker = MemoryTracker(log_path)


def snapshot(label: str) -> float | None:
    if _tracker is None:
        return None
    return _tracker.snapshot(label)


def collect(label: str = "gc_collect") -> float | None:
    if _tracker is None:
        return None
    return _tracker.collect(label)


def tracemalloc_snapshot(label: str, top_n: int = 20):
    if _tracker is None:
        return
    snap = tracemalloc.take_snapshot()
    top = snap.statistics("lineno")
    lines = [f"--- tracemalloc top {top_n} at '{label}' ---"]
    for stat in top[:top_n]:
        lines.append(str(stat))
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    if _tracker._log:
        _tracker._log.write(text)
        _tracker._log.flush()


def shutdown():
    global _tracker
    if _tracker is not None:
        _tracker.close()
        _tracker = None


def malloc_trim(pad: int = 0) -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return libc.malloc_trim(pad) == 1
    except Exception:
        return False


def trim_heap(label: str = "trim_heap") -> float | None:
    malloc_trim()
    gc.collect()
    return snapshot(label)
