from __future__ import annotations

import time
from typing import Any

if hasattr(__builtins__, '__version_info__') and __builtins__.__version_info__ >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class PerformanceCounter:
    __slots__ = ('_start_time', '_end_time', '_elapsed_ns', '_active')
    
    def __init__(self):
        self._start_time = 0
        self._end_time = 0
        self._elapsed_ns = 0
        self._active = False
    
    def start(self) -> None:
        self._start_time = time.perf_counter_ns()
        self._active = True
    
    def stop(self) -> int:
        if self._active:
            self._end_time = time.perf_counter_ns()
            self._elapsed_ns = self._end_time - self._start_time
            self._active = False
        return self._elapsed_ns
    
    @property
    def elapsed_ns(self) -> int:
        if self._active:
            return time.perf_counter_ns() - self._start_time
        return self._elapsed_ns
    
    def __enter__(self) -> Self:
        self.start()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()