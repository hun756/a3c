from __future__ import annotations

from collections import deque
from threading import RLock
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar('T')


class LockFreeQueue(Generic[T]):
    __slots__ = ('_queue', '_lock')
    
    def __init__(self, maxsize: int = 0):
        self._queue: Deque[T] = deque()
        self._lock = RLock()
    
    def put(self, item: T) -> None:
        with self._lock:
            self._queue.append(item)
    
    def get(self) -> Optional[T]:
        with self._lock:
            try:
                return self._queue.popleft()
            except IndexError:
                return None
    
    def size(self) -> int:
        return len(self._queue)
    
    def empty(self) -> bool:
        return len(self._queue) == 0


class WorkStealingQueue(Generic[T]):
    __slots__ = ('_local_queue', '_steal_queue', '_lock', '_worker_id')
    
    def __init__(self, worker_id: int):
        self._local_queue: Deque[T] = deque()
        self._steal_queue: Deque[T] = deque()
        self._lock = RLock()
        self._worker_id = worker_id
    
    def push_local(self, item: T) -> None:
        with self._lock:
            self._local_queue.append(item)
    
    def pop_local(self) -> Optional[T]:
        with self._lock:
            try:
                return self._local_queue.pop()
            except IndexError:
                return None
    
    def steal(self) -> Optional[T]:
        with self._lock:
            try:
                return self._local_queue.popleft()
            except IndexError:
                return None
    
    def size(self) -> int:
        return len(self._local_queue)