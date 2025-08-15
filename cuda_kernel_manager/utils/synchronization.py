from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from threading import Event, RLock
from typing import Any, Dict, List, Optional

from ..types.aliases import DeviceID, StreamHandle


class SynchronizationPrimitive(ABC):
    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> bool:
        pass
    
    @abstractmethod
    async def wait_async(self, timeout: Optional[float] = None) -> bool:
        pass


class DeviceBarrier(SynchronizationPrimitive):
    __slots__ = ('_device_count', '_arrived_devices', '_event', '_lock')
    
    def __init__(self, device_count: int):
        self._device_count = device_count
        self._arrived_devices: set[DeviceID] = set()
        self._event = Event()
        self._lock = RLock()
    
    def arrive(self, device_id: DeviceID) -> None:
        with self._lock:
            self._arrived_devices.add(device_id)
            if len(self._arrived_devices) >= self._device_count:
                self._event.set()
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout)
    
    async def wait_async(self, timeout: Optional[float] = None) -> bool:
        try:
            if timeout is not None:
                await asyncio.wait_for(
                    asyncio.to_thread(self._event.wait), 
                    timeout=timeout
                )
            else:
                await asyncio.to_thread(self._event.wait)
            return True
        except asyncio.TimeoutError:
            return False
    
    def reset(self) -> None:
        with self._lock:
            self._arrived_devices.clear()
            self._event.clear()


class StreamSynchronizer:
    __slots__ = ('_dependencies', '_lock')
    
    def __init__(self):
        self._dependencies: Dict[StreamHandle, List[StreamHandle]] = {}
        self._lock = RLock()
    
    def add_dependency(self, dependent_stream: StreamHandle, 
                      prerequisite_stream: StreamHandle) -> None:
        with self._lock:
            if dependent_stream not in self._dependencies:
                self._dependencies[dependent_stream] = []
            self._dependencies[dependent_stream].append(prerequisite_stream)
    
    def get_dependencies(self, stream: StreamHandle) -> List[StreamHandle]:
        with self._lock:
            return self._dependencies.get(stream, []).copy()
    
    def clear_dependencies(self, stream: StreamHandle) -> None:
        with self._lock:
            self._dependencies.pop(stream, None)