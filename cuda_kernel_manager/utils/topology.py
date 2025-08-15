from __future__ import annotations

import time
from threading import RLock
from typing import Dict, List, Set, Tuple

from ..types.aliases import DeviceID, EventHandle


class MultiDeviceSynchronizer:
    __slots__ = ('_event_graph', '_synchronization_points', '_lock')
    
    def __init__(self):
        self._event_graph: Dict[EventHandle, Set[EventHandle]] = {}
        self._synchronization_points: List[Tuple[float, List[DeviceID]]] = []
        self._lock = RLock()
    
    def add_synchronization_point(self, devices: List[DeviceID]) -> None:
        timestamp = time.perf_counter()
        self._synchronization_points.append((timestamp, devices))
    
    def create_cross_device_dependency(self, src_device: DeviceID, dst_device: DeviceID,
                                     src_stream: 'CUDAStream', dst_stream: 'CUDAStream') -> None:
        event = src_stream.record_event()
        dst_stream.wait_event(event)
    
    def synchronize_device_group(self, devices: List[DeviceID], 
                                device_manager: 'DeviceManager') -> None:
        for device_id in devices:
            device = device_manager.get_device(device_id)
            device.synchronize()