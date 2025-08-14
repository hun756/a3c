from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from ...types.aliases import ByteSize
from ...types.enums import MemoryBackendType
from ...types.protocols import IMemorySegment


class MemoryBackend(ABC):
    def __init__(self, backend_type: MemoryBackendType):
        self._backend_type = backend_type
        self._capabilities: Dict[str, Any] = {}
        self._numa_nodes: List[int] = []
        self._initialized = False
    
    @property
    def backend_type(self) -> MemoryBackendType:
        return self._backend_type
    
    @property
    def capabilities(self) -> Dict[str, Any]:
        return self._capabilities.copy()
    
    @property
    def numa_nodes(self) -> List[int]:
        return self._numa_nodes.copy()
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @abstractmethod
    def create_segment(self, size: ByteSize, numa_node: int = -1) -> IMemorySegment:
        pass
    
    @abstractmethod
    def get_optimal_alignment(self) -> int:
        pass
    
    @abstractmethod
    def supports_numa(self) -> bool:
        pass
    
    @abstractmethod
    def detect_capabilities(self) -> Dict[str, Any]:
        pass
    
    def initialize(self) -> None:
        if not self._initialized:
            self._capabilities = self.detect_capabilities()
            self._numa_nodes = self._detect_numa_nodes()
            self._initialized = True
    
    def _detect_numa_nodes(self) -> List[int]:
        try:
            import os
            numa_nodes = []
            for node_dir in os.listdir('/sys/devices/system/node'):
                if node_dir.startswith('node') and node_dir[4:].isdigit():
                    numa_nodes.append(int(node_dir[4:]))
            return sorted(numa_nodes)
        except (OSError, ImportError):
            return [0]
    
    def get_memory_info(self) -> Dict[str, Any]:
        return {
            'backend_type': self._backend_type.name,
            'capabilities': self._capabilities,
            'numa_nodes': self._numa_nodes,
            'initialized': self._initialized
        }