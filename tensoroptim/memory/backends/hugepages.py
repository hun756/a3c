from __future__ import annotations
import mmap
import os
from typing import Dict, Any

from .base import MemoryBackend
from ...types.aliases import ByteSize
from ...types.enums import MemoryBackendType
from ...types.protocols import IMemorySegment
from ...exceptions import BackendError
from ..segments.hugepages import HugePagesSegment


class HugePagesBackend(MemoryBackend):
    def __init__(self):
        super().__init__(MemoryBackendType.HUGEPAGES)
        self._hugepage_sizes = []
        self._default_hugepage_size = 2 * 1024 * 1024  # 2MB
    
    def create_segment(self, size: ByteSize, numa_node: int = -1) -> IMemorySegment:
        if not self._initialized:
            self.initialize()
        
        if not self._capabilities.get('hugepages_available', False):
            raise BackendError("Huge pages not available", backend_type="hugepages")
        
        return HugePagesSegment(size, numa_node, self._default_hugepage_size)
    
    def get_optimal_alignment(self) -> int:
        return self._default_hugepage_size
    
    def supports_numa(self) -> bool:
        return True
    
    def detect_capabilities(self) -> Dict[str, Any]:
        capabilities = {
            'hugepages_available': False,
            'hugepage_sizes': [],
            'transparent_hugepages': False,
            'numa_support': True
        }
        
        try:
            hugepage_sizes = self._detect_hugepage_sizes()
            capabilities['hugepage_sizes'] = hugepage_sizes
            capabilities['hugepages_available'] = len(hugepage_sizes) > 0
            
            if hugepage_sizes:
                self._hugepage_sizes = hugepage_sizes
                self._default_hugepage_size = hugepage_sizes[0]
            
            capabilities['transparent_hugepages'] = self._check_transparent_hugepages()
            
        except Exception:
            pass
        
        return capabilities
    
    def _detect_hugepage_sizes(self) -> list[int]:
        sizes = []
        try:
            hugepages_dir = '/sys/kernel/mm/hugepages'
            if os.path.exists(hugepages_dir):
                for entry in os.listdir(hugepages_dir):
                    if entry.startswith('hugepages-') and entry.endswith('kB'):
                        size_kb = int(entry[10:-2])
                        sizes.append(size_kb * 1024)
            
            sizes.sort(reverse=True)
        except (OSError, ValueError):
            pass
        
        return sizes
    
    def _check_transparent_hugepages(self) -> bool:
        try:
            with open('/sys/kernel/mm/transparent_hugepage/enabled', 'r') as f:
                content = f.read().strip()
                return '[always]' in content or '[madvise]' in content
        except (OSError, IOError):
            return False
    
    def get_hugepage_info(self) -> Dict[str, Any]:
        return {
            'available_sizes': self._hugepage_sizes,
            'default_size': self._default_hugepage_size,
            'transparent_enabled': self._capabilities.get('transparent_hugepages', False)
        }