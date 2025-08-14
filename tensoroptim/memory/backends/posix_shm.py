from __future__ import annotations
import os
import tempfile
from typing import Dict, Any

from .base import MemoryBackend
from ...types.aliases import ByteSize
from ...types.enums import MemoryBackendType
from ...types.protocols import IMemorySegment
from ...exceptions import BackendError
from ..segments.posix_shm import PosixShmSegment


class PosixShmBackend(MemoryBackend):
    def __init__(self):
        super().__init__(MemoryBackendType.POSIX_SHM)
        self._shm_mount_point = "/dev/shm"
    
    def create_segment(self, size: ByteSize, numa_node: int = -1) -> IMemorySegment:
        if not self._initialized:
            self.initialize()
        
        if not self._capabilities.get('posix_shm_available', False):
            raise BackendError("POSIX shared memory not available", backend_type="posix_shm")
        
        return PosixShmSegment(size, numa_node)
    
    def get_optimal_alignment(self) -> int:
        return 4096  # Page size
    
    def supports_numa(self) -> bool:
        return False
    
    def detect_capabilities(self) -> Dict[str, Any]:
        capabilities = {
            'posix_shm_available': False,
            'shm_mount_point': None,
            'max_shm_size': 0,
            'numa_support': False
        }
        
        try:
            if os.path.exists(self._shm_mount_point):
                capabilities['posix_shm_available'] = True
                capabilities['shm_mount_point'] = self._shm_mount_point
                capabilities['max_shm_size'] = self._get_max_shm_size()
            
            test_shm = self._test_shm_creation()
            capabilities['posix_shm_available'] = test_shm
            
        except Exception:
            pass
        
        return capabilities
    
    def _get_max_shm_size(self) -> int:
        try:
            stat = os.statvfs(self._shm_mount_point)
            return stat.f_bavail * stat.f_frsize
        except OSError:
            return 0
    
    def _test_shm_creation(self) -> bool:
        try:
            import mmap
            test_name = f"/tensoroptim_test_{os.getpid()}"
            
            try:
                fd = os.open(f"{self._shm_mount_point}{test_name}", 
                           os.O_CREAT | os.O_RDWR, 0o600)
                os.ftruncate(fd, 4096)
                
                mm = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
                mm.close()
                os.close(fd)
                os.unlink(f"{self._shm_mount_point}{test_name}")
                return True
            except Exception:
                try:
                    os.close(fd)
                    os.unlink(f"{self._shm_mount_point}{test_name}")
                except:
                    pass
                return False
        except Exception:
            return False
    
    def get_shm_info(self) -> Dict[str, Any]:
        return {
            'mount_point': self._shm_mount_point,
            'max_size': self._capabilities.get('max_shm_size', 0),
            'available': self._capabilities.get('posix_shm_available', False)
        }