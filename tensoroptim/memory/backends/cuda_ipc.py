from __future__ import annotations
from typing import Dict, Any

from .base import MemoryBackend
from ...types.aliases import ByteSize
from ...types.enums import MemoryBackendType
from ...types.protocols import IMemorySegment
from ...exceptions import BackendError


class CudaIpcBackend(MemoryBackend):
    def __init__(self):
        super().__init__(MemoryBackendType.CUDA_IPC)
        self._cuda_available = False
        self._cuda_devices = []
    
    def create_segment(self, size: ByteSize, numa_node: int = -1) -> IMemorySegment:
        if not self._initialized:
            self.initialize()
        
        if not self._capabilities.get('cuda_available', False):
            raise BackendError("CUDA not available", backend_type="cuda_ipc")
        
        if not self._capabilities.get('ipc_support', False):
            raise BackendError("CUDA IPC not supported", backend_type="cuda_ipc")
        
        raise NotImplementedError("CUDA IPC segment creation not yet implemented")
    
    def get_optimal_alignment(self) -> int:
        return 256  # CUDA memory alignment
    
    def supports_numa(self) -> bool:
        return True
    
    def detect_capabilities(self) -> Dict[str, Any]:
        capabilities = {
            'cuda_available': False,
            'ipc_support': False,
            'device_count': 0,
            'devices': [],
            'numa_support': True
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                capabilities['cuda_available'] = True
                capabilities['device_count'] = torch.cuda.device_count()
                
                devices = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    devices.append({
                        'id': i,
                        'name': props.name,
                        'memory': props.total_memory,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                
                capabilities['devices'] = devices
                capabilities['ipc_support'] = self._check_ipc_support()
                
        except ImportError:
            pass
        except Exception:
            pass
        
        return capabilities
    
    def _check_ipc_support(self) -> bool:
        try:
            import torch
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1, device='cuda:0')
                handle = test_tensor.share_memory_()
                return True
        except Exception:
            pass
        return False
    
    def get_cuda_info(self) -> Dict[str, Any]:
        return {
            'available': self._capabilities.get('cuda_available', False),
            'device_count': self._capabilities.get('device_count', 0),
            'devices': self._capabilities.get('devices', []),
            'ipc_support': self._capabilities.get('ipc_support', False)
        }