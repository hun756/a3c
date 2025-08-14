from __future__ import annotations
from typing import Optional

from .types.aliases import TensorID


class TensorOptimError(Exception):
    def __init__(self, message: str, **kwargs):
        super().__init__(message)
        self.message = message
        self.context = kwargs


class MemoryError(TensorOptimError):
    pass


class AllocationFailure(MemoryError):
    def __init__(self, message: str, requested_size: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.requested_size = requested_size


class MemoryPoolExhausted(MemoryError):
    def __init__(self, message: str, pool_size: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.pool_size = pool_size


class TensorError(TensorOptimError):
    def __init__(self, message: str, tensor_id: Optional[TensorID] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.tensor_id = tensor_id


class TensorCorruption(TensorError):
    def __init__(self, message: str, tensor_id: Optional[TensorID] = None, 
                 expected_checksum: Optional[int] = None, 
                 actual_checksum: Optional[int] = None, **kwargs):
        super().__init__(message, tensor_id=tensor_id, **kwargs)
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum


class TensorNotFound(TensorError):
    pass


class CompressionError(TensorOptimError):
    pass


class BackendError(TensorOptimError):
    def __init__(self, message: str, backend_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.backend_type = backend_type