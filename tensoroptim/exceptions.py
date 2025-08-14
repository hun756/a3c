"""
Exception classes for TensorOptim library.

This module defines all custom exceptions used throughout the TensorOptim library,
providing structured error handling for tensor management operations.
"""

from __future__ import annotations
from typing import Any, Optional
from uuid import UUID

TensorID = UUID


class TensorError(Exception):
    """Base exception class for all tensor-related errors.
    
    Attributes:
        error_code: Numeric error code for programmatic handling
        context: Additional context information about the error
        tensor_id: ID of the tensor that caused the error (if applicable)
    """
    __slots__ = ('error_code', 'context', 'tensor_id')
    
    def __init__(
        self, 
        message: str, 
        error_code: int = 0, 
        context: Any = None, 
        tensor_id: Optional[TensorID] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context
        self.tensor_id = tensor_id
    
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.tensor_id:
            base_msg += f" (Tensor ID: {self.tensor_id})"
        if self.error_code:
            base_msg += f" (Error Code: {self.error_code})"
        return base_msg


class MemoryPoolExhausted(TensorError):
    """Raised when memory pool cannot allocate more memory.
    
    This exception is raised when the memory pool has insufficient
    space to satisfy an allocation request.
    """
    pass


class TensorCorruption(TensorError):
    """Raised when tensor data integrity check fails.
    
    This exception indicates that tensor data has been corrupted,
    either during storage, transmission, or retrieval.
    """
    pass


class AllocationFailure(TensorError):
    """Raised when memory allocation fails.
    
    This exception is raised when the underlying memory allocation
    system cannot satisfy a request, typically due to system constraints.
    """
    pass


class TensorNotFound(TensorError):
    """Raised when a requested tensor cannot be found.
    
    This exception is raised when attempting to access a tensor
    that doesn't exist in the registry or has been deallocated.
    """
    pass


class InvalidTensorState(TensorError):
    """Raised when tensor is in an invalid state for the requested operation.
    
    This exception is raised when attempting to perform an operation
    on a tensor that is not in the appropriate lifecycle state.
    """
    pass


class CompressionError(TensorError):
    """Raised when tensor compression/decompression fails.
    
    This exception is raised when compression or decompression
    operations fail due to codec issues or data corruption.
    """
    pass


class BackendNotAvailable(TensorError):
    """Raised when a requested memory backend is not available.
    
    This exception is raised when attempting to use a memory backend
    that is not supported on the current system or not properly configured.
    """
    pass


class ConfigurationError(TensorError):
    """Raised when there's an error in system configuration.
    
    This exception is raised when the library detects invalid
    configuration parameters or missing system requirements.
    """
    pass


class ConcurrencyError(TensorError):
    """Raised when concurrent access causes conflicts.
    
    This exception is raised when concurrent operations on tensors
    result in race conditions or deadlocks.
    """
    pass