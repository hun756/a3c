"""
Core components of TensorOptim library.

This module contains the main components for tensor management,
including the shared tensor manager, tensor pools, and registries.
"""

from .manager import SharedTensorManager
from .tensor import SharedTensor, TensorReference
from .pool import TensorPool
from .registry import TensorRegistry

__all__ = [
    "SharedTensorManager",
    "SharedTensor",
    "TensorReference", 
    "TensorPool",
    "TensorRegistry",
]