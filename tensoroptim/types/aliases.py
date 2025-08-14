"""
Type aliases for TensorOptim library.

This module defines type aliases used throughout the library
for better type safety and code clarity.
"""

from typing import NewType
from uuid import UUID

# Core type aliases
TensorID = NewType('TensorID', UUID)
MemoryOffset = NewType('MemoryOffset', int)
ByteSize = NewType('ByteSize', int)