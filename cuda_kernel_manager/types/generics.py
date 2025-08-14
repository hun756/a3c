from __future__ import annotations

from typing import TypeVar, Protocol

T = TypeVar('T')
P = TypeVar('P', bound=Protocol)
DeviceT = TypeVar('DeviceT', bound='AbstractDevice')
StreamT = TypeVar('StreamT', bound='AbstractStream')
MemoryT = TypeVar('MemoryT', bound='AbstractMemory')
KernelT = TypeVar('KernelT', bound='AbstractKernel')
NodeT = TypeVar('NodeT', bound='ExecutionNode')