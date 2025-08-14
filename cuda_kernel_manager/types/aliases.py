from typing import NewType, TypeAlias

DeviceID = NewType('DeviceID', int)
StreamHandle = NewType('StreamHandle', int)
MemoryPtr = NewType('MemoryPtr', int)
ModuleHandle = NewType('ModuleHandle', int)
FunctionHandle = NewType('FunctionHandle', int)
EventHandle = NewType('EventHandle', int)
GraphHandle = NewType('GraphHandle', int)
ProfilerHandle = NewType('ProfilerHandle', int)

KernelHash: TypeAlias = str
SourceHash: TypeAlias = str
BinaryHash: TypeAlias = str
GraphHash: TypeAlias = str

__all__ = [
    'DeviceID',
    'StreamHandle', 
    'MemoryPtr',
    'ModuleHandle',
    'FunctionHandle',
    'EventHandle',
    'GraphHandle',
    'ProfilerHandle',
    'KernelHash',
    'SourceHash',
    'BinaryHash',
    'GraphHash',
]