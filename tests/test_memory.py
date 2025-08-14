import pytest
import torch
from unittest.mock import Mock, patch

from tensoroptim.types import ByteSize, MemoryOffset
from tensoroptim.memory.backends import HugePagesBackend, PosixShmBackend
from tensoroptim.memory.segments import HugePagesSegment, PosixShmSegment, OptimizedBuffer
from tensoroptim.memory.allocators import SlabAllocator
from tensoroptim.exceptions import AllocationFailure


class TestHugePagesBackend:
    def setup_method(self):
        self.backend = HugePagesBackend()
    
    def test_backend_initialization(self):
        assert self.backend.backend_type.name == "HUGEPAGES"
        assert not self.backend.is_initialized
    
    def test_backend_capabilities_detection(self):
        capabilities = self.backend.detect_capabilities()
        assert 'hugepages_available' in capabilities
        assert 'hugepage_sizes' in capabilities
        assert 'transparent_hugepages' in capabilities
        assert 'numa_support' in capabilities
    
    def test_optimal_alignment(self):
        alignment = self.backend.get_optimal_alignment()
        assert alignment > 0
        assert alignment % 1024 == 0
    
    def test_numa_support(self):
        assert self.backend.supports_numa() is True


class TestPosixShmBackend:
    def setup_method(self):
        self.backend = PosixShmBackend()
    
    def test_backend_initialization(self):
        assert self.backend.backend_type.name == "POSIX_SHM"
        assert not self.backend.is_initialized
    
    def test_backend_capabilities_detection(self):
        capabilities = self.backend.detect_capabilities()
        assert 'posix_shm_available' in capabilities
        assert 'shm_mount_point' in capabilities
        assert 'max_shm_size' in capabilities
        assert 'numa_support' in capabilities
    
    def test_optimal_alignment(self):
        alignment = self.backend.get_optimal_alignment()
        assert alignment == 4096
    
    def test_numa_support(self):
        assert self.backend.supports_numa() is False


class TestOptimizedBuffer:
    def test_buffer_creation(self):
        size = ByteSize(1024)
        buffer = OptimizedBuffer(size)
        
        assert buffer.size == size
        assert buffer.virtual_address > 0
        assert buffer.numa_node == -1
    
    def test_buffer_read_write(self):
        size = ByteSize(1024)
        buffer = OptimizedBuffer(size)
        
        test_data = b"Hello, World!"
        offset = MemoryOffset(0)
        
        buffer.write_vectorized(offset, test_data)
        read_data = buffer.read_vectorized(offset, ByteSize(len(test_data)))
        
        assert bytes(read_data) == test_data
    
    def test_buffer_bounds_checking(self):
        size = ByteSize(100)
        buffer = OptimizedBuffer(size)
        
        with pytest.raises(ValueError, match="beyond buffer bounds"):
            buffer.read_vectorized(MemoryOffset(90), ByteSize(20))
        
        with pytest.raises(ValueError, match="beyond buffer bounds"):
            buffer.write_vectorized(MemoryOffset(90), b"this is too long for the remaining space")
    
    def test_buffer_alignment(self):
        size = ByteSize(1024)
        alignment = 128
        buffer = OptimizedBuffer(size, alignment=alignment)
        
        assert buffer.virtual_address % alignment == 0
    
    def test_buffer_cleanup(self):
        size = ByteSize(1024)
        buffer = OptimizedBuffer(size)
        
        buffer.close()


class TestSlabAllocator:
    def setup_method(self):
        self.mock_segment = Mock()
        self.mock_segment.size = ByteSize(10 * 1024 * 1024)  # 10MB
        self.object_size = ByteSize(1024)  # 1KB objects
        self.slab_size = ByteSize(1024 * 1024)  # 1MB slabs
        
        self.allocator = SlabAllocator(
            self.mock_segment,
            self.object_size,
            self.slab_size
        )
    
    def test_allocator_initialization(self):
        assert self.allocator.object_size == self.object_size
        assert self.allocator.slab_size == self.slab_size
        assert self.allocator.objects_per_slab == 1024 
    
    def test_allocate_aligned(self):
        offset, segment = self.allocator.allocate_aligned(ByteSize(512))
        
        assert isinstance(offset, int)
        assert segment == self.mock_segment
    
    def test_allocate_oversized(self):
        with pytest.raises(AllocationFailure, match="exceeds slab object size"):
            self.allocator.allocate_aligned(ByteSize(2048))
    
    def test_deallocate_fast(self):
        offset, segment = self.allocator.allocate_aligned(ByteSize(512))
        
        self.allocator.deallocate_fast(offset, segment)
    
    def test_utilization_stats(self):
        offsets = []
        for _ in range(5):
            offset, _ = self.allocator.allocate_aligned(ByteSize(512))
            offsets.append(offset)
        
        stats = self.allocator.get_utilization_stats()
        
        assert 'utilization' in stats
        assert 'fragmentation' in stats
        assert 'allocation_count' in stats
        assert 'deallocation_count' in stats
        assert stats['allocation_count'] == 5.0
    
    def test_fragmentation_ratio(self):
        ratio = self.allocator.get_fragmentation_ratio()
        assert 0.0 <= ratio <= 1.0
    
    def test_reallocate(self):
        offset, segment = self.allocator.allocate_aligned(ByteSize(512))
        
        self.mock_segment.read_vectorized.return_value = b"test_data"
        
        new_offset = self.allocator.reallocate(
            offset, 
            ByteSize(512), 
            ByteSize(256), 
            segment
        )
        
        assert new_offset == offset


class TestMemoryIntegration:
    def test_backend_segment_integration(self):
        """Test that backends can create segments successfully"""
        backend = PosixShmBackend()
        backend.initialize()
        
        if backend.capabilities.get('posix_shm_available', False):
            try:
                segment = backend.create_segment(ByteSize(4096))
                assert segment.size == ByteSize(4096)
                segment.close()
            except Exception:
                pytest.skip("POSIX shared memory not available")
    
    def test_allocator_segment_integration(self):
        """Test allocator with real segment"""
        segment = OptimizedBuffer(ByteSize(1024 * 1024))
        allocator = SlabAllocator(segment, ByteSize(1024))
        
        offsets = []
        for _ in range(10):
            offset, _ = allocator.allocate_aligned(ByteSize(512))
            offsets.append(offset)
        
        assert len(set(offsets)) == len(offsets)
        
        for offset in offsets:
            allocator.deallocate_fast(offset, segment)
        
        segment.close()


if __name__ == "__main__":
    pytest.main([__file__])