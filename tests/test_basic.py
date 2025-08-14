"""
Basic tests for TensorOptim library.

This module contains basic functionality tests to ensure the library
works correctly after modularization.
"""

import pytest
import torch
import numpy as np

from tensoroptim import (
    TensorManager,
    create_manager,
    create_memory_efficient_manager,
    MemoryBackendType,
    CompressionType
)


class TestBasicFunctionality:
    """Test basic tensor sharing functionality."""
    
    def test_manager_creation(self):
        """Test manager creation with different backends."""
        manager = create_manager()
        assert manager is not None
        assert not manager.is_closed()
        manager.close()
        assert manager.is_closed()
    
    def test_tensor_sharing_basic(self):
        """Test basic tensor sharing and retrieval."""
        manager = create_memory_efficient_manager()
        
        try:
            tensor = torch.randn(100, 100, dtype=torch.float32)
            
            shared = manager.share_tensor_persistent(tensor)
            assert shared is not None
            assert shared.is_valid
            assert shared.shape == (100, 100)
            assert shared.dtype == torch.float32
            
            retrieved_tensor = shared.get()
            assert torch.allclose(tensor, retrieved_tensor)
            
            shared.detach_optimized()
            
        finally:
            manager.close()
    
    def test_tensor_sharing_context_manager(self):
        """Test tensor sharing with context manager."""
        manager = create_memory_efficient_manager()
        
        try:
            tensor = torch.randn(50, 50, dtype=torch.float32)
            
            with manager.share_tensor(tensor) as shared:
                assert shared.is_valid
                retrieved = shared.get()
                assert torch.allclose(tensor, retrieved)
            
            # Tensor should be automatically cleaned up
            
        finally:
            manager.close()
    
    def test_tensor_retrieval_by_id(self):
        """Test tensor retrieval by ID."""
        manager = create_memory_efficient_manager()
        
        try:
            tensor = torch.randn(75, 75, dtype=torch.float32)
            
            shared = manager.share_tensor_persistent(tensor)
            tensor_id = shared.tensor_id
            
            retrieved_shared = manager.get_tensor(tensor_id)
            assert retrieved_shared is not None
            assert retrieved_shared.tensor_id == tensor_id
            
            retrieved_tensor = retrieved_shared.get()
            assert torch.allclose(tensor, retrieved_tensor)
            
            shared.detach_optimized()
            
        finally:
            manager.close()
    
    def test_multiple_tensors(self):
        """Test sharing multiple tensors."""
        manager = create_memory_efficient_manager()
        
        try:
            tensors = [
                torch.randn(10, 10, dtype=torch.float32),
                torch.randn(20, 20, dtype=torch.float64),
                torch.randint(0, 100, (30, 30), dtype=torch.int32)
            ]
            
            shared_tensors = []
            for tensor in tensors:
                shared = manager.share_tensor_persistent(tensor)
                shared_tensors.append(shared)
            
            for original, shared in zip(tensors, shared_tensors):
                retrieved = shared.get()
                if original.dtype == torch.int32:
                    assert torch.equal(original, retrieved)
                else:
                    assert torch.allclose(original, retrieved)
            
            for shared in shared_tensors:
                shared.detach_optimized()
                
        finally:
            manager.close()
    
    def test_tensor_metrics(self):
        """Test tensor access metrics."""
        manager = create_memory_efficient_manager()
        
        try:
            tensor = torch.randn(25, 25, dtype=torch.float32)
            shared = manager.share_tensor_persistent(tensor)
            
            for _ in range(5):
                _ = shared.get()
            
            metrics = shared.access_metrics
            assert metrics['access_count'] >= 5
            assert metrics['performance']['get_calls'] >= 5
            
            shared.detach_optimized()
            
        finally:
            manager.close()
    
    def test_manager_stats(self):
        """Test manager statistics."""
        manager = create_memory_efficient_manager()
        
        try:
            tensors = []
            for i in range(3):
                tensor = torch.randn(10 + i*10, 10 + i*10, dtype=torch.float32)
                shared = manager.share_tensor_persistent(tensor)
                tensors.append(shared)
            
            stats = manager.get_performance_metrics()
            assert 'performance' in stats
            assert 'registry' in stats
            assert 'allocator' in stats
            
            memory_summary = manager.get_memory_usage_summary()
            assert 'backend' in memory_summary
            assert 'memory_usage' in memory_summary
            
            for shared in tensors:
                shared.detach_optimized()
                
        finally:
            manager.close()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_closed_manager_operations(self):
        """Test operations on closed manager."""
        manager = create_memory_efficient_manager()
        manager.close()
        
        tensor = torch.randn(10, 10)
        
        with pytest.raises(Exception):
            manager.share_tensor_persistent(tensor)
    
    def test_invalid_tensor_access(self):
        """Test accessing invalid tensor."""
        manager = create_memory_efficient_manager()
        
        try:
            tensor = torch.randn(10, 10)
            shared = manager.share_tensor_persistent(tensor)
            
            shared.detach_optimized()
            
            assert not shared.is_valid
            
        finally:
            manager.close()


if __name__ == '__main__':
    pytest.main([__file__])