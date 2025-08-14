import pytest
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensoroptim
from tensoroptim import (
    get_default_manager,
    create_manager,
    create_memory_efficient_manager,
    TensorManager,
    MemoryBackendType
)


class TestBasicIntegration:
    def test_import_tensoroptim(self):
        """Test that tensoroptim can be imported successfully"""
        assert tensoroptim.__version__ == "1.0.0"
        assert len(tensoroptim.__all__) > 30
    
    def test_factory_functions(self):
        """Test factory functions work correctly"""
        # Skip complex managers on Windows due to platform limitations
        import os
        if os.name == 'nt':
            pytest.skip("Complex memory backends not supported on Windows")
        
        default_manager = get_default_manager()
        assert isinstance(default_manager, TensorManager)
        
        custom_manager = create_manager(backend=MemoryBackendType.POSIX_SHM)
        assert isinstance(custom_manager, TensorManager)
        
        efficient_manager = create_memory_efficient_manager()
        assert isinstance(efficient_manager, TensorManager)
    
    def test_manager_lifecycle(self):
        """Test manager creation and cleanup"""
        import os
        if os.name == 'nt':
            pytest.skip("Complex memory backends not supported on Windows")
            
        manager = create_manager()
        
        assert not manager.is_closed()
        
        manager.close()
        assert manager.is_closed()
    
    def test_manager_context_manager(self):
        """Test manager as context manager"""
        import os
        if os.name == 'nt':
            pytest.skip("Complex memory backends not supported on Windows")
            
        with create_manager() as manager:
            assert not manager.is_closed()
        
        assert manager.is_closed()


class TestTensorOperations:
    def setup_method(self):
        self.manager = create_memory_efficient_manager()
    
    def teardown_method(self):
        self.manager.close()
    
    def test_basic_tensor_sharing(self):
        """Test basic tensor sharing functionality"""
        original_tensor = torch.randn(10, 10)
        
        with self.manager.share_tensor(original_tensor) as shared:
            retrieved_tensor = shared.get()
            
            assert torch.allclose(original_tensor, retrieved_tensor)
            
            modified_tensor = original_tensor * 2
            shared.set(modified_tensor)
            
            retrieved_again = shared.get()
            assert torch.allclose(modified_tensor, retrieved_again)
    
    def test_tensor_properties(self):
        """Test tensor property access"""
        original_tensor = torch.randn(5, 8, dtype=torch.float32)
        
        with self.manager.share_tensor(original_tensor) as shared:
            assert shared.shape == (5, 8)
            assert shared.dtype == torch.float32
            assert shared.device == torch.device("cpu")
            assert shared.is_valid
    
    def test_tensor_metrics(self):
        """Test tensor access metrics"""
        original_tensor = torch.randn(3, 3)
        
        with self.manager.share_tensor(original_tensor) as shared:
            _ = shared.get()
            
            metrics = shared.access_metrics
            assert metrics['access_count'] > 0
            assert 'last_access' in metrics
            assert 'performance' in metrics
    
    def test_multiple_tensors(self):
        """Test sharing multiple tensors"""
        tensors = [torch.randn(i+1, i+1) for i in range(5)]
        shared_tensors = []
        
        try:
            for tensor in tensors:
                shared = self.manager.share_tensor_persistent(tensor)
                shared_tensors.append(shared)
            
            for original, shared in zip(tensors, shared_tensors):
                retrieved = shared.get()
                assert torch.allclose(original, retrieved)
        
        finally:
            for shared in shared_tensors:
                shared.detach()


class TestConcurrentAccess:
    def setup_method(self):
        self.manager = create_manager()
    
    def teardown_method(self):
        self.manager.close()
    
    def test_concurrent_tensor_access(self):
        """Test concurrent access to shared tensors"""
        original_tensor = torch.randn(100, 100)
        shared = self.manager.share_tensor_persistent(original_tensor)
        
        def access_tensor(thread_id):
            try:
                for _ in range(10):
                    tensor = shared.get()
                    assert tensor.shape == (100, 100)
                    time.sleep(0.001)  # Small delay
                return f"Thread {thread_id} completed"
            except Exception as e:
                return f"Thread {thread_id} failed: {e}"
        
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(access_tensor, i) for i in range(4)]
                results = [future.result() for future in as_completed(futures)]
            
            for result in results:
                assert "completed" in result
        
        finally:
            shared.detach()
    
    def test_concurrent_tensor_modification(self):
        """Test concurrent tensor modifications"""
        original_tensor = torch.zeros(50, 50)
        shared = self.manager.share_tensor_persistent(original_tensor)
        
        def modify_tensor(thread_id, value):
            try:
                tensor = torch.full((50, 50), value)
                shared.set(tensor)
                return f"Thread {thread_id} set value {value}"
            except Exception as e:
                return f"Thread {thread_id} failed: {e}"
        
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(modify_tensor, i, i+1) 
                    for i in range(3)
                ]
                results = [future.result() for future in as_completed(futures)]
            
            for result in results:
                assert "set value" in result
            
            final_tensor = shared.get()
            unique_values = torch.unique(final_tensor)
            assert len(unique_values) == 1
            assert unique_values[0].item() in [1, 2, 3]
        
        finally:
            shared.detach()


class TestPerformance:
    def setup_method(self):
        self.manager = create_manager()
    
    def teardown_method(self):
        self.manager.close()
    
    def test_large_tensor_performance(self):
        """Test performance with large tensors"""
        large_tensor = torch.randn(1000, 1000)  # ~4MB tensor
        
        start_time = time.perf_counter()
        
        with self.manager.share_tensor(large_tensor) as shared:
            for _ in range(10):
                retrieved = shared.get()
                assert retrieved.shape == (1000, 1000)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        assert duration < 5.0, f"Large tensor operations took too long: {duration}s"
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        tensors = []
        shared_tensors = []
        
        try:
            for i in range(10):
                tensor = torch.randn(100, 100)
                tensors.append(tensor)
                shared = self.manager.share_tensor_persistent(tensor)
                shared_tensors.append(shared)
            
            stats = self.manager.get_memory_usage_summary()
            assert 'memory_usage' in stats
            
            for original, shared in zip(tensors, shared_tensors):
                retrieved = shared.get()
                assert torch.allclose(original, retrieved)
        
        finally:
            for shared in shared_tensors:
                shared.detach()


class TestErrorHandling:
    def setup_method(self):
        self.manager = create_manager()
    
    def teardown_method(self):
        self.manager.close()
    
    def test_invalid_tensor_operations(self):
        """Test error handling for invalid operations"""
        tensor = torch.randn(5, 5)
        
        with self.manager.share_tensor(tensor) as shared:
            with pytest.raises(TypeError):
                shared.set("not a tensor")
    
    def test_closed_manager_operations(self):
        """Test operations on closed manager"""
        tensor = torch.randn(3, 3)
        
        self.manager.close()
        
        with pytest.raises(Exception):
            self.manager.share_tensor_persistent(tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])