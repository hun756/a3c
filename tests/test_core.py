import pytest
import torch
import time
from unittest.mock import Mock, patch

from tensoroptim.types import TensorID, TensorDescriptor, CompressionType
from tensoroptim.core import TensorReference, SharedTensor, TensorRegistry
from tensoroptim.exceptions import TensorError


class TestTensorDescriptor:
    def test_descriptor_creation(self):
        tensor_id = TensorID("test_tensor")
        shape = (10, 20)
        dtype = torch.float32
        device = torch.device("cpu")
        
        descriptor = TensorDescriptor(
            tensor_id=tensor_id,
            shape=shape,
            dtype=dtype,
            device=device
        )
        
        assert descriptor.tensor_id == tensor_id
        assert descriptor.shape == shape
        assert descriptor.dtype == dtype
        assert descriptor.device == device
        assert descriptor.numel == 200
        assert descriptor.element_size == 4
        assert descriptor.raw_byte_size == 800
    
    def test_descriptor_validation(self):
        with pytest.raises(ValueError, match="Invalid tensor shape"):
            TensorDescriptor(
                tensor_id=TensorID("test"),
                shape=(0, 10),
                dtype=torch.float32,
                device=torch.device("cpu")
            )
        
        with pytest.raises(ValueError, match="Alignment must be a positive power of 2"):
            TensorDescriptor(
                tensor_id=TensorID("test"),
                shape=(10, 10),
                dtype=torch.float32,
                device=torch.device("cpu"),
                alignment=63
            )
    
    def test_descriptor_properties(self):
        descriptor = TensorDescriptor(
            tensor_id=TensorID("test"),
            shape=(5, 5),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        assert descriptor.is_cpu is True
        assert descriptor.is_cuda is False
        assert descriptor.aligned_byte_size >= descriptor.raw_byte_size
    
    def test_descriptor_with_checksum(self):
        descriptor = TensorDescriptor(
            tensor_id=TensorID("test"),
            shape=(5, 5),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        new_descriptor = descriptor.with_checksum(12345)
        assert new_descriptor.checksum == 12345
        assert new_descriptor.tensor_id == descriptor.tensor_id
        assert new_descriptor.last_accessed > descriptor.last_accessed


class TestTensorRegistry:
    def setup_method(self):
        self.registry = TensorRegistry()
        self.mock_reference = Mock()
        self.mock_reference.descriptor = TensorDescriptor(
            tensor_id=TensorID("test_tensor"),
            shape=(10, 10),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
    
    def test_register_tensor(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        
        assert tensor_id in self.registry
        assert len(self.registry) == 1
        assert self.registry.get_tensor_count() == 1
    
    def test_register_duplicate_tensor(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register_tensor(tensor_id, self.mock_reference)
    
    def test_unregister_tensor(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        
        result = self.registry.unregister_tensor(tensor_id)
        assert result is True
        assert tensor_id not in self.registry
        assert len(self.registry) == 0
    
    def test_unregister_nonexistent_tensor(self):
        tensor_id = TensorID("nonexistent")
        result = self.registry.unregister_tensor(tensor_id)
        assert result is False
    
    def test_get_tensor(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        
        retrieved = self.registry.get_tensor(tensor_id)
        assert retrieved == self.mock_reference
    
    def test_find_tensors_by_criteria(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        
        results = self.registry.find_tensors_by_criteria(
            shape=(10, 10),
            dtype="torch.float32"
        )
        assert len(results) == 1
        assert results[0] == self.mock_reference
    
    def test_registry_stats(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        self.registry.get_tensor(tensor_id)  # Access to increment stats
        
        stats = self.registry.get_registry_stats()
        assert stats['total_tensors'] == 1
        assert stats['access_stats']['total_accesses'] == 1
    
    def test_cleanup_expired_tensors(self):
        tensor_id = TensorID("test_tensor")
        self.registry.register_tensor(tensor_id, self.mock_reference)
        
        # Simulate old access time
        self.registry._last_access_times[tensor_id] = time.perf_counter() - 3700  # 1+ hour ago
        
        cleaned = self.registry.cleanup_expired_tensors(max_age_seconds=3600)  # 1 hour
        assert cleaned == 1
        assert len(self.registry) == 0


class TestSharedTensor:
    def setup_method(self):
        self.mock_reference = Mock()
        self.mock_reference.descriptor = TensorDescriptor(
            tensor_id=TensorID("test_tensor"),
            shape=(5, 5),
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        self.mock_reference.is_valid = True
        self.mock_reference.access_count = 0
        self.mock_reference.last_access = time.perf_counter()
        
        self.shared_tensor = SharedTensor(self.mock_reference)
    
    def test_shared_tensor_properties(self):
        assert self.shared_tensor.tensor_id == TensorID("test_tensor")
        assert self.shared_tensor.shape == (5, 5)
        assert self.shared_tensor.dtype == torch.float32
        assert self.shared_tensor.device == torch.device("cpu")
        assert self.shared_tensor.is_valid is True
    
    def test_shared_tensor_get(self):
        mock_tensor = torch.zeros(5, 5)
        self.mock_reference.materialize.return_value = mock_tensor
        
        result = self.shared_tensor.get()
        assert torch.equal(result, mock_tensor)
        self.mock_reference.materialize.assert_called_once()
    
    def test_shared_tensor_set(self):
        test_tensor = torch.ones(5, 5)
        
        self.shared_tensor.set(test_tensor)
        self.mock_reference.persist.assert_called_once_with(test_tensor)
    
    def test_shared_tensor_set_invalid_type(self):
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            self.shared_tensor.set("not a tensor")
    
    def test_shared_tensor_context_manager(self):
        with self.shared_tensor as st:
            assert st == self.shared_tensor
        
        self.mock_reference.detach.assert_called_once()
    
    def test_shared_tensor_access_metrics(self):
        metrics = self.shared_tensor.access_metrics
        assert 'access_count' in metrics
        assert 'last_access' in metrics
        assert 'performance' in metrics
    
    def test_shared_tensor_repr(self):
        repr_str = repr(self.shared_tensor)
        assert "SharedTensor" in repr_str
        assert "test_tensor" in repr_str
        assert "(5, 5)" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])