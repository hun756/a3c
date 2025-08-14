"""
Command-line interface for TensorOptim library.

This module provides CLI commands for benchmarking and profiling
tensor operations.
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List

import torch

from .core.manager import (
    create_ultra_manager,
    create_high_throughput_manager,
    create_memory_efficient_manager
)
from .types.enums import MemoryBackendType


def benchmark_command():
    """CLI command for benchmarking tensor operations."""
    parser = argparse.ArgumentParser(description='Benchmark TensorOptim performance')
    parser.add_argument('--backend', choices=['hugepages', 'posix_shm', 'numa_aware'], 
                       default='hugepages', help='Memory backend to use')
    parser.add_argument('--tensor-size', type=int, nargs='+', default=[1000, 1000],
                       help='Tensor dimensions')
    parser.add_argument('--num-tensors', type=int, default=100,
                       help='Number of tensors to benchmark')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of benchmark iterations')
    parser.add_argument('--compression', action='store_true',
                       help='Enable compression')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    backend_map = {
        'hugepages': MemoryBackendType.HUGEPAGES,
        'posix_shm': MemoryBackendType.POSIX_SHM,
        'numa_aware': MemoryBackendType.NUMA_AWARE
    }
    
    backend = backend_map[args.backend]
    
    manager = create_ultra_manager(
        backend=backend,
        compression=args.compression,
        numa_aware=True
    )
    
    try:
        results = run_benchmark(
            manager, 
            args.tensor_size, 
            args.num_tensors, 
            args.iterations
        )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(json.dumps(results, indent=2))
            
    finally:
        manager.close()


def profile_command():
    """CLI command for profiling tensor operations."""
    parser = argparse.ArgumentParser(description='Profile TensorOptim operations')
    parser.add_argument('--duration', type=int, default=60,
                       help='Profiling duration in seconds')
    parser.add_argument('--preset', choices=['default', 'high_throughput', 'memory_efficient'],
                       default='default', help='Manager preset to use')
    parser.add_argument('--output', type=str, help='Output file for profile data')
    
    args = parser.parse_args()
    
    if args.preset == 'high_throughput':
        manager = create_high_throughput_manager()
    elif args.preset == 'memory_efficient':
        manager = create_memory_efficient_manager()
    else:
        manager = create_ultra_manager()
    
    try:
        profile_data = run_profiling(manager, args.duration)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(profile_data, f, indent=2)
        else:
            print(json.dumps(profile_data, indent=2))
            
    finally:
        manager.close()


def run_benchmark(
    manager, 
    tensor_size: List[int], 
    num_tensors: int, 
    iterations: int
) -> Dict[str, Any]:
    """Run benchmark tests."""
    results = {
        'config': {
            'tensor_size': tensor_size,
            'num_tensors': num_tensors,
            'iterations': iterations,
            'backend': manager.backend_type.name,
            'compression': manager.compression_enabled
        },
        'results': {}
    }
    
    test_tensors = [
        torch.randn(*tensor_size, dtype=torch.float32)
        for _ in range(num_tensors)
    ]
    
    print(f"Running benchmark with {num_tensors} tensors of size {tensor_size}")
    print(f"Backend: {manager.backend_type.name}, Compression: {manager.compression_enabled}")
    
    share_times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        shared_tensors = []
        for tensor in test_tensors:
            shared = manager.share_tensor_persistent(tensor)
            shared_tensors.append(shared)
        
        share_time = time.perf_counter() - start_time
        share_times.append(share_time)
        
        for shared in shared_tensors:
            shared.detach_optimized()
        
        print(f"Iteration {i+1}/{iterations}: Share time = {share_time:.4f}s")
    
    shared_tensors = []
    tensor_ids = []
    for tensor in test_tensors:
        shared = manager.share_tensor_persistent(tensor)
        shared_tensors.append(shared)
        tensor_ids.append(shared.tensor_id)
    
    get_times = []
    for i in range(iterations):
        start_time = time.perf_counter()
        
        for tensor_id in tensor_ids:
            retrieved = manager.get_tensor_ultra(tensor_id)
            if retrieved:
                _ = retrieved.get_ultra_fast()
        
        get_time = time.perf_counter() - start_time
        get_times.append(get_time)
        
        print(f"Iteration {i+1}/{iterations}: Get time = {get_time:.4f}s")
    
    for shared in shared_tensors:
        shared.detach_optimized()
    
    results['results'] = {
        'share_times': {
            'mean': sum(share_times) / len(share_times),
            'min': min(share_times),
            'max': max(share_times),
            'all': share_times
        },
        'get_times': {
            'mean': sum(get_times) / len(get_times),
            'min': min(get_times),
            'max': max(get_times),
            'all': get_times
        },
        'throughput': {
            'tensors_per_second_share': num_tensors / (sum(share_times) / len(share_times)),
            'tensors_per_second_get': num_tensors / (sum(get_times) / len(get_times))
        }
    }
    
    return results


def run_profiling(manager, duration: int) -> Dict[str, Any]:
    """Run profiling for specified duration."""
    print(f"Profiling for {duration} seconds...")
    
    start_time = time.time()
    end_time = start_time + duration
    
    test_tensors = [
        torch.randn(100, 100, dtype=torch.float32),
        torch.randn(500, 500, dtype=torch.float32),
        torch.randn(1000, 1000, dtype=torch.float32)
    ]
    
    shared_tensors = []
    operation_count = 0
    
    try:
        while time.time() < end_time:
            for tensor in test_tensors:
                shared = manager.share_tensor_persistent(tensor)
                shared_tensors.append(shared)
                operation_count += 1
            
            for shared in shared_tensors[-len(test_tensors):]:
                _ = shared.get_ultra_fast()
                operation_count += 1
            
            if len(shared_tensors) > 100:
                for shared in shared_tensors[:50]:
                    shared.detach_optimized()
                shared_tensors = shared_tensors[50:]
            
            time.sleep(0.01)  # Small delay
    
    finally:
        for shared in shared_tensors:
            shared.detach_optimized()
    
    metrics = manager.get_ultra_performance_metrics()
    
    profile_data = {
        'duration': duration,
        'operations_performed': operation_count,
        'operations_per_second': operation_count / duration,
        'metrics': metrics,
        'memory_summary': manager.get_memory_usage_summary()
    }
    
    print(f"Completed {operation_count} operations ({operation_count/duration:.2f} ops/sec)")
    
    return profile_data


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m tensoroptim.cli <command>")
        print("Commands: benchmark, profile")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'benchmark':
        benchmark_command()
    elif command == 'profile':
        profile_command()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)