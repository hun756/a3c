from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Set, Tuple, TypedDict

from .aliases import DeviceID, BinaryHash, SourceHash


@dataclass(frozen=True, slots=True)
class DeviceTopology:
    device_id: DeviceID
    numa_node: int
    pci_domain: int
    pci_bus: int
    pci_device: int
    pci_function: int
    interconnect_bandwidth: Dict[DeviceID, float]
    memory_bandwidth: float
    compute_units: int
    peer_access_matrix: Dict[DeviceID, bool]
    nvlink_connections: Set[DeviceID]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device_id: DeviceID
    name: str
    compute_capability: Tuple[int, int]
    total_memory: int
    free_memory: int
    multiprocessor_count: int
    max_threads_per_block: int
    max_block_dimensions: Tuple[int, int, int]
    max_grid_dimensions: Tuple[int, int, int]
    warp_size: int
    memory_clock_rate: int
    memory_bus_width: int
    l1_cache_size: int
    l2_cache_size: int
    shared_memory_per_block: int
    registers_per_block: int
    clock_rate: int
    texture_alignment: int
    surface_alignment: int
    device_overlap: bool
    concurrent_kernels: bool
    concurrent_managed_access: bool
    ecc_enabled: bool
    tcc_driver: bool
    unified_addressing: bool
    stream_priorities_supported: bool
    global_l1_cache_supported: bool
    local_l1_cache_supported: bool
    max_shared_memory_per_multiprocessor: int
    max_registers_per_multiprocessor: int
    topology: DeviceTopology
    
    @cached_property
    def memory_bandwidth_gbps(self) -> float:
        return (self.memory_clock_rate * 2 * self.memory_bus_width) / (8 * 1e9)
    
    @cached_property
    def theoretical_peak_gflops(self) -> float:
        return (self.multiprocessor_count * self.clock_rate * 2) / 1e6
    
    @cached_property
    def compute_capability_score(self) -> float:
        major, minor = self.compute_capability
        return major * 10 + minor


@dataclass(frozen=True, slots=True)
class MemoryInfo:
    total: int
    free: int
    used: int
    cached: int
    reserved: int
    fragmentation_bytes: int
    largest_free_block: int
    allocation_count: int
    deallocation_count: int
    peak_usage: int
    
    @cached_property
    def usage_ratio(self) -> float:
        return self.used / self.total if self.total > 0 else 0.0
    
    @cached_property
    def fragmentation_ratio(self) -> float:
        return self.fragmentation_bytes / self.total if self.total > 0 else 0.0
    
    @cached_property
    def efficiency_score(self) -> float:
        return (1 - self.fragmentation_ratio) * (1 - self.usage_ratio)


@dataclass(frozen=True, slots=True)
class KernelInfo:
    name: str
    source_hash: SourceHash
    binary_hash: BinaryHash
    binary_size: int
    compile_time_ns: int
    registers_per_thread: int
    shared_memory_per_block: int
    const_memory_used: int
    local_memory_per_thread: int
    max_threads_per_block: int
    occupancy_calculator: Callable[[int, int], float]
    ptx_version: Tuple[int, int]
    sass_instructions: int
    instruction_cache_config: str
    shared_memory_config: str
    launch_bounds: Tuple[int, int]
    
    @cached_property
    def theoretical_occupancy(self) -> float:
        return self.occupancy_calculator(self.max_threads_per_block, self.shared_memory_per_block)
    
    @cached_property
    def memory_efficiency(self) -> float:
        return min(1.0, (48 * 1024) / max(1, self.shared_memory_per_block))


class KernelMetrics(TypedDict, total=False):
    compile_time_ns: int
    execution_time_ns: int
    memory_usage_bytes: int
    occupancy_ratio: float
    cache_hits: int
    cache_misses: int
    registers_used: int
    shared_memory_used: int
    global_memory_transactions: int
    shared_memory_transactions: int
    branch_efficiency: float
    warp_execution_efficiency: float
    instruction_throughput: float
    memory_throughput: float
    achieved_bandwidth: float
    theoretical_bandwidth: float
    arithmetic_intensity: float
    flop_count: int
    memory_access_count: int
    divergent_branches: int
    serialized_access_count: int


class ExecutionMetrics(TypedDict, total=False):
    grid_dimensions: Tuple[int, int, int]
    block_dimensions: Tuple[int, int, int]
    dynamic_shared_memory: int
    stream_id: int
    execution_time_ns: int
    kernel_launch_overhead_ns: int
    memory_transfer_time_ns: int
    synchronization_overhead_ns: int
    queue_wait_time_ns: int
    context_switch_time_ns: int
    occupancy_achieved: float
    sm_utilization: float
    memory_utilization: float
    power_consumption_watts: float
    temperature_celsius: float


class ProfileMetrics(TypedDict, total=False):
    cpu_time_ns: int
    gpu_time_ns: int
    memory_bandwidth_utilization: float
    compute_utilization: float
    tensor_core_utilization: float
    pcie_bandwidth_utilization: float
    nvlink_bandwidth_utilization: float
    energy_consumption_joules: float
    carbon_footprint_grams: float