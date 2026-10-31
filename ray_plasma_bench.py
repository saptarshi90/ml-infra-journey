import os
import ray
import time
import numpy as np

# Initialize the localized secure Ray core execution engine
if not ray.is_initialized():
    ray.init(logging_level="error")

@ray.remote
class HighPerformanceGPUWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id

    def process_tensor_by_value(self, matrix: np.ndarray):
        # Tracking memory footprint via unique array data memory addresses
        address = matrix.__array_interface__["data"]
        # Simulate quick mathematical calculation step
        result = np.mean(matrix)
        return {"worker": self.worker_id, "address": address, "result": float(result)}

    def process_tensor_by_reference(self, matrix):
        # Precision Fix: Ray already automatically unwrapped the ObjectRef into a NumPy array!
        address = matrix.__array_interface__["data"]
        result = np.mean(matrix)
        return {"worker": self.worker_id, "address": address, "result": float(result)}

def run_plasma_memory_benchmark():
    print("⚡ [PLASMA MATRIX] Generating heavy 4,000,000 parameter model weight layer matrix (float32)...")
    # Generate a heavy 2000x2000 float32 matrix (~16MB of raw memory blocks)
    heavy_weight_matrix = np.random.randn(2000, 2000).astype(np.float32)
    
    worker = HighPerformanceGPUWorker.remote(worker_id=0)
    
    print("\n🚨 BENCHMARK CRITICAL 1: Passing Heavy Tensors By Value (Standard Serialization Loop)")
    start_value = time.time()
    # Passing by value forces Ray to duplicate and re-serialize the entire 16MB matrix across processes
    ref_value = worker.process_tensor_by_value.remote(heavy_weight_matrix)
    res_value = ray.get(ref_value)
    duration_value = time.time() - start_value
    print(f"📥 [BY-VALUE STATUS] Completed in {duration_value:.5f}s. Worker memory address: {res_value["address"]}")
    
    print("\n🚀 BENCHMARK CRITICAL 2: Passing Heavy Tensors By Immutable Plasma Reference (Zero-Copy Core)")
    start_ref = time.time()
    # Explicitly anchor the weight matrix into the Plasma Object Store exactly ONCE!
    plasma_object_ref = ray.put(heavy_weight_matrix)
    # Pass only the tiny, lightweight reference pointer handle!
    ref_plasma = worker.process_tensor_by_reference.remote(plasma_object_ref)
    res_ref = ray.get(ref_plasma)
    duration_ref = time.time() - start_ref
    print(f"📥 [PLASMA STATUS] Completed in {duration_ref:.5f}s. Worker memory address: {res_ref["address"]}")
    
    # Calculate performance gains
    speedup = duration_value / duration_ref
    print(f"\n🏁 [BENCHMARK RESULTS] Plasma Shared Memory Architecture achieved a {speedup:.2f}x execution speedup!")
    print("--------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    run_plasma_memory_benchmark()
    ray.shutdown()
