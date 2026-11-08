import ray
import time
import numpy as np

# Connect directly to your lightweight native cluster instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

@ray.remote
def execute_tensor_math_on_reference(data_array):
    """Worker directly reads the implicitly resolved zero-copy array out of Plasma."""
    # Optimization Fix: Remove redundant ray.get() layer! Ray automatically unwraps the reference argument!
    matrix_sum = float(np.sum(data_array))
    return f"⚡ [WORKER] Evaluated shared tensor zero-copy space. Elements sum: {matrix_sum:.2f}"

def run_memory_optimization_benchmark():
    print("🚀 [ORCHESTRATOR] Initializing Zero-Copy Object Store Optimization Sprints...")
    
    # 1. Simulate a heavy 5,000,000 element model weight tensor array
    print("📊 Generating massive mock model weight tensor array space...")
    heavy_weights = np.random.randn(5000000)
    
    # 2. SURGICAL MOVE: Explicitly pin the heavy data straight into the Plasma Object Store pool!
    print("📌 Pinning heavy weights tensor array into the shared memory registry pool...")
    start_pin = time.time()
    heavy_weights_ref = ray.put(heavy_weights)
    print(f"🆔 Generated Shared Object Reference ID: {heavy_weights_ref} | Time to pin: {(time.time() - start_pin)*1000:.2f}ms")
    
    # 3. Pass ONLY the lightweight ObjectRef ID to the worker task
    print("\n📡 Launching parallel worker by injecting the Object Reference ID...")
    worker_ref = execute_tensor_math_on_reference.remote(heavy_weights_ref)
    
    # 4. Resolve the final computation result out of memory lanes
    result = ray.get(worker_ref)
    print(result)
    
    print("\n🏁 [ORCHESTRATOR] Zero-Copy Reference Passing Complete with Absolute Safety Parameters.")

if __name__ == "__main__":
    run_memory_optimization_benchmark()
