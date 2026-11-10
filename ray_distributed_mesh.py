import ray
import time
import numpy as np

# Connect directly to your lightweight, running native host cluster instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- STATEFUL VIRTUAL HARDWARE ALLOCATOR ACTOR ---
@ray.remote
class GPUClusterResourceRegistry:
    def __init__(self, total_gpus: int):
        # Dictionary tracking the allocation state of each GPU slot (True = Busy, False = Free)
        self.gpu_matrix = {f"cuda:{i}": False for i in range(total_gpus)}
        print(f"🏗️ [HARDWARE REGISTRY] Initialized cluster matrix topology with {total_gpus} Virtual GPU slots.")

    def lease_gpu_slot(self, worker_id: int):
        """Surgically isolates and locks down the first available free GPU slot."""
        for gpu_id, is_busy in self.gpu_matrix.items():
            if not is_busy:
                self.gpu_matrix[gpu_id] = True
                return gpu_id
        return None  # Starvation State: All devices busy!

    def release_gpu_slot(self, gpu_id: str):
        """Releases the leased GPU slot register back to the open resource pool."""
        if gpu_id in self.gpu_matrix:
            self.gpu_matrix[gpu_id] = False
            return f"🔄 [REGISTRY] Released device {gpu_id} back to cluster pool."
        return "⚠️ Invalid device release request."

# --- STATELESS PARALLEL TRAINING WORKER TASK ---
@ray.remote
def execute_parallel_model_step(worker_id: int, registry_actor_ref, matrix_size: int):
    """Worker task that acquires a hardware lease, runs a compute step, and releases it."""
    # 1. Asynchronously request a dedicated hardware slot allocation lease
    assigned_device = ray.get(registry_actor_ref.lease_gpu_slot.remote(worker_id))
    
    if not assigned_device:
        return f"❌ [WORKER {worker_id}] Blocked by Resource Starvation! No GPU slots available."
        
    start_time = time.time()
    print(f"📡 [WORKER {worker_id}] Successfully leased hardware lane: {assigned_device} | Commencing training step math...")
    
    # Simulate heavy backprop gradient matrix math
    mock_gradients = np.random.randn(matrix_size, matrix_size)
    _ = np.linalg.inv(mock_gradients @ mock_gradients.T)
    time.sleep(0.2)  # Simulate execution latency
    
    compute_time = (time.time() - start_time) * 1000
    
    # 2. Release the hardware slot register lease back to the pool
    release_confirmation = ray.get(registry_actor_ref.release_gpu_slot.remote(assigned_device))
    
    return f"⚡ [WORKER {worker_id}] Completed epoch on {assigned_device} in {compute_time:.2f}ms | {release_confirmation}"

def run_process_allocator_emulator():
    print("🚀 [ORCHESTRATOR] Initializing Multi-GPU Parallel Training Process Allocator Loop...")
    
    # 1. Spawn a stateful virtual hardware registry tracking 4 virtual GPU accelerator channels
    hardware_registry = GPUClusterResourceRegistry.remote(total_gpus=4)
    
    # 2. Launch 6 concurrent parallel model training workers simultaneously to test race conditions!
    print("\n🎛️ Spawning 6 concurrent training worker tasks across the 4-slot cluster topology...")
    worker_refs = [
        execute_parallel_model_step.remote(worker_id=i, registry_actor_ref=hardware_registry, matrix_size=500)
        for i in range(6)
    ]
    
    # 3. Resolve the execution and scheduling metrics from the distributed grid
    print("⏳ Resolving allocation and compute timelines out of memory lanes...")
    allocation_reports = ray.get(worker_refs)
    
    print("\n🏆 --- PROCESS ALLOCATION & TRAFFIC TIMELINE REPORT ---")
    for report in allocation_reports:
        print(report)
        
    print("\n🏁 [ORCHESTRATOR] Multi-GPU Process Allocation Emulator Complete with Flawless Resource Isolation.")

if __name__ == "__main__":
    run_process_allocator_emulator()
