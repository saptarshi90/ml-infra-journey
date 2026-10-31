import os
import ray
import time

# Initialize a localized Ray core compute engine inside the execution thread
if not ray.is_initialized():
    print("⚡ [RAY CORE] Initializing distributed execution runtime plane...")
    ray.init(logging_level="info")

# Define a distributed Ray Actor representing an isolated virtual GPU worker node
@ray.remote
class GPUWorkerNode:
    def __init__(self, worker_id: int, vram_blocks: int):
        self.worker_id = worker_id
        self.vram_blocks = vram_blocks
        print(f"📦 [WORKER #{self.worker_id}] Virtual GPU Worker initialized with {self.vram_blocks} PagedAttention memory slots.")

    def compute_tensor_slice(self, prompt_tokens: int, weight_shard_id: int):
        # Calculate PagedAttention block needs for this specific partition slice
        required_blocks = (prompt_tokens + 3) // 4
        time.sleep(0.1)  # Simulate localized CUDA kernel execution latency
        
        return {
            "worker_id": self.worker_id,
            "weight_shard_processed": weight_shard_id,
            "allocated_blocks": required_blocks,
            "status": "PARTITION_COMPUTE_SUCCESS"
        }

def execute_tensor_parallel_inference():
    print("\n🚀 [ORCHESTRATOR] Initializing Tensor Parallelism (TP=2) across split worker matrix...")
    
    # 1. Spin up two distinct GPU worker actors inside the distributed object store
    worker_1 = GPUWorkerNode.remote(worker_id=0, vram_blocks=16)
    worker_2 = GPUWorkerNode.remote(worker_id=1, vram_blocks=16)
    
    # Mock inference prompt length (12 words)
    prompt_tokens = 12
    
    print("📡 [SCANCE GRID] Broadcasting weight shard operations to worker nodes concurrently...")
    start_time = time.time()
    
    # 2. Fire concurrent remote method invocations (Returns Ray Object References)
    ref_worker_1 = worker_1.compute_tensor_slice.remote(prompt_tokens, weight_shard_id=101)
    ref_worker_2 = worker_2.compute_tensor_slice.remote(prompt_tokens, weight_shard_id=102)
    
    # 3. Resolve and gather the distributed object states out of the cluster memory pool
    results = ray.get([ref_worker_1, ref_worker_2])
    duration = time.time() - start_time
    
    print(f"🏁 [ORCHESTRATOR] Distributed computation resolved in {duration:.4f}s!")
    print("\n--- DISTRIBUTED TELEMETRY RESULTS ---")
    for block in results:
        print(f"🔹 Worker {block['worker_id']} processed Shard {block['weight_shard_processed']} using {block['allocated_blocks']} blocks. Status: {block['status']}")
    print("-------------------------------------\n")

if __name__ == "__main__":
    execute_tensor_parallel_inference()
    # Cleanly release Ray resources upon execution completion
    ray.shutdown()
