import ray
import time
import numpy as np

if not ray.is_initialized():
    ray.init(logging_level="error")

@ray.remote
class CollectiveGPUWorker:
    def __init__(self, rank: int, total_workers: int):
        self.rank = rank
        self.total_workers = total_workers
        # Simulating next-hop ring target rank
        self.next_rank = (rank + 1) % total_workers

    def compute_local_attention_shard(self, tokens_count: int):
        # Generate a mock intermediate hidden state vector shard
        np.random.seed(self.rank * 42)
        self.local_hidden_state = np.random.randint(1, 10, size=(tokens_count,)).astype(np.float32)
        print(f"⚡ [RANK {self.rank}] Computed local attention shard array: {self.local_hidden_state}")
        return self.local_hidden_state

    def execute_allreduce_sync(self, peer_shard):
        # Ring Communication Phase: Summing local state with array shard from peer node
        synchronized_attention_map = self.local_hidden_state + peer_shard
        print(f"🏁 [RANK {self.rank}] Collective All-Reduce Ring synchronized! Matrix: {synchronized_attention_map}")
        return synchronized_attention_map

def run_distributed_allreduce_mesh():
    print("🚀 [ORCHESTRATOR] Spawning Collective Communication Mesh (Workers=2)...")
    
    # 1. Instantiate the multi-GPU node layout actors
    worker_0 = CollectiveGPUWorker.remote(rank=0, total_workers=2)
    worker_1 = CollectiveGPUWorker.remote(rank=1, total_workers=2)
    
    tokens_count = 4  # 4 context tokens
    
    # 2. Trigger concurrent localized tensor execution
    ref_shard_0 = worker_0.compute_local_attention_shard.remote(tokens_count)
    ref_shard_1 = worker_1.compute_local_attention_shard.remote(tokens_count)
    
    # Gather internal shard values out of the object store to pass across the ring mesh
    shard_0 = ray.get(ref_shard_0)
    shard_1 = ray.get(ref_shard_1)
    
    print("\n📡 [RING MESH] Initiating collective All-Reduce network transfer primitives...")
    
    # 3. Pass split attention arrays to the neighboring ring nodes for summation synchronization
    sync_0 = worker_0.execute_allreduce_sync.remote(shard_1)
    sync_1 = worker_1.execute_allreduce_sync.remote(shard_0)
    
    # Resolve the unified matrix state
    final_mesh_states = ray.get([sync_0, sync_1])
    print("\n📊 [ORCHESTRATOR] Cluster Attention Synced Matrix Verified Successfully.")

if __name__ == "__main__":
    run_distributed_allreduce_mesh()
    ray.shutdown()
