import ray
import time
import numpy as np

# Connect to the running native host cluster instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- STATEFUL ROUTING NODE PRIMITIVE ---
@ray.remote
class DataShuffleRouterNode:
    def __init__(self, node_id: int):
        self.node_id = node_id
        # In-memory buffer register to hold routed arrays from separate workers
        self.memory_buffer = []

    def receive_data_packet(self, worker_id: int, data_subset: np.ndarray):
        """Asynchronously receives and stores a shuffled data slice in RAM."""
        self.memory_buffer.append(data_subset)
        return f"📡 [ROUTER NODE {self.node_id}] Cached packet from Worker {worker_id} | Total Buffered Buckets: {len(self.memory_buffer)}"

    def finalize_and_aggregate_shard(self):
        """Collapses all buffered memory arrays into a single unified training vector."""
        if not self.memory_buffer:
            return {"node": self.node_id, "total_samples": 0, "mean_weight": 0.0}
        # Merge all zero-copy slices held in memory
        unified_shard = np.concatenate(self.memory_buffer)
        return {
            "node": self.node_id,
            "total_samples": len(unified_shard),
            "mean_weight": float(np.mean(unified_shard)) if len(unified_shard) > 0 else 0.0
        }

def run_distributed_shuffle_pipeline():
    print("🚀 [ORCHESTRATOR] Initializing Distributed Actor-Driven Data Shuffle Engine...")
    
    # 1. Spawn a matrix of 2 distinct stateful router node processes in memory
    print("🏗️ Provisioning stateful distributed memory router nodes...")
    routers = [DataShuffleRouterNode.remote(node_id=i) for i in range(2)]
    
    # 2. Simulate 3 concurrent parallel worker operations generating intermediate vectors
    print("📊 Simulating 3 concurrent data workers generating local features...")
    worker_data_1 = np.random.randn(1000) + 1.0  # Mean shifted around 1.0
    worker_data_2 = np.random.randn(1000) + 2.0  # Mean shifted around 2.0
    worker_data_3 = np.random.randn(1000) + 3.0  # Mean shifted around 3.0
    
    # 3. CROSS-ROUTING SHUFFLE PHASE: Asynchronously route data slices to routers across memory lines
    print("🔀 Executing multi-channel data shuffle and routing lines...")
    route_ops = [
        # Worker 1 shards data: routes half to Router 0, half to Router 1
        routers[0].receive_data_packet.remote(worker_id=1, data_subset=worker_data_1[:500]),
        routers[1].receive_data_packet.remote(worker_id=1, data_subset=worker_data_1[500:]),
        
        # Worker 2 shards data
        routers[0].receive_data_packet.remote(worker_id=2, data_subset=worker_data_2[:500]),
        routers[1].receive_data_packet.remote(worker_id=2, data_subset=worker_data_2[500:]),
        
        # Worker 3 shards data
        routers[0].receive_data_packet.remote(worker_id=3, data_subset=worker_data_3[:500]),
        routers[1].receive_data_packet.remote(worker_id=3, data_subset=worker_data_3[500:])
    ]
    
    # Resolve the acknowledgement logs from the router grid
    acknowledgements = ray.get(route_ops)
    for ack in acknowledgements:
        print(ack)
        
    # 4. AGGREGATION PHASE: Command routers to collapse buffers into clean training matrices
    print("\n⏳ Triggering memory aggregation and computing global cluster metrics...")
    aggregation_refs = [router.finalize_and_aggregate_shard.remote() for router in routers]
    aggregated_reports = ray.get(aggregation_refs)
    
    print("\n🏆 --- SHUFFLE SHARD AGGREGATION REPORT ---")
    for report in aggregated_reports:
        print(f"📦 [Router Node {report['node']}] Aggregated Samples: {report['total_samples']} | Shuffled Pool Mean Weight: {report['mean_weight']:.4f}")
        
    print("\n🏁 [ORCHESTRATOR] Distributed Memory Shuffle Complete with Zero Disk I/O Bottlenecks.")

if __name__ == "__main__":
    run_distributed_shuffle_pipeline()
