import ray
import time
import numpy as np

# Connect directly to your lightweight running native host instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- STATEFUL GPU RUNTIME NODE ACTOR ---
@ray.remote
class DistGPUComputeNode:
    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        # Generate initial mock local gradients (split into equal array shards)
        # For simplicity, Node i starts with unique values to trace the math
        self.gradient_shards = [
            np.array([float(node_id * 10 + shard_idx)], dtype=np.float64)
            for shard_idx in range(total_nodes)
        ]
        print(f"🏗️ [GPU NODE {self.node_id}] Initialized with gradient shards: {[s.tolist() for s in self.gradient_shards]}")

    def get_shard_value(self, shard_idx: int):
        """Allows neighboring nodes to read a specific shard register block."""
        return self.gradient_shards[shard_idx]

    def accumulate_incoming_shard(self, shard_idx: int, incoming_data: np.ndarray):
        """Performs a mathematical reduction step (addition) on a shard."""
        self.gradient_shards[shard_idx] += incoming_data
        return f"📥 [GPU NODE {self.node_id}] Reduced Shard {shard_idx} | New Val: {self.gradient_shards[shard_idx].tolist()}"

    def overwrite_shard_value(self, shard_idx: int, broadcast_data: np.ndarray):
        """Overwrites a local shard block with globally aggregated weights."""
        self.gradient_shards[shard_idx] = broadcast_data.copy()
        return f"🔄 [GPU NODE {self.node_id}] Broadcast Sync Shard {shard_idx} to Global Sum."

    def retrieve_final_gradients(self):
        """Returns the fully compiled, unified gradient sheets."""
        return [s.tolist()[0] for s in self.gradient_shards]

def execute_ring_allreduce_simulation():
    print("🚀 [ORCHESTRATOR] Initializing Multi-Node Ring AllReduce Collective Mesh...")
    
    # 1. Spawn a ring of 3 stateful GPU actor compute nodes
    num_nodes = 3
    nodes = [DistGPUComputeNode.remote(node_id=i, total_nodes=num_nodes) for i in range(num_nodes)]
    
    # 2. RUN STAGE 1: SCATTER-REDUCE BUCKET BRIGADE
    print("\n🌀 --- STAGE 1: COMMENCING SCATTER-REDUCE RING CYCLE ---")
    # In a ring of size N, it takes exactly N-1 rotation cycles to globally sum a shard slice
    for step in range(num_nodes - 1):
        print(f"\n📍 [SCATTER-REDUCE STEP {step + 1}] Rotating gradient shards around the mesh ring...")
        step_ops = []
        for i in range(num_nodes):
            # Target right-hand neighbor: (i + 1) modulo total nodes
            right_neighbor = nodes[(i + 1) % num_nodes]
            
            # Determine which shard slice is rotating down this step lane
            shard_to_send_idx = (i - step) % num_nodes
            
            # Extract data from current node and stream/accumulate onto the right neighbor
            data_ref = nodes[i].get_shard_value.remote(shard_to_send_idx)
            op_ref = right_neighbor.accumulate_incoming_shard.remote(shard_to_send_idx, data_ref)
            step_ops.append(op_ref)
            
        # Complete network barrier synchronization for this rotation step
        logs = ray.get(step_ops)
        for log in logs:
            print(log)

    # 3. RUN STAGE 2: ALLGATHER BROADCAST SYNC CYCLE
    print("\n📢 --- STAGE 2: COMMENCING ALLGATHER BROADCAST SYNC CYCLE ---")
    # In a ring of size N, it takes exactly N-1 more rotations to broadcast the summed values everywhere
    for step in range(num_nodes - 1):
        print(f"\n📍 [ALLGATHER STEP {step + 1}] Syncing globally aggregated weight matrices...")
        step_ops = []
        for i in range(num_nodes):
            right_neighbor = nodes[(i + 1) % num_nodes]
            
            # Identify which globally summed shard slice is ready for broadcasting
            shard_to_broadcast_idx = (i + 1 - step) % num_nodes
            
            # Read compiled values from sender and overwrite onto receiver memory banks
            data_ref = nodes[i].get_shard_value.remote(shard_to_broadcast_idx)
            op_ref = right_neighbor.overwrite_shard_value.remote(shard_to_broadcast_idx, data_ref)
            step_ops.append(op_ref)
            
        # Complete network barrier synchronization for this broadcast step
        logs = ray.get(step_ops)
        for log in logs:
            print(log)

    # 4. FINAL VERIFICATION CHECKS
    print("\n⏳ Fetching post-reduction ground-truth telemetry vectors out of node registers...")
    final_reports = ray.get([node.retrieve_final_gradients.remote() for node in nodes])
    
    print("\n🏆 --- RING ALLREDUCE CHANNELS VERIFICATION REPORT ---")
    for idx, report in enumerate(final_reports):
        print(f"📦 [GPU NODE {idx} Final State] Unified Gradient Array: {report}")
        
    print("\n🏁 [ORCHESTRATOR] Ring AllReduce Collective Simulator Completed with Absolute Bandwidth Saturation.")

if __name__ == "__main__":
    execute_ring_allreduce_simulation()
