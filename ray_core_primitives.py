import ray
import time
import numpy as np

# Connect directly to your lightweight, running native host cluster instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- PRIMITIVE 1: STATELESS REMOTE TASK ---
@ray.remote
def process_stateless_token_shard(token_id: int, matrix_size: int):
    """Simulates an independent, stateless matrix math execution block."""
    time.sleep(0.1)
    mock_matrix = np.random.randn(matrix_size, matrix_size)
    determinant = float(np.linalg.det(mock_matrix))
    return f"✨ [TASK {token_id}] Processed shard matrix det: {determinant:.4f}"

# --- PRIMITIVE 2: STATEFUL REMOTE ACTOR ---
@ray.remote
class StatefulKVMemoryRegister:
    def __init__(self):
        # Persistent memory register dictionary held inside this specific actor process space
        self.kv_store = {}

    def update_kv_cache(self, layer_name: str, key_weights: list):
        self.kv_store[layer_name] = key_weights
        return f"💾 [ACTOR] Persistent KV State Saved for {layer_name} | Total Layers Stored: {len(self.kv_store)}"

    def retrieve_kv_cache(self, layer_name: str):
        return self.kv_store.get(layer_name, None)

def run_native_primitives_benchmark():
    print("🚀 [ORCHESTRATOR] Launching High-Velocity Ray Core Primitives Test Room...")
    
    # 1. Fire Concurrently Spawning Stateless Tasks (Returns Object References instantly!)
    print("\n📡 Launching concurrent stateless task instances across the CPU topology...")
    task_refs = [process_stateless_token_shard.remote(token_id=i, matrix_size=4) for i in range(3)]
    
    # 2. Resolve Object Store References out of memory registers
    print("⏳ Resolving stateless task object data pools out of the Ray Object Store...")
    task_results = ray.get(task_refs)
    for result in task_results:
        print(result)
        
    # 3. Instantiate a Stateful Persistent Actor Register in memory
    print("\n🏗️ Spawning stateful persistent worker memory actor...")
    kv_register = StatefulKVMemoryRegister.remote()
    
    # 4. Trigger state modifications across separate execution blocks on the same actor
    ref_update_1 = kv_register.update_kv_cache.remote("transformer.layer.0.attention", [0.1, 0.5, 0.9])
    ref_update_2 = kv_register.update_kv_cache.remote("transformer.layer.1.attention", [0.2, 0.4, 0.8])
    
    # Resolve the status tracking states
    print(ray.get(ref_update_1))
    print(ray.get(ref_update_2))
    
    # 5. Verify the internal persistence of the state register
    final_fetch_ref = kv_register.retrieve_kv_cache.remote("transformer.layer.0.attention")
    print(f"🏁 [VERIFICATION] Actor Memory Retrieval for Layer 0: {ray.get(final_fetch_ref)}")
    print("\n📊 [ORCHESTRATOR] Ray Core Primitives Executed Successfully with Zero Memory Swapping Lag.")

if __name__ == "__main__":
    run_native_primitives_benchmark()
