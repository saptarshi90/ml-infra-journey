import ray
import time
import numpy as np

# Connect directly to your lightweight, running native host cluster instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- STATEFUL GLOBAL CHECKPOINT VAULT ACTOR ---
@ray.remote
class GlobalCheckpointVault:
    def __init__(self):
        self.vault = {}
        print("🏗️ [CHECKPOINT VAULT] Resilient distributed state vault successfully initialized.")

    def save_checkpoint(self, worker_id: int, epoch: int, weights_metric: float):
        self.vault[worker_id] = {"epoch": epoch, "metric": weights_metric}
        return f"💾 [VAULT] Stored pristine Epoch {epoch} state for Worker {worker_id} | Metric: {weights_metric:.4f}"

    def fetch_latest_checkpoint(self, worker_id: int):
        return self.vault.get(worker_id, {"epoch": -1, "metric": 0.0})

# --- RESILIENT TRAINING LOOP WITH FAULT TOLERANCE ---
@ray.remote(max_retries=2)
def execute_resilient_training_loop(worker_id: int, vault_actor_ref):
    print(f"\n🚀 [WORKER {worker_id}] Launching parallel training pipeline thread...")
    
    checkpoint = ray.get(vault_actor_ref.fetch_latest_checkpoint.remote(worker_id))
    start_epoch = checkpoint["epoch"] + 1
    
    if start_epoch > 0:
        print(f"🔄 [WORKER {worker_id} RECOVERY] Detected crash footprint! Resuming from saved Epoch {checkpoint['epoch']} matrix state.")
        current_metric = checkpoint["metric"]
    else:
        print(f"🌱 [WORKER {worker_id}] Starting fresh training loop from scratch (Epoch 0).")
        current_metric = 1.0

    for epoch in range(start_epoch, 3):
        if worker_id == 1 and epoch == 1 and start_epoch == 0:
            print(f"💥 [WORKER 1] !!! CRITICAL HARDWARE FAULT INJECTED !!! Simulating preemption / core crash at Epoch {epoch}...")
            time.sleep(0.1)
            raise RuntimeError("Catastrophic hardware node preemption event detected.")

        time.sleep(0.2)
        current_metric *= 0.95
        
        save_log = ray.get(vault_actor_ref.save_checkpoint.remote(worker_id, epoch, current_metric))
        print(f"🔹 [WORKER {worker_id}] Finished Epoch {epoch} | {save_log}")
        
    return f"🏁 [WORKER {worker_id}] Completed training cycle. Final optimized matrix weight: {current_metric:.4f}"

def run_fault_tolerance_benchmark():
    print("🚀 [ORCHESTRATOR] Initializing Asynchronous Worker Preemption Recovery Engine...")
    vault = GlobalCheckpointVault.remote()
    
    print("\n📡 Deploying 2 parallel training threads into the cluster grid topology...")
    worker_refs = [
        execute_resilient_training_loop.remote(worker_id=i, vault_actor_ref=vault)
        for i in range(2)
    ]
    
    print("⏳ Resolving execution tracking matrices from the distributed mesh...")
    try:
        results = ray.get(worker_refs)
        print("\n🏆 --- CLUSTER TRAINING EXECUTION REPORT ---")
        for res in results:
            print(res)
    except Exception as e:
        print(f"⚠️ Orchestrator caught top-level loop crash: {e}")

    print("\n🏁 [ORCHESTRATOR] Fault-Tolerance and Preemption Recovery Benchmarks Finalized Successfully.")

if __name__ == "__main__":
    run_fault_tolerance_benchmark()
