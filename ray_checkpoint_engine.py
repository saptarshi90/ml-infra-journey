import ray
import time
import torch
import os

# Initialize Ray if not already active
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

os.makedirs("checkpoints", exist_ok=True)

# Stateful Actor for out-of-band weight persistence
@ray.remote
class AsyncCheckpointVault:
    def __init__(self):
        self.checkpoint_counter = 0

    def persist_weights_out_of_band(self, epoch: int, weights_dict: dict):
        start_io = time.time()
        self.checkpoint_counter += 1
        time.sleep(0.3)  # Simulate I/O write delay
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pt"
        torch.save(weights_dict, checkpoint_path)
        io_latency = (time.time() - start_io) * 1000
        print(f"💾 [VAULT BACKGROUND THREAD] Hard disk serialization complete for Epoch {epoch} -> {checkpoint_path} | Disk I/O Write Time: {io_latency:.2f}ms")
        return f"SUCCESS_EPOCH_{epoch}"

# Parallel worker simulating non-blocking weight handoff
@ray.remote
def execute_training_epoch_with_async_io(worker_id: int, epoch: int, vault_actor_ref):
    time.sleep(0.2)
    mock_weights = {"weights": torch.randn(5, 5), "bias": torch.randn(5)}
    # Fire-and-forget call to the vault actor
    vault_actor_ref.persist_weights_out_of_band.remote(epoch, mock_weights)
    return f"🚀 [WORKER {worker_id}] Released to start Epoch {epoch + 1} instantly!"

def run_asynchronous_checkpoint_benchmark():
    storage_vault = AsyncCheckpointVault.remote()
    start_pipeline = time.time()
    worker_timeline_refs = [
        execute_training_epoch_with_async_io.remote(worker_id=0, epoch=epoch, vault_actor_ref=storage_vault)
        for epoch in range(3)
    ]
    worker_logs = ray.get(worker_timeline_refs)
    for log in worker_logs:
        print(log)
    time.sleep(1.2)
    total_time = (time.time() - start_pipeline) * 1000
    print(f"\n📊 Total Ingestion and Persistence Runtime Cycle: {total_time:.2f}ms")

if __name__ == "__main__":
    run_asynchronous_checkpoint_benchmark()
