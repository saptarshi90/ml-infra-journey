import ray
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# --- CORE DISTRIBUTED INFRASTRUCTURE MODEL ---
class AIInfrastructureModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple deep feature weight matrix layer
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

# --- THE CONCURRENT WORKER TRAINING TASK LOOP ---
def worker_training_loop(config):
    # 1. Initialize Ray Train's backend communication hooks automatically
    device = "cpu"
    
    # 2. Build mock tensor metrics shards
    X_mock = torch.randn(1000, 10)
    y_mock = torch.randn(1000, 1)
    dataset = TensorDataset(X_mock, y_mock)
    
    # 3. Optimize data loading via distributed sharding preparation
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    loader = train.torch.prepare_data_loader(loader)
    
    # 4. Instantiate and wrap model inside distributed synchronization primitives
    model = AIInfrastructureModel()
    model = train.torch.prepare_model(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Run a compact 3-epoch distributed optimization sprint
    for epoch in range(3):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Report the isolated shard metrics to the orchestrator layer
        train.report({"loss": epoch_loss / len(loader)})

# --- MASTER ORCHESTRATOR EXECUTION ENGINE ---
def run_distributed_training_pipeline():
    print("🚀 [ORCHESTRATOR] Booting Ray TorchTrainer Mesh across the parallel topology...")
    
    # Configure a 2-worker scaling layout using standard backends
    trainer = TorchTrainer(
        train_loop_per_worker=worker_training_loop,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        torch_config=train.torch.TorchConfig(backend="gloo")
    )
    
    # Run the optimization matrix
    result = trainer.fit()
    print(f"\n📊 [ORCHESTRATOR] Distributed Training Complete.")
    print(f"🏁 Final Aggregated Loss Metric Resolved: {result.metrics['loss']:.4f}")

if __name__ == "__main__":
    # Ensure loopback connection is locked
    if not ray.is_initialized():
        ray.init(address="auto", logging_level="error")
    run_distributed_training_pipeline()
