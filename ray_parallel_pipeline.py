import ray
import time
import numpy as np

# Connect directly to your lightweight, running native host cluster instance
if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

@ray.remote
def process_data_shard(shard_id: int, data_chunk: np.ndarray):
    """Stateless parallel worker executing heavy vector operations on a data shard."""
    start_time = time.time()
    
    # Simulate data transformation pipeline logic (e.g., standardizing feature weights)
    mean = float(np.mean(data_chunk))
    std = float(np.std(data_chunk))
    normalized_chunk = (data_chunk - mean) / (std + 1e-5)
    transformed_metric = float(np.sum(np.square(normalized_chunk)))
    
    execution_time = (time.time() - start_time) * 1000
    return {
        "shard": shard_id,
        "metric": transformed_metric,
        "latency_ms": execution_time
    }

def run_distributed_pipeline():
    print("🚀 [ORCHESTRATOR] Initializing High-Throughput Parallel Processing Pipeline...")
    
    # 1. Generate a massive base mock dataset (10 Million elements)
    print("📊 Generating 10,000,000 element raw dataset space...")
    raw_dataset = np.random.randn(10000000)
    
    # 2. Shard the dataset dynamically into 4 distinct parallel execution blocks
    print("✂️ Splitting dataset into 4 parallel shards for processing...")
    shards = np.array_split(raw_dataset, 4)
    
    # 3. Pin the shards into the zero-copy shared memory registry using implicit reference routing
    print("📌 Ingesting data shards into the Plasma shared memory pool...")
    start_ingest = time.time()
    shard_refs = [ray.put(shard) for shard in shards]
    print(f"🆔 Generated 4 Shared Memory Object References | Ingestion Time: {(time.time() - start_ingest)*1000:.2f}ms")
    
    # 4. Fire the concurrent mapping tasks across the worker topology simultaneously
    print("\n📡 Launching parallel execution pipeline workers...")
    pipeline_refs = [
        process_data_shard.remote(shard_id=i, data_chunk=shard_refs[i]) 
        for i in range(len(shard_refs))
    ]
    
    # 5. Resolve the data metrics out of the object store registers
    print("⏳ Resolving transformation metrics from the distributed grid...")
    results = ray.get(pipeline_refs)
    
    print("\n🏆 --- PIPELINE METRICS REPORT ---")
    total_latency = 0
    for res in results:
        print(f"🔹 [Shard {res['shard']}] Sum Sq Metric: {res['metric']:.2f} | Worker Compute Latency: {res['latency_ms']:.2f}ms")
        total_latency += res['latency_ms']
        
    print(f"\n📊 Total Distributed Compute Shard Time: {total_latency:.2f}ms")
    print("🏁 [ORCHESTRATOR] Parallel Data Processing Complete with Zero Data-Duplication Overhead.")

if __name__ == "__main__":
    run_distributed_pipeline()
