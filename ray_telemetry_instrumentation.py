import ray
import time
import numpy as np

if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- STATEFUL NETWORK TELEMETRY AGGREGATOR ---
@ray.remote
class ClusterNetworkTelemetryRegistry:
    def __init__(self):
        # Local state database tracking bytes transferred over virtual lanes
        self.metrics_store = {"total_bytes_transferred": 0, "total_packets_sent": 0}

    def record_data_plane_traffic(self, array_size_bytes: int):
        """Atomically increments the cluster's network data plane tracking metrics."""
        self.metrics_store["total_bytes_transferred"] += array_size_bytes
        self.metrics_store["total_packets_sent"] += 1
        return self.metrics_store.copy()

    def fetch_global_metrics(self):
        return self.metrics_store.copy()

# --- DISTRIBUTED PARALLEL GRADIENT TRANSMISSION TASK ---
@ray.remote
def transmit_gradient_packet_mesh(worker_id: int, target_node_id: int, data_vector: np.ndarray, telemetry_actor_ref):
    """Simulates a Ring AllReduce node transmitting a sliced gradient packet over the link."""
    start_tx = time.time()
    
    # Calculate exact array size footprint in memory (float64 = 8 bytes per element)
    array_bytes = data_vector.nbytes
    
    # Simulate network fiber latency over the link
    time.sleep(0.1)
    
    # Atomically log metrics to the stateful tracking registrar
    updated_telemetry = ray.get(telemetry_actor_ref.record_data_plane_traffic.remote(array_bytes))
    
    tx_latency = (time.time() - start_tx) * 1000
    print(f"📡 [LINK CHANNEL] Worker {worker_id} ──> Node {target_node_id} | Transmitted {array_bytes / (1024*1024):.2f} MB in {tx_latency:.2f}ms")
    return updated_telemetry

def run_telemetry_instrumentation_suite():
    print("🚀 [ORCHESTRATOR] Initializing Interconnect Telemetry Custom Instrumentation Engine...")
    telemetry_registry = ClusterNetworkTelemetryRegistry.remote()
    
    # 1. Simulate a heavy 2,500,000 element gradient tensor slice (20 Megabytes)
    print("📊 Allocating 20MB mock model gradient matrix sheet arrays...")
    mock_gradients = np.random.randn(2500000)
    
    # 2. Launch 4 parallel concurrent network transfer operations across the ring
    print("🔀 Injecting concurrent gradient data plane transmissions into the loop...")
    tx_refs = [
        transmit_gradient_packet_mesh.remote(worker_id=i, target_node_id=(i+1)%4, data_vector=mock_gradients, telemetry_actor_ref=telemetry_registry)
        for i in range(4)
    ]
    
    # 3. Complete a synchronization barrier block to resolve execution metrics
    ray.get(tx_refs)
    
    # 4. Pull ground-truth final metrics report out of the registry
    print("\n⏳ Fetching global cluster network metrics from the database registry...")
    final_report = ray.get(telemetry_registry.fetch_global_metrics.remote())
    
    print("\n🏆 --- DISTRIBUTED DATA PLANE TELEMETRY REPORT ---")
    print(f"📦 Total Packets Transmitted over Cluster Mesh: {final_report['total_packets_sent']} packets")
    print(f"📊 Total Data Ingested over Ring Links: {final_report['total_bytes_transferred'] / (1024*1024):.2f} MB")
    print("🏁 [ORCHESTRATOR] Custom Interconnect Telemetry Capture Finalized Successfully.")

if __name__ == "__main__":
    run_telemetry_instrumentation_suite()
