import ray
import time
import urllib.request
import numpy as np
from prometheus_client import start_http_server, Counter, Gauge

if not ray.is_initialized():
    ray.init(address="auto", logging_level="error")

# --- PROMETHEUS METRIC DEFINITIONS ---
RING_BYTES_COUNTER = Counter(
    "ray_ring_allreduce_bytes_total",
    "Total bytes transferred across distributed Ring AllReduce links"
)
RING_PACKETS_COUNTER = Counter(
    "ray_ring_allreduce_packets_total",
    "Total gradient tensor packets transmitted across nodes"
)
LINK_BANDWIDTH_GAUGE = Gauge(
    "ray_ring_link_throughput_mbps",
    "Instantaneous throughput of the last gradient transfer in MB/s"
)

# --- STATEFUL TELEMETRY AGGREGATOR ACTOR ---
@ray.remote
class PrometheusTelemetryBridge:
    def __init__(self):
        self.total_bytes = 0
        self.total_packets = 0

    def record_transfer(self, size_bytes: int, duration_sec: float):
        self.total_bytes += size_bytes
        self.total_packets += 1
        throughput_mbps = (size_bytes / (1024 * 1024)) / max(duration_sec, 1e-6)
        return {
            "bytes": size_bytes,
            "throughput_mbps": throughput_mbps,
            "cumulative_mb": self.total_bytes / (1024 * 1024)
        }

# --- DISTRIBUTED WORKER TRANSMISSION TASK ---
@ray.remote
def simulate_ring_transfer(worker_id: int, bridge_actor_ref):
    start_tx = time.time()
    # Allocate a 10 MB gradient shard (1,310,720 float64 elements)
    tensor_shard = np.random.randn(1310720)
    time.sleep(0.15)  # Simulate network wire serialization
    duration = time.time() - start_tx

    stats = ray.get(bridge_actor_ref.record_transfer.remote(tensor_shard.nbytes, duration))
    return stats

def run_exporter_pipeline():
    print("🚀 [ORCHESTRATOR] Starting Prometheus HTTP Metrics Server on port 8000...")
    start_http_server(8000)

    bridge = PrometheusTelemetryBridge.remote()

    print("📡 [ORCHESTRATOR] Executing 4 concurrent Ring AllReduce gradient transfers...")
    refs = [simulate_ring_transfer.remote(i, bridge) for i in range(4)]
    results = ray.get(refs)

    # Update live Prometheus gauges and counters on the driver node
    for res in results:
        RING_BYTES_COUNTER.inc(res["bytes"])
        RING_PACKETS_COUNTER.inc(1)
        LINK_BANDWIDTH_GAUGE.set(res["throughput_mbps"])

    print("\n🔍 [SCRAPE TEST] Querying live Prometheus HTTP endpoint (http://localhost:8000/metrics)...")
    response = urllib.request.urlopen("http://localhost:8000/metrics").read().decode("utf-8")

    print("\n🏆 --- LIVE PROMETHEUS SCRAPE PAYLOAD ---")
    for line in response.splitlines():
        if line.startswith("ray_ring_"):
            print(line)

    print("\n🏁 [ORCHESTRATOR] Prometheus HTTP Exporter Validation Complete.")

if __name__ == "__main__":
    run_exporter_pipeline()
