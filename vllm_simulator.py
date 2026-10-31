import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Gauge, Counter

app = FastAPI(title="vLLM Specialized High-Throughput Inference Engine with Metrics")

# ==========================================
# PROMETHEUS HARDWARE METRICS DECLARATION
# ==========================================
VRAM_FREE_BLOCKS_GAUGE = Gauge(
    "vllm_free_vram_blocks", 
    "Total number of unallocated PagedAttention memory blocks inside physical VRAM pool"
)
CUDA_OOM_COUNTER = Counter(
    "vllm_cuda_oom_total", 
    "Total number of times incoming inference request stream triggered a CUDA Out-of-Memory throttling guardrail"
)

class PagedAttentionMemoryManager:
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.free_blocks = total_blocks
        self.allocated_blocks_map = {}
        # Synchronize our initial hardware profile state with Prometheus registers
        VRAM_FREE_BLOCKS_GAUGE.set(self.free_blocks)

    def allocate_request(self, request_id: str, prompt_tokens: int) -> bool:
        required_blocks = (prompt_tokens + 3) // 4
        if required_blocks <= self.free_blocks:
            self.free_blocks -= required_blocks
            self.allocated_blocks_map[request_id] = required_blocks
            # Update live VRAM gauge metric
            VRAM_FREE_BLOCKS_GAUGE.set(self.free_blocks)
            return True
        
        # Increment the Prometheus CUDA OOM counter tracker
        CUDA_OOM_COUNTER.inc()
        return False

    def release_request(self, request_id: str):
        if request_id in self.allocated_blocks_map:
            self.free_blocks += self.allocated_blocks_map[request_id]
            del self.allocated_blocks_map[request_id]
            VRAM_FREE_BLOCKS_GAUGE.set(self.free_blocks)

# Initializing a mock 16GB VRAM pool profile mapping 32 blocks
gpu_vram = PagedAttentionMemoryManager(total_blocks=32)

class InferenceRequest(BaseModel):
    prompt: str

# Expose the standard Prometheus tracking endpoint metrics route (/metrics)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
def check_hardware_state():
    return {
        "engine": "vllm-paged-attention-active",
        "free_vram_blocks": gpu_vram.free_blocks,
        "total_vram_blocks": gpu_vram.total_blocks
    }

@app.post("/predict")
def run_paged_inference(request: InferenceRequest):
    request_id = str(uuid.uuid4())[:8]
    tokens_count = len(request.prompt.split()) + 3
    
    success = gpu_vram.allocate_request(request_id, tokens_count)
    
    if success:
        allocated_pages = gpu_vram.allocated_blocks_map[request_id]
        print(f"⚡ [CUDA KERNEL] Allocated {allocated_pages} PagedAttention blocks for req [{request_id}].")
        return {
            "status": "success",
            "request_id": request_id,
            "allocated_blocks": allocated_pages,
            "inference_text": f"Simulated output response token generation stream completed for prompt: {request.prompt[:20]}..."
        }
    else:
        return {
            "status": "error",
            "detail": "🔴 CUDA Out-of-Memory! Engine throttled inbound prompt request stream."
        }
