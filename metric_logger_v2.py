import json
import pika
import requests
import random
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import generate_latest, Gauge, REGISTRY

app = FastAPI(title="ML Enterprise Ingestion & Proxy Gateway + GPU Telemetry")

RABBITMQ_HOST = "rabbitmq-service.default.svc.cluster.local"
# Explicit CoreDNS service pointer on port 8000
VLLM_SERVICE_URL = "http://vllm-service:8000/predict"

# --- TELEMETRY INSTRUMENTATION REGISTRY ---
VRAM_BLOCKS_FREE = Gauge(
    'vllm_free_vram_blocks',
    'Total number of unallocated PagedAttention memory blocks inside physical VRAM pool'
)
CUDA_OOM_TOTAL = Gauge(
    'vllm_cuda_oom_total',
    'Total number of times incoming inference request stream triggered a CUDA Out-of-Memory throttling guardrail'
)

# Initialize standard cluster baselines
VRAM_BLOCKS_FREE.set(512.0)
CUDA_OOM_TOTAL.set(0.0)

class PromptRequest(BaseModel):
    prompt: str

def send_to_queue(payload: dict):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue="ml_tasks", durable=True)
        channel.basic_publish(
            exchange="",
            routing_key="ml_tasks",
            body=json.dumps(payload),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()
        return True
    except Exception as e:
        print(f"🔴 AMQP Logging Failure: {str(e)}")
        return False

@app.get("/")
def read_root():
    return {"status": "gateway-routing-telemetry-active", "mesh": "FastAPI -> Prometheus Registry"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest(REGISTRY).decode("utf-8")

@app.post("/predict")
def route_inference_request(request: PromptRequest):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    audit_log = {
        "event": "inference_request_received",
        "prompt_snippet": request.prompt[:30],
        "timestamp": timestamp
    }
    send_to_queue(audit_log)
    
    try:
        print(f"📡 [DNS ROUTING] Forwarding token metrics directly to: {VLLM_SERVICE_URL}")
        
        # Simulate active PagedAttention allocation changes per surge request
        current_free = random.choice([30.0, 45.0, 12.0, 88.0])
        VRAM_BLOCKS_FREE.set(current_free)
        if current_free < 15.0:
            CUDA_OOM_TOTAL.set(1.0)

        response = requests.post(
            VLLM_SERVICE_URL, 
            json={"prompt": request.prompt},
            timeout=5.0
        )
        vllm_data = response.json()
        
        if response.status_code == 200 and vllm_data.get("status") == "success":
            return {
                "gateway_status": "forwarded_successfully",
                "timestamp": timestamp,
                "vllm_response": vllm_data
            }
        else:
            raise HTTPException(status_code=502, detail=f"vLLM Engine Backend Throttled: {vllm_data.get('detail', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"🔴 CoreDNS Inter-Pod Resolution Failure: {str(e)}")
        # Increment the OOM/Failure counter on drop
        CUDA_OOM_TOTAL.inc()
        raise HTTPException(status_code=503, detail=f"Target model serving pod unreachable via internal DNS mesh. Error: {str(e)}")
