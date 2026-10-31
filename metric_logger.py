import os
import json
import pika
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML Enterprise Ingestion & Proxy Gateway")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq-service.default.svc.cluster.local")
# Precision Fix: Hardcoded fallback completely decoupled from naming collisions and pointing to port 8000
VLLM_SERVICE_URL = os.getenv("TARGET_VLLM_URL", "http://cluster.local")

class PromptRequest(BaseModel):
    prompt: str

def send_to_queue(payload: dict):
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()
        channel.queue_declare(queue='ml_tasks', durable=True)
        channel.basic_publish(
            exchange='',
            routing_key='ml_tasks',
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
    return {"status": "gateway-routing-active", "mesh": "FastAPI -> CoreDNS -> vLLM Core"}

@app.post("/predict")
def route_inference_request(request: PromptRequest):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    audit_log = {
        "event": "inference_request_received",
        "prompt_snippet": request.prompt[:30],
        "timestamp": timestamp
    }
    send_to_queue(audit_log)
    
    try:
        print(f"📡 [DNS ROUTING] Forwarding token metrics directly to: {VLLM_SERVICE_URL}")
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
        raise HTTPException(status_code=503, detail=f"Target model serving pod unreachable via internal DNS mesh. Error: {str(e)}")
