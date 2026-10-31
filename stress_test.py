import time
import requests
import threading

GATEWAY_URL = "http://localhost:8080/predict"
PROMPT_PAYLOAD = {
    "prompt": "Surgically optimize parallel CUDA kernel threads and distribute massive tensor blocks cleanly across VRAM tracks"
}

def fire_inference_request(request_id: int):
    try:
        print(f"🚀 [LOAD GENERATOR] Firing payload shot #{request_id}")
        response = requests.post(GATEWAY_URL, json=PROMPT_PAYLOAD, timeout=5.0)
        print(f"📥 [RESPONSE #{request_id}] Status Code: {response.status_code} -> Data: {response.text}")
    except Exception as e:
        print(f"🔴 [CONNECTION FAILURE #{request_id}] Gateway timed out or throttled: {str(e)}")

def run_stress_surge():
    threads = []
    print("⚡ WAKING UP CONCURRENT THREAD POOL ENGINE... LAUNCHING BURST ATTACK!")
    
    # Fire 12 heavy concurrent prompts simultaneously to consume all available PagedAttention blocks
    for i in range(12):
        t = threading.Thread(target=fire_inference_request, args=(i+1,))
        threads.append(t)
        t.start()
        # Micro-sleep delay to simulate natural network bursts
        time.sleep(0.05)

    for t in threads:
        t.join()
        
    print("🏁 [SURGE COMPLETE] Concurrent token payload stream closed out.")

if __name__ == "__main__":
    run_stress_surge()
