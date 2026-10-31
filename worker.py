import os
import time
import json
import pika

# Internal cluster network routing parameters for RabbitMQ
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq-service.default.svc.cluster.local")

def process_ml_task(ch, method, properties, body):
    """Callback function triggered automatically the moment a message arrives."""
    try:
        # Decode the raw binary packet payload back into structural JSON
        payload = json.loads(body.decode('utf-8'))
        print(f"\n📥 [WORKER RECEIVED] New job extracted from queue matrix!")
        print(f"   |-- Task Type: {payload.get('task_type')}")
        print(f"   |-- Data Block: {payload.get('data')}")
        print(f"   |-- Inbound Timestamp: {payload.get('inbound_timestamp')}")
        
        # Simulate heavy deep learning compute cycles (matrix math, tensor load)
        print("⏳ [COMPUTE] Simulating background ML model inference...")
        time.sleep(3)  # Heavy work simulation
        print("✅ [COMPUTE SUCCESS] Inference cycle complete!")
        
        # CRITICAL SYSTEMS STEP: Send explicit delivery acknowledgment back to broker!
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("🚀 [ACK SENT] Message safely cleared from broker queue storage.")
        
    except Exception as e:
        print(f"🔴 Worker Processing Exception: {str(e)}")

def start_consumer():
    print("📡 Initializing Decoupled Background Worker Grid...")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    
    # Declaratively guarantee the target queue exists inside the pipeline broker
    channel.queue_declare(queue='ml_tasks', durable=True)
    
    # Fair Dispatch System configuration: Tell broker to only send 1 job at a time to this worker
    channel.basic_qos(prefetch_count=1)
    
    # Bind the queue channel to our worker processing callback system
    channel.basic_consume(queue='ml_tasks', on_message_callback=process_ml_task)
    
    print("🚀 Worker Engine Fully Armed. Listening intently for data streams... Press Ctrl+C to exit.")
    channel.start_consuming()

if __name__ == "__main__":
    try:
        start_consumer()
    except KeyboardInterrupt:
        print("\n🛑 Worker Engine shutdown gracefully via dashboard cockpit exit.")
