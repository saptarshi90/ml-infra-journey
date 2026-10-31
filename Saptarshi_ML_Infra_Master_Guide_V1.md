# ML Infrastructure & Systems Engineering
## Technical Reference Log (Phase 1 & 2 Master Study Guide)

### 1. Device Topology & Resource Isolation Layer
To safely isolate high-leverage machine learning and infrastructure software pipelines without creating system latency, local virtual memory spaces are bounded at the kernel hypervisor gate.

*   **Host Silicon Specifications:** 11th Gen Intel Core i5-1135G7 @ 2.40GHz | 8.00 GB Physical RAM.
*   **Hardware Co-Processor Acceleration:** NVIDIA GeForce MX330 Dedicated GPU (2GB VRAM / CUDA Engine Ready for Phase 3 Model Quantization inference loops).
*   **Hypervisor Isolation Vector:** Windows Subsystem for Linux (WSL 2) running a native utility virtual machine environment.

#### The Master Resource Limit Matrix (.wslconfig Configuration)
Located at the host filesystem root route `/mnt/c/Users/saptarshi/.wslconfig` to enforce a strict memory fence, ensuring complete laptop stability during high-throughput client tutoring links:
```text
[wsl2]
memory=2560MB
```

---

### 2. Production Linux Networking Fabric
Analyzing system network registry socket states entirely via the text command console interface to decode protocol layer routing rules.

#### Layer 4 Protocol Architectural Comparison
*   **TCP (Transmission Control Protocol):** Connection-oriented. Enforces an explicit, synchronous 3-way handshake sequence before payload streaming. Displays explicit `LISTEN` status flags when a service room socket is waiting to intercept client connections. Used for API gateways, model server interfaces, and reverse proxy streams requiring zero data packet loss bounds.
*   **UDP (User Datagram Protocol):** Connectionless. Streams raw datagrams instantly at destination addresses without tracking transmission verification metrics. Leaves the state column completely `BLANK` because the port operates like an open physical drop slot. Used for high-speed DNS data queries, video streaming pipelines, and telemetry metric transmission fabrics.

#### The Netstat System Diagnostic Scan Metric
```text
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
tcp        0      0 127.0.0.53:53           0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.54:53           0.0.0.0:*               LISTEN
tcp        0      0 10.255.255.254:53       0.0.0.0:*               LISTEN
udp        0      0 127.0.0.54:53           0.0.0.0:*
udp        0      0 127.0.0.53:53           0.0.0.0:*
```
*Note: Sockets bound to `127.0.0.X` address layers denote local loopback structures where the computer communicates strictly with itself inside an isolated scope. Port 53 is universally reserved by operating systems to handle background Domain Name System (DNS) query lookups.*

---

### 3. Telemetry Automation & Shell Script Architecture
Combining the procedural coordination power of Bash scripting with the granular data-processing capability of Python to track kernel status fields natively.

#### The Core Data Collector: metric_logger.py
A native telemetry script engineered to read the Linux virtual file structure array `/proc/meminfo`, split text streams, and log converted memory statistics into an append-only time-series CSV database file:
```python
import os
import time
import csv
from datetime import datetime

log_file = 'system_metrics.csv'

def get_ram_metrics():
    with open('/proc/meminfo', 'r') as f:
        lines = f.readlines()
    mem_total = 0
    mem_free = 0
    for line in lines:
        if 'MemTotal' in line:
            mem_total = int(line.split()[1])
        if 'MemFree' in line:
            mem_free = int(line.split()[1])
    mem_used = mem_total - mem_free
    return round(mem_used / 1024, 2)

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Memory_Used_MB'])

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ram_used = get_ram_metrics()

with open(log_file, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([timestamp, ram_used])

print(f'[{timestamp}] Successfully logged metrics: {ram_used} MB used.')
```

#### The Timing Coordination Engine: auto_monitor.sh
An autonomous shell automation script that acts as a continuous background timing loop process daemon, triggering the collector agent precisely every 5 seconds:
```bash
#!/bin/bash

echo "Starting system telemetry daemon... Press [Ctrl+C] to stop."

while true
do
    python3 metric_logger.py
    sleep 5
done
```
*Note: Newly instantiated text script applications require explicit permission bit upgrades before the kernel will authorize execution hooks. The command `chmod +x auto_monitor.sh` flips the metadata security switches to an active 755 execution state.*

---

### 4. Virtualization & Container Isolation Layer
Mastering the shift from environment dependency friction to portable, cloud-native containerized delivery formats.

#### Virtual Machines vs. Docker Containers
*   **Virtual Machines (VMs):** Heavy architecture. Boots a completely isolated, redundant Guest OS kernel layer on top of a hypervisor. Pre-allocates massive, rigid chunks of hardware system memory blocks. Slow boot lifecycle parameters (minutes).
*   **Docker Containers:** Ultra-lightweight agility. Bypasses guest operating system duplication entirely by sharing the Host OS Linux Kernel directly via native isolation namespaces and cgroups. Boots instantaneously in milliseconds and tracks zero extra memory footprint inflation beyond the target code process requirements.

#### The Deployment Blueprint Manifest: Dockerfile
The explicit recipe file used to bundle your custom scripts and their complete language dependencies into a standalone, reproducible container image layer:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY metric_logger.py auto_monitor.sh ./
RUN chmod +x auto_monitor.sh
CMD ["./auto_monitor.sh"]
```
