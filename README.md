# Distributed AI & ML Infrastructure Engineering Portfolio

[![ML Infrastructure CI Validation](https://github.com)](https://github.com)

A 6-month hands-on systems engineering repository documenting the design, implementation, observability, and stress benchmarking of distributed machine learning infrastructure.

---

## System Architecture Overview

    +-----------------------------------------------------------------------+

    |                     CI/CD & OBSERVABILITY PLANE                       |
    |  [GitHub Actions CI] ---> [Prometheus Exporter] ---> [Grafana JSON]   |
    +-----------------------------------------------------------------------+
                                       |
                                       v
    +-----------------------------------------------------------------------+

    |                  ORCHESTRATION & RESOURCE SCHEDULING                  |
    |  [Kubernetes / KubeRay] ---> [Stateful GPU Slot Lease Registry]       |
    +-----------------------------------------------------------------------+
                                       |
                                       v
    +-----------------------------------------------------------------------+

    |             DISTRIBUTED COMPUTE, MEMORY & DATA PLANE                  |
    |  [PyTorch DDP]  <--->  [Ring AllReduce Mesh]  <--->  [Plasma Memory]  |
    |  [Async Prefetcher]    [Preemption Recovery]         [Async Disk I/O] |
    +-----------------------------------------------------------------------+

---

## Verified System Benchmarks

| Subsystem | Module | Measured Production Metric |
| :--- | :--- | :--- |
| **Zero-Copy Shared Memory** | `ray_cluster_stress_audit.py` | **76.41 MB/s** effective throughput across 8 contending workers (0 dropped tasks) |
| **Ring AllReduce Fabric** | `ray_prometheus_exporter.py` | **44.17 MB/s** link bandwidth exposed over OpenMetrics HTTP `/metrics` |
| **Async Data Prefetching** | `ray_prefetch_pipeline.py` | **2x latency reduction** (881 ms down to 416 ms) with 0 I/O cache stalls |
| **Out-of-Band Checkpointing** | `ray_checkpoint_engine.py` | **0 ms compute worker block** while persisting PyTorch tensors asynchronously |

---

## 6-Month Engineering Progression

* **Month 1 – Linux & System Fundamentals:** System telemetry scripts (`auto_monitor.sh`, `metric_logger_v2.py`), process monitoring, and LRU memory caching (`lru_cache.py`).
* **Month 2 – Containerization & Reverse Proxies:** Custom `Dockerfile`, multi-container orchestration (`docker-compose.yml`), and reverse proxy load routing (`nginx.conf`).
* **Month 3 – Distributed Queues & Inference:** Decoupled message brokers (`queue_systems.py`, `rabbitmq-deployment.yml`), stream processing, and LLM serving simulation (`vllm_simulator.py`).
* **Month 4 – Kubernetes & Observability:** Declarative pods, services, RBAC manifests (`prometheus-rbac.yml`), and Prometheus/Grafana deployments.
* **Month 5 – KubeRay & Zero-Copy Memory:** KubeRay cluster specifications (`ray-cluster.yml`), distributed data shuffling (`ray_data_shuffle.py`), and Plasma object store optimization.
* **Month 6 – Distributed Training & CI/CD:** PyTorch DDP orchestration (`ray_distributed_training.py`), Ring AllReduce collective math (`ray_allreduce_mesh.py`), stateful GPU lease registries (`ray_distributed_mesh.py`), fault-tolerance recovery (`ray_plasma_bench.py`), and automated GitHub Actions CI validation.
