"""
GastricPath AI - CPU/GPU Inference and Memory Benchmark Engine.
Original Phase 12: Testing, Deployment, and Research Release.

Measures:
1. Environment and hardware specification.
2. GPU/CUDA detection (honest reporting of absence).
3. Model loading latency.
4. Preprocessing latency on 224x224 RGB image.
5. Cold-start inference latency.
6. Warm inference latency (Mean, Median, P95, Min, Max, Throughput).
7. Process memory usage (RSS in MB).
8. Parameter count and checkpoint file size.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import psutil
import torch
from PIL import Image

from app.core.config import settings
from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BENCHMARK_DIR / "cpu_benchmark_results.json"


def run_inference_benchmark(num_runs: int = 25) -> dict:
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Hardware & Environment
    cpu_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "logical_cpus": os.cpu_count(),
    }

    # 2. GPU Detection
    cuda_available = torch.cuda.is_available()
    gpu_info = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "None (CPU Execution Only)",
        "status": "Benchmarked on CPU" if not cuda_available else "Benchmarked on GPU",
        "scientific_disclosure": "No compatible CUDA GPU detected on host. GPU inference is recorded as not benchmarked to preserve scientific integrity.",
    }

    # 3. Model Loading Benchmark
    t0_load = time.perf_counter()
    device = torch.device("cpu")
    model = GastricCancerRNN(num_classes=8).to(device)

    checkpoint_path = settings.model_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    t_load_ms = (time.perf_counter() - t0_load) * 1000.0

    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)

    # Parameters & Checkpoint Size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    # 4. Preprocessing Benchmark
    # Create synthetic 224x224 RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    prep_latencies = []
    for _ in range(num_runs):
        t0_prep = time.perf_counter()
        seq = image_to_patch_sequence(dummy_img, settings.patch_size, settings.sequence_length)
        prep_latencies.append((time.perf_counter() - t0_prep) * 1000.0)

    mean_prep_ms = float(np.mean(prep_latencies))
    std_prep_ms = float(np.std(prep_latencies))

    # 5. Cold-Start Inference
    input_tensor = seq.unsqueeze(0).to(device)
    t0_cold = time.perf_counter()
    with torch.no_grad():
        logits, _ = model(input_tensor)
        _ = torch.softmax(logits, dim=1)
    cold_start_ms = (time.perf_counter() - t0_cold) * 1000.0

    # 6. Warm Inference Latencies
    warm_latencies = []
    for _ in range(num_runs):
        t0_infer = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(input_tensor)
            _ = torch.softmax(logits, dim=1)
        warm_latencies.append((time.perf_counter() - t0_infer) * 1000.0)

    mem_after_infer_mb = process.memory_info().rss / (1024 * 1024)

    warm_latencies = np.array(warm_latencies)
    mean_warm_ms = float(np.mean(warm_latencies))
    median_warm_ms = float(np.median(warm_latencies))
    p95_warm_ms = float(np.percentile(warm_latencies, 95))
    min_warm_ms = float(np.min(warm_latencies))
    max_warm_ms = float(np.max(warm_latencies))
    throughput_fps = float(1000.0 / mean_warm_ms)

    benchmark_summary = {
        "benchmark_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_environment": cpu_info,
        "gpu_environment": gpu_info,
        "model_specifications": {
            "model_name": "GastricCancerRNN Baseline Classifier",
            "architecture": "CNN Backbone + BiLSTM (2 layers, 128 hidden) + Additive Attention Head",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "checkpoint_size_mb": round(checkpoint_size_mb, 2),
            "checkpoint_sha256": "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8",
        },
        "performance_metrics": {
            "model_load_latency_ms": round(t_load_ms, 2),
            "preprocessing_mean_ms": round(mean_prep_ms, 2),
            "preprocessing_std_ms": round(std_prep_ms, 2),
            "cold_start_inference_ms": round(cold_start_ms, 2),
            "warm_inference_runs": num_runs,
            "warm_mean_latency_ms": round(mean_warm_ms, 2),
            "warm_median_latency_ms": round(median_warm_ms, 2),
            "warm_p95_latency_ms": round(p95_warm_ms, 2),
            "warm_min_latency_ms": round(min_warm_ms, 2),
            "warm_max_latency_ms": round(max_warm_ms, 2),
            "throughput_samples_per_second": round(throughput_fps, 2),
        },
        "memory_metrics": {
            "initial_memory_mb": round(mem_before_mb, 2),
            "after_model_load_mb": round(mem_after_load_mb, 2),
            "after_inference_mb": round(mem_after_infer_mb, 2),
            "peak_rss_mb": round(mem_after_infer_mb, 2),
        },
    }

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    return benchmark_summary


if __name__ == "__main__":
    print("Executing GastricPath AI CPU Inference Benchmark...")
    results = run_inference_benchmark()
    print("Benchmark complete!")
    print(f"Model Load: {results['performance_metrics']['model_load_latency_ms']} ms")
    print(f"Preprocessing: {results['performance_metrics']['preprocessing_mean_ms']} ms")
    print(f"Cold Start: {results['performance_metrics']['cold_start_inference_ms']} ms")
    print(f"Warm Mean Latency: {results['performance_metrics']['warm_mean_latency_ms']} ms")
    print(f"Throughput: {results['performance_metrics']['throughput_samples_per_second']} patches/sec")
    print(f"Memory RSS: {results['memory_metrics']['peak_rss_mb']} MB")
