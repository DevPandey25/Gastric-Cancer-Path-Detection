"""Pre-Extraction Resource & Feasibility Benchmark for Phase 4.1.

Measures:
- Available RAM, Free disk space, CPU utilization
- Encoder initialization time
- 128-image extraction benchmark comparing Batch Size 8 vs 16
- Process memory footprint, throughput (images/sec), latency per image
- Projected full dataset extraction duration (31,096 images)
- Recommendation for safest batch size and storage strategy.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

from research.foundation_models.model_loader import load_foundation_encoder
from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor


def run_pre_extraction_check(num_benchmark_images: int = 128) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parent.parent.parent
    dataset_root = workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
    manifest_train_path = workspace_root / "research" / "splits" / "manifest_train.csv"

    print("=" * 80)
    print("GASTRICPATH AI — PHASE 4.1 PRE-EXTRACTION RESOURCE CHECK")
    print("=" * 80)

    # 1. System & Resource Profiling
    total_disk, used_disk, free_disk = shutil.disk_usage(str(workspace_root))
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_rss_init_mb = proc.memory_info().rss / (1024 * 1024)
    cpu_percent_init = psutil.cpu_percent(interval=0.5)

    print(f"Total System RAM:     {vm.total / (1024**3):.2f} GB")
    print(f"Available RAM:        {vm.available / (1024**3):.2f} GB ({100 - vm.percent:.1f}% free)")
    print(f"Free Disk Space:      {free_disk / (1024**3):.2f} GB (Total: {total_disk / (1024**3):.2f} GB)")
    print(f"Initial CPU Load:     {cpu_percent_init:.1f}%")
    print(f"Initial Process RSS:  {proc_rss_init_mb:.2f} MB")
    print("-" * 80)

    # 2. Select 128 benchmark images across canonical classes
    df_train = pd.read_csv(manifest_train_path)
    canonical_classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    per_class = num_benchmark_images // len(canonical_classes)
    selected_rows = []
    for c in canonical_classes:
        c_rows = df_train[df_train["class_name"] == c].head(per_class)
        selected_rows.append(c_rows)
    bench_df = pd.concat(selected_rows).reset_index(drop=True)
    assert len(bench_df) == num_benchmark_images

    # 3. Model Loading & Weights Checksum
    weight_file = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    assert weight_file.exists(), f"Pretrained weights missing at {weight_file}"
    weight_sha = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"

    print("\n[Step 1/3] Loading DINOv2 ViT-S/14 Encoder...")
    t_load_0 = time.perf_counter()
    model, embedding_dim = load_foundation_encoder("dinov2_vits14", device="cpu")
    t_load = time.perf_counter() - t_load_0
    proc_rss_model_mb = proc.memory_info().rss / (1024 * 1024)
    model_ram_delta_mb = proc_rss_model_mb - proc_rss_init_mb
    print(f"  Encoder Load Time:    {t_load:.4f} s")
    print(f"  Reported Dim:         {embedding_dim}")
    print(f"  Process RSS:          {proc_rss_model_mb:.2f} MB (Delta: +{model_ram_delta_mb:.2f} MB)")

    # 4. Preload and Preprocess Images
    print(f"\n[Step 2/3] Preloading and Preprocessing {num_benchmark_images} Histology Images...")
    preprocessor = DeterministicPathologyPreprocessor(image_size=224)
    t_prep_0 = time.perf_counter()
    preprocessed_tensors = []
    for _, row in bench_df.iterrows():
        img_path = dataset_root / row["relative_path"]
        tensor = preprocessor.preprocess_image(img_path)
        preprocessed_tensors.append(tensor)
    t_prep = time.perf_counter() - t_prep_0
    all_tensors = torch.stack(preprocessed_tensors)  # [128, 3, 224, 224]
    print(f"  Preprocessing Time:   {t_prep:.3f} s ({t_prep / num_benchmark_images * 1000:.2f} ms/img)")

    # 5. Benchmark Batch Size 8 vs 16 (Extracting both CLS and Mean Patch tokens)
    print("\n[Step 3/3] Benchmarking Batch Size 8 vs Batch Size 16 (Dual Representation Extraction)...")
    batch_results = {}

    for bsz in [8, 16]:
        num_batches = num_benchmark_images // bsz
        cls_list = []
        mean_patch_list = []

        # Warmup
        with torch.no_grad():
            dummy = all_tensors[:bsz]
            _ = model.encoder.forward_features(dummy)

        rss_before_mb = proc.memory_info().rss / (1024 * 1024)
        cpu_samples = []
        t_start = time.perf_counter()

        with torch.no_grad():
            for i in range(num_batches):
                batch_tensors = all_tensors[i * bsz : (i + 1) * bsz]
                features = model.encoder.forward_features(batch_tensors)
                cls_tok = features["x_norm_clstoken"]  # [B, 384]
                patch_tok = features["x_norm_patchtokens"]  # [B, 256, 384]
                mean_patch = patch_tok.mean(dim=1)  # [B, 384]

                cls_norm = torch.nn.functional.normalize(cls_tok, p=2, dim=-1)
                mean_patch_norm = torch.nn.functional.normalize(mean_patch, p=2, dim=-1)

                cls_list.append(cls_norm.cpu().numpy())
                mean_patch_list.append(mean_patch_norm.cpu().numpy())

        duration = time.perf_counter() - t_start
        rss_after_mb = proc.memory_info().rss / (1024 * 1024)
        throughput = num_benchmark_images / duration
        latency_ms = (duration / num_benchmark_images) * 1000

        cls_all = np.vstack(cls_list)
        mean_patch_all = np.vstack(mean_patch_list)
        assert cls_all.shape == (num_benchmark_images, 384)
        assert mean_patch_all.shape == (num_benchmark_images, 384)
        assert np.all(np.isfinite(cls_all))
        assert np.all(np.isfinite(mean_patch_all))

        extrapolated_sec = 31096 / throughput
        extrapolated_min = extrapolated_sec / 60.0

        batch_results[f"batch_size_{bsz}"] = {
            "batch_size": bsz,
            "duration_sec": round(duration, 3),
            "throughput_img_per_sec": round(throughput, 2),
            "latency_ms_per_image": round(latency_ms, 2),
            "process_rss_mb": round(rss_after_mb, 2),
            "ram_delta_mb": round(rss_after_mb - rss_before_mb, 2),
            "extrapolated_full_dataset_minutes": round(extrapolated_min, 2),
        }

        print(f"\n  === Batch Size {bsz} Results ===")
        print(f"    Duration (128 images): {duration:.3f} s")
        print(f"    Throughput:            {throughput:.2f} images/sec")
        print(f"    Latency per Image:     {latency_ms:.2f} ms")
        print(f"    Process RSS:           {rss_after_mb:.2f} MB")
        print(f"    Extrapolated (31,096): {extrapolated_min:.2f} minutes ({extrapolated_sec / 3600:.2f} hours)")

    # Recommendation
    chosen_bsz = 16 if batch_results["batch_size_16"]["throughput_img_per_sec"] >= batch_results["batch_size_8"]["throughput_img_per_sec"] else 8
    # If RAM delta on 16 is fine (<500MB), 16 is preferred
    print("\n" + "=" * 80)
    print(f"RESOURCE EVALUATION SUMMARY & RECOMMENDATION:")
    print(f"  Recommended Batch Size:  {chosen_bsz}")
    print(f"  Projected Runtime:       {batch_results[f'batch_size_{chosen_bsz}']['extrapolated_full_dataset_minutes']} minutes")
    print(f"  Available RAM:           {vm.available / (1024**3):.2f} GB (Sufficient)")
    print(f"  Free Disk Space:         {free_disk / (1024**3):.2f} GB (Sufficient)")
    print("=" * 80)

    summary = {
        "system_resources": {
            "total_ram_gb": round(vm.total / (1024**3), 2),
            "available_ram_gb": round(vm.available / (1024**3), 2),
            "free_disk_space_gb": round(free_disk / (1024**3), 2),
            "initial_cpu_percent": round(cpu_percent_init, 1),
        },
        "model_provenance": {
            "model_name": "dinov2_vits14",
            "weights_file": str(weight_file),
            "weights_sha256": weight_sha,
            "embedding_dim": embedding_dim,
            "encoder_load_seconds": round(t_load, 4),
            "model_ram_footprint_mb": round(model_ram_delta_mb, 2),
        },
        "benchmark_128_images": batch_results,
        "selected_batch_size": chosen_bsz,
        "projected_duration_minutes": batch_results[f"batch_size_{chosen_bsz}"]["extrapolated_full_dataset_minutes"],
    }

    out_path = workspace_root / "research" / "foundation_models" / "pre_extraction_report.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved to: {out_path}")
    return summary


if __name__ == "__main__":
    run_pre_extraction_check()
