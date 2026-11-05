from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def set_reproducible_seed(seed: int = 42, deterministic: bool = True) -> dict[str, Any]:
    """
    Enforce deterministic execution across Python, NumPy, and PyTorch.
    
    Parameters:
        seed: Integer random seed.
        deterministic: If True, enables PyTorch deterministic algorithms where supported.
        
    Returns:
        Dictionary detailing the applied reproducibility configuration.
    """
    # 1. Python standard library seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy seed
    np.random.seed(seed)

    # 3. PyTorch CPU seed
    torch.manual_seed(seed)

    # 4. PyTorch CUDA seeds (if available)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. PyTorch Determinism settings
    if deterministic:
        if cuda_available:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            # warn_only=True ensures operations that have no deterministic implementation
            # raise a warning rather than crashing the execution.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    return {
        "seed": seed,
        "deterministic": deterministic,
        "cuda_seeded": cuda_available,
        "pythonhashseed_set": bool(os.environ.get("PYTHONHASHSEED") == str(seed)),
    }


def get_git_commit_hash() -> str | None:
    """Safely retrieves the current Git commit hash if in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_environment_provenance() -> dict[str, Any]:
    """
    Captures complete system, library, and hardware provenance for scientific reproducibility.
    """
    cuda_available = torch.cuda.is_available()
    gpu_info = None
    if cuda_available:
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "gpu_info": gpu_info,
        "git_commit": get_git_commit_hash(),
    }


def get_nondeterministic_operations_disclosure() -> list[str]:
    """
    Scientific disclosure listing operations that may exhibit non-deterministic behavior
    despite global seed setting.
    """
    return [
        "Multi-threaded PyTorch DataLoader workers can introduce non-deterministic sample arrival order.",
        "Certain PyTorch CUDA backward operations (e.g. atomicAdd in scatter/gather or specific conv backprops) lack deterministic implementations.",
        "Adaptive average pooling across differing SIMD instruction sets (AVX2 vs AVX-512) can yield minor floating point discrepancies.",
        "Multi-threaded CPU BLAS operations (OpenBLAS/MKL) may sum floating-point numbers in non-associative order.",
    ]

