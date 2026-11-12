import random
import numpy as np
import torch

from research.reproducibility import (
    get_environment_provenance,
    get_nondeterministic_operations_disclosure,
    set_reproducible_seed,
)


def test_seed_determinism():
    # Test NumPy reproducibility
    set_reproducible_seed(1234)
    arr1 = np.random.rand(5)

    set_reproducible_seed(1234)
    arr2 = np.random.rand(5)

    np.testing.assert_allclose(arr1, arr2)

    # Test PyTorch reproducibility
    set_reproducible_seed(1234)
    t1 = torch.randn(4, 4)

    set_reproducible_seed(1234)
    t2 = torch.randn(4, 4)

    assert torch.all(t1.eq(t2))

    # Test Python random
    set_reproducible_seed(1234)
    r1 = [random.random() for _ in range(5)]

    set_reproducible_seed(1234)
    r2 = [random.random() for _ in range(5)]

    assert r1 == r2


def test_environment_provenance():
    prov = get_environment_provenance()
    assert "python_version" in prov
    assert "torch_version" in prov
    assert "numpy_version" in prov
    assert "platform" in prov
    assert "git_commit" in prov
    assert prov["cpu_count"] is not None
    assert isinstance(prov["cuda_available"], bool)


def test_nondeterministic_disclosures():
    disclosures = get_nondeterministic_operations_disclosure()
    assert isinstance(disclosures, list)
    assert len(disclosures) >= 3
    assert any("DataLoader" in d for d in disclosures)

