"""Tests for Benchmark Model Architectures."""

from pathlib import Path
import torch
from torch import nn

from research.benchmark.models import (
    create_benchmark_model,
    GastricCancerRNNWrapper,
    ResNet50Classifier,
    EfficientNetB0Classifier,
)


def test_model_instantiation_and_forward_shapes():
    # Model A: GastricCancerRNN (Sequence mode)
    model_a, mode_a = create_benchmark_model("GastricCancerRNN", num_classes=8)
    assert mode_a == "sequence"
    seq_input = torch.randn(2, 16, 3, 56, 56)
    out_a = model_a(seq_input)
    assert out_a.shape == (2, 8)

    # Model B: ResNet50 (2D mode)
    model_b, mode_b = create_benchmark_model("ResNet50", num_classes=8, pretrained=False)
    assert mode_b == "2d"
    img_input = torch.randn(2, 3, 224, 224)
    out_b = model_b(img_input)
    assert out_b.shape == (2, 8)

    # Model C: EfficientNet-B0 (2D mode)
    model_c, mode_c = create_benchmark_model("EfficientNet-B0", num_classes=8, pretrained=False)
    assert mode_c == "2d"
    out_c = model_c(img_input)
    assert out_c.shape == (2, 8)


def test_load_actual_baseline_checkpoint():
    cp_path = Path("models/checkpoints/gastric_rnn_classifier.pt")
    if cp_path.exists():
        model, mode = create_benchmark_model("GastricCancerRNN", num_classes=8, checkpoint_path=cp_path)
        assert mode == "sequence"
        model.eval()
        dummy_seq = torch.randn(1, 16, 3, 56, 56)
        with torch.no_grad():
            logits = model(dummy_seq)
        assert logits.shape == (1, 8)

