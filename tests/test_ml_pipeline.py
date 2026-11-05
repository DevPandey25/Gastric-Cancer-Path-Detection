from pathlib import Path
import torch
from PIL import Image

from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence, load_rgb_image, softmax


def test_image_preprocessing(tmp_path: Path):
    # Create test image with non-standard dimensions
    test_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    img_path = tmp_path / "raw_tissue.png"
    test_img.save(img_path)

    # 1. Test load_rgb_image resizing
    resized = load_rgb_image(img_path, 224)
    assert resized.size == (224, 224)
    assert resized.mode == "RGB"

    # 2. Test patch sequence generation (224 / 56 = 4x4 = 16 patches)
    sequence = image_to_patch_sequence(resized, patch_size=56, sequence_length=16)
    assert isinstance(sequence, torch.Tensor)
    assert sequence.shape == (16, 3, 56, 56)
    assert sequence.min() >= 0.0
    assert sequence.max() <= 1.0


def test_softmax_utility():
    probs = softmax([2.0, 1.0, 0.1])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[0] > probs[1] > probs[2]


def test_gastric_rnn_forward_pass():
    model = GastricCancerRNN(num_classes=8, feature_dim=128, hidden_dim=128, num_layers=2)
    model.eval()

    batch_size = 2
    # Tensor shape: [batch, sequence_length=16, channels=3, patch_h=56, patch_w=56]
    dummy_input = torch.rand(batch_size, 16, 3, 56, 56)

    with torch.no_grad():
        logits, attention = model(dummy_input)

    # Validate output tensor shapes
    assert logits.shape == (batch_size, 8)
    assert attention.shape == (batch_size, 16)

    # Validate that attention weights per sample sum to 1.0
    for i in range(batch_size):
        attn_sum = attention[i].sum().item()
        assert abs(attn_sum - 1.0) < 1e-5
