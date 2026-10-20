"""BiSeNet face parsing inference shared by the library's nodes.

The upstream model code lives in the ``face_parsing_pytorch`` submodule. The advanced library puts that
directory on ``sys.path`` before any node loads, so the upstream ``model`` module is imported lazily here.
"""

import io
import logging
from typing import Any

import numpy as np
import torch
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.files.file import File
from huggingface_hub import hf_hub_download
from PIL import Image

logger = logging.getLogger("face_parsing_library")

INPUT_SIZE = 512
NUM_CLASSES = 19
CHECKPOINT_FILENAME = "79999_iter.pth"

# Index order matches the training labels in upstream prepropess_data.py (0 is background).
CLASS_NAMES = [
    "Background",
    "Skin",
    "Left eyebrow",
    "Right eyebrow",
    "Left eye",
    "Right eye",
    "Glasses",
    "Left ear",
    "Right ear",
    "Earring",
    "Nose",
    "Mouth",
    "Upper lip",
    "Lower lip",
    "Neck",
    "Necklace",
    "Clothing",
    "Hair",
    "Hat",
]

# Overlay colors from upstream vis_parsing_maps, indexed by class.
PART_COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Keeps one checkpoint resident so switching nodes or repos does not stack models in memory.
_model_cache: dict[tuple[str, str, str], torch.nn.Module] = {}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(repo_id: str, revision: str) -> torch.nn.Module:
    device = get_device()
    key = (repo_id, revision, device)
    if key in _model_cache:
        return _model_cache[key]

    # DEFERRED IMPORT: the upstream code is only importable once the advanced library has set up sys.path.
    from model import BiSeNet
    from resnet import Resnet18

    # Upstream downloads ImageNet resnet18 weights from pytorch.org on every construction. The BiSeNet checkpoint
    # contains the backbone weights too, so skip the download instead of patching the submodule.
    Resnet18.init_weight = lambda self: None  # type: ignore[method-assign]

    _model_cache.clear()
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=CHECKPOINT_FILENAME, revision=revision)
    logger.info("Loading face parsing weights from %s on %s", checkpoint_path, device)
    net = BiSeNet(n_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    _model_cache[key] = net
    return net


def read_image(artifact: Any) -> Image.Image:
    """Decode an image artifact. The caller has already checked the type."""
    if isinstance(artifact, ImageUrlArtifact):
        data = File(artifact.value).read_bytes()
    elif isinstance(artifact, ImageArtifact):
        data = artifact.value
    else:
        msg = "Expected an image artifact."
        raise TypeError(msg)
    return Image.open(io.BytesIO(data)).convert("RGB")


def predict_labels(net: torch.nn.Module, image: Image.Image) -> np.ndarray:
    """Return a label map the same size as ``image``, with one class index per pixel."""
    resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    normalized = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(next(net.parameters()).device)

    with torch.inference_mode():
        logits = net(tensor)[0]
    small_labels = logits.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

    # Nearest-neighbor keeps label ids intact when scaling back to the input size.
    labels = Image.fromarray(small_labels).resize(image.size, Image.Resampling.NEAREST)
    return np.asarray(labels)


def colorize_overlay(image: Image.Image, labels: np.ndarray) -> Image.Image:
    """Blend a colored label map over the image, the same way upstream visualizes it."""
    base = np.asarray(image, dtype=np.float32)
    colored = np.full_like(base, 255.0)
    for index in range(1, NUM_CLASSES):
        colored[labels == index] = PART_COLORS[index]
    blended = 0.4 * base + 0.6 * colored
    return Image.fromarray(blended.astype(np.uint8))


def selection_mask(labels: np.ndarray, class_indices: list[int], invert: bool) -> Image.Image:
    """White where a selected class appears, black elsewhere."""
    selected = np.isin(labels, class_indices)
    if invert:
        selected = ~selected
    return Image.fromarray((selected * 255).astype(np.uint8), mode="L")


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
