# Griptape Nodes Face Parsing Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for face parsing using [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch).

## Overview

This library segments a face image into 19 parts: skin, eyebrows, eyes, glasses, ears, earrings, nose, mouth, lips, neck, necklace, clothing, hair and hat. The **Face Parsing** node returns a black and white mask of the parts you pick, which is useful for inpainting, relighting, or recoloring hair and lips. It also returns a color overlay that previews every detected part, so you can check the result at a glance.

## Requirements

- **GPU**: CUDA (NVIDIA) or MPS (Apple Silicon) is used when available. CPU is supported but slower.
- **Griptape Nodes Engine**: Version 0.104.1 or later

## Nodes

### Face Parsing

Segments a face and creates a black and white mask of the parts you select, plus a color overlay preview.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | HuggingFace model | The face parsing weights to use. |
| `image` | ImageUrlArtifact | Image containing a face to segment. |
| `parts` | Multi-select | Face parts to include in the mask. Defaults to Skin. Leave empty to produce only the overlay. |
| `invert` | bool | Swap the mask so the selected parts are black and everything else is white. |
| `output_file` | File path | Where to save the mask. Defaults to `face_mask.png`. |
| `overlay_file` | File path | Where to save the overlay. Defaults to `face_overlay.png`. |
| `mask` | ImageUrlArtifact (output) | White where a selected part appears, black elsewhere. |
| `overlay` | ImageUrlArtifact (output) | The input image with every detected part tinted in its own color. The overlay always shows all parts, whatever you select. |

**Available parts:** Skin, Left eyebrow, Right eyebrow, Left eye, Right eye, Glasses, Left ear, Right ear, Earring, Nose, Mouth, Upper lip, Lower lip, Neck, Necklace, Clothing, Hair, Hat. To mask the background, select every part and turn on `invert`.

Both nodes also have the standard **Was Successful** and **Result Details** outputs, and a **Failed** exit.

## Available Models

The following models are available from HuggingFace:

| Model | Description |
|-------|-------------|
| `vivym/face-parsing-bisenet` | A mirror of the BiSeNet weights (`79999_iter.pth`) trained on CelebAMask-HQ. |

Models are downloaded through Griptape Nodes Model Management and cached for subsequent runs. The upstream project distributes these weights through Google Drive, so this library uses a third-party HuggingFace mirror. That repository does not declare a license, so check that your use is permitted.

## Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- Optional: a CUDA-capable NVIDIA GPU or an Apple Silicon Mac for faster inference

### Install the Library

1. **Clone the repository** to your Griptape Nodes workspace directory:

   ```bash
   cd `gtn config show workspace_directory`
   git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-library-researchlabs.git
   ```

   The repository holds several libraries, each in its own folder. This one is in `face_parsing/`.

2. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-library-researchlabs/face_parsing/griptape_nodes_library_face_parsing/griptape-nodes-library.json
     ```
   - You can check your workspace directory with `gtn config show workspace_directory`
   - Close the Settings Panel
   - Click on *Refresh Libraries*

3. **Download the model** from the Model Management page in the editor if the node shows "Model Download Required".

4. **Verify installation** by checking that the nodes appear in the node palette as **Face Parsing** under **Image → Mask**.

## Usage

### Face Parsing

1. Add a **Face Parsing** node to your workflow
2. Connect an image that contains a face to `image`
3. Pick the face parts to include in `parts`, for example Hair
4. Turn on `invert` if you want everything except those parts
5. Run the node and connect `mask` to your next node, such as an inpainting or compositing node
6. Connect `overlay` to a display if you want to check what the model found

The model was trained on aligned, cropped faces and works best on a single, front-facing face. It runs at 512x512 internally and scales the result back to your image size.

## Troubleshooting

### Library Not Loading

- Ensure the git submodule is initialized. If you cloned without `--recurse-submodules`, run:
  ```bash
  git submodule update --init --recursive
  ```

### GPU Not Available

- The nodes fall back to CPU automatically, so they still work, just slower
- For NVIDIA GPUs, verify your drivers are up to date and CUDA is properly installed
- For Apple Silicon, ensure you're running on macOS 12.3 or later

### Out of Memory Errors

- Close other GPU-intensive applications
- Resize very large images before parsing; the model only sees a 512x512 version anyway

## Additional Resources

- [face-parsing.PyTorch GitHub](https://github.com/zllrunning/face-parsing.PyTorch)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0. The bundled face-parsing.PyTorch submodule is subject to its own license: MIT.
