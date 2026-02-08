# GSAM-SD Inpainting

A comprehensive image inpainting solution that combines **Grounding DINO**, **Segment Anything Model (SAM)**, and **Stable Diffusion** to intelligently detect, segment, and inpaint regions in images.

## Overview

This repository implements an advanced inpainting pipeline for diverse scenarios including:
- **Clothing transformation**: Upper and lower body clothing style changes (e.g., casual to formal wear)
- **Object removal**: Remove unwanted objects (e.g., helmets) from images
- **Region-based inpainting**: Intelligently target and modify specific image regions

The system leverages state-of-the-art models:
- **Grounding DINO**: Text-prompted object detection
- **SAM**: Precise semantic segmentation of detected regions
- **Stable Diffusion Inpainting**: High-quality inpainting with text guidance

## Project Structure

```
├── img_base3-c.py                      # Main batch inpainting pipeline
├── inference_GSAM_c-multi.py           # Multi-task inference and evaluation
├── sd_baseline_inference_multi.py      # Baseline SD inference
├── sd_baseline_batch.py                # Batch baseline processing
├── sample_results/                     # Example output results
└── out_*/                              # Output directories by task
    ├── out_upper_formal/               # Upper body formal clothing results
    ├── out_lower_casual/               # Lower body casual clothing results
    ├── out_remove_helmet/              # Helmet removal results
    └── out_uniform_navy/               # Navy uniform results
```

## Features

- **Multi-task Support**: Handle multiple inpainting scenarios in a single pipeline
- **Batch Processing**: Process multiple images efficiently
- **Quality Metrics**: Built-in evaluation metrics (SSIM, LPIPS, FID, CLIP similarity)
- **Debug Visualization**: Save intermediate steps for debugging and analysis
- **GPU Acceleration**: CUDA support for fast inference

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 1.9+

### Dependencies

```bash
pip install torch torchvision
pip install diffusers transformers
pip install opencv-python pillow numpy scipy
pip install tqdm

# Optional metrics dependencies
pip install lpips scikit-image
```

### Model Downloads

The pipeline requires the following pre-trained models:
- Grounding DINO weights
- SAM (Segment Anything Model) checkpoint
- Stable Diffusion Inpainting model (automatically downloaded via HuggingFace)

## Usage

### Basic Inpainting Pipeline

```python
from img_base3-c import ModelBundle

# Initialize models
bundle = ModelBundle(
    groundingdino_cfg="path/to/GroundingDINO_SwinB_cfg.py",
    groundingdino_ckpt="path/to/groundingdino_swinb_cogvoi.pth",
    sam_ckpt="path/to/sam_vit_h_4b8939.pth",
    sd_inpaint_repo="stabilityai/stable-diffusion-2-inpainting"
)

# Process image
image = Image.open("input_image.jpg")
prompt = "upper body wearing a white dress shirt and a navy blazer"
result = bundle.inpaint(image, prompt)
```

### Batch Processing

Run the batch inpainting script for multiple images:

```bash
python img_base3-c.py
```

### Evaluation and Metrics

Evaluate results using multiple quality metrics:

```bash
python inference_GSAM_c-multi.py
```

## Supported Tasks

| Task | Description | Prompt Example |
|------|-------------|----------------|
| `upper_to_formal` | Transform upper body to formal wear | "upper body wearing a white dress shirt and a navy blazer" |
| `lower_to_casual` | Transform lower body to casual wear | "lower body wearing blue denim jeans and white canvas sneakers" |
| `remove_helmet` | Remove helmet from image | "no helmet" |
| `uniform_navy` | Apply navy uniform | "wearing navy uniform" |

## Evaluation Metrics

The pipeline includes several quality metrics:
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FID**: Fréchet Inception Distance
- **CLIP Score**: Semantic alignment with text prompt  

(Main metric is FID in paper. Other Metrics are etc.)

## Dataset

The experiment dataset is available at:
- [Google Drive Link](https://drive.google.com/file/d/19vzA0c1cUrT_5X30qddUWViq7AEcUPsN/view?usp=sharing)

Download the dataset and extract it to your working directory to use with the pipeline.

## Publication

If you use this work in your research, please cite:

```bibtex
@article{Augmenting_Construction_Safety_Datasets_2025,
  title={Augmenting Construction Safety Datasets with Training-Free Diffusion Based PPE Editing},
  journal={Journal of Korean Information and Communications Society (J-KICS)},
  year={2025, published at 2026}
}
```

This work demonstrates the application of our GSAM-SD inpainting pipeline for augmenting construction safety datasets through intelligent PPE (Personal Protective Equipment) editing.

## Output Structure

Results are organized by task in separate output directories:
- Each task creates a corresponding `out_<task_name>/` directory
- Images are saved with task-specific suffixes (e.g., `_upper_to_formal`)
- Debug visualizations show detection boxes and segmentation masks

## Performance Notes

- GPU memory: ~12-16GB recommended for batch processing
- Inference time: ~5-10 seconds per image depending on resolution
- Model sizes: ~2.5GB total for all models

## References

- [Grounding DINO](https://github.com/IDEA-Research/Grounding-DINO)
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)

## License

Please refer to the individual model licenses for usage terms.

## Authors

DGIST DIP Lab

---

For issues, questions, or contributions, please create an issue or submit a pull request.
