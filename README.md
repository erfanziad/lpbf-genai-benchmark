# LPBF Generative AI Benchmark

A rigorous benchmarking study of four conditional generative architectures for synthesizing in-situ LPBF layerwise process imagery, with a focus on the **fidelity–latency trade-off** critical for real-time industrial monitoring.

> **Paper:** Ziad, E., Ju, F., Lu, Y. "Dual-Conditioned Generative Modeling for LPBF Monitoring: A Fidelity–Latency Benchmark of Diffusion and GAN Architectures." *ASME MSEC 2026*, State College, PA.

---

## Overview

Labeled defect data in LPBF is scarce and severely class-imbalanced. This project addresses that bottleneck by developing and benchmarking generative models that synthesize targeted **bright spot** and **streak** anomalies under a unified **dual-conditioning framework**:

- **Lighting condition** (LED a / b / c) — captures surface reflectivity variation across three in-situ illumination setups
- **Binary anomaly mask** (bright spot + streak channels) — provides pixel-level spatial guidance for defect placement

All four architectures are trained and evaluated on the **NIST AMMT dataset** (996 unique part-layers from four identical LPBF builds).

---

## Architectures Benchmarked

| Model | Family | Parameters | Inference Latency |
|---|---|---|---|
| **Proposed LDM** | Diffusion (ControlNet + SD 1.5) | 1,304 M | 1.76 s |
| VAE-DDPM | Diffusion (two-stage) | ~20 M | ~2.5 s |
| StyleGAN2-ADA | GAN | ~30 M | 1.21 ms |
| WGAN-GP | GAN | ~25 M | 1.23 ms |

### Key Results (Part 04 hold-out)

| Metric | WGAN-GP | StyleGAN2 | VAE-DDPM | **Proposed LDM** |
|---|---|---|---|---|
| FID ↓ | 277.77 | 249.18 | 293.27 | **44.04** |
| KID ↓ | 0.2822 | 0.2353 | 0.2581 | **0.0152** |
| LPIPS ↑ | 0.5280 | 0.5083 | 0.4670 | 0.1302 |

The LDM achieves state-of-the-art fidelity at the cost of a ~1,450× latency increase over GAN baselines.

---

## Repository Structure

```
lpbf-gen-ai-benchmark/
│
├── lpbf_gen_ai_benchmark.py    # Main pipeline (dataset, models, training, evaluation)
├── README.md
├── LICENSE
├── requirements.txt
│
└── checkpoints/                # Generated at runtime
    └── stylegan/
        ├── G_best.pth
        └── validation_grid.png
```

---

## Requirements

```
python >= 3.10
torch >= 2.0
torchvision
torchmetrics[image]
diffusers
transformers
accelerate
numpy
pillow
matplotlib
tqdm
```

Install:

```bash
pip install -r requirements.txt
```

The LDM pipeline additionally requires:

```bash
pip install diffusers transformers accelerate
```

---

## Data

This project uses the **NIST AMMT dataset** (Lane & Yeung, 2020), consisting of layerwise optical images from four LPBF builds of identical overhang geometries. Images are captured before (`A`) and after (`B`) each laser scan under three lighting conditions (`a`, `b`, `c`).

**Expected data format** — two pickle files:

```
data/
├── labeled_training_set.pkl     # Parts 01–03 with spot/streak masks
└── unlabeled_training_set.pkl   # Parts 01–03 without masks (self-supervised)
```

Each pickle is a dict keyed by `part01`, `part02`, etc. Each part contains a list of layer dicts with:
- `images`: dict mapping `A0035a`, `A0035b`, `A0035c` → H×W×3 uint8 arrays
- `spot_label`: H×W binary mask
- `streak_label`: H×W binary mask

**Data split:** Parts 01–03 → training (747 layers), Part 04 → test (249 layers).

---

## Usage

### 1. Configure paths

Edit the configuration block at the top of `lpbf_gen_ai_benchmark.py`:

```python
DATA_ROOT      = Path("data")
LABELED_PATH   = DATA_ROOT / "labeled_training_set.pkl"
UNLABELED_PATH = DATA_ROOT / "unlabeled_training_set.pkl"
```

### 2. Train StyleGAN2-ADA

```bash
python lpbf_gen_ai_benchmark.py
```

This runs the full pipeline: data loading → training → evaluation → qualitative grid.

### 3. Run LDM inference benchmark (optional)

Uncomment the LDM block at the bottom of `main` to benchmark the ControlNet-driven Stable Diffusion pipeline against the trained GAN. Requires a CUDA GPU with ≥8 GB VRAM.

---

## Model Details

### Dual-Conditioning Mechanism

All architectures share the same conditioning signals:

```
z (latent noise)
    + light_embedding(y_light)  →  style modulation
    + mask_encoder(m_mask)      →  spatial feature injection at 18×32
```

The **ControlNet-driven LDM** additionally processes the anomaly mask through a 361M-parameter adapter that injects learned spatial residuals into the 859M U-Net backbone via zero-convolution layers.

### StyleGAN2-ADA Architecture

```
z (512) → Mapping Network (MLP) → w (512)
                                        ↓
                                Lighting Embedding
                                        ↓
Const (1024, 9×16) → ResBlock×4 (with InstanceNorm + γ modulation)
                              ↑
               MaskEncoder(2ch → 512ch at 18×32) fused at up1
                                        ↓
                              RGB output (144×256)
```

Key design choices:
- **InstanceNorm** over BatchNorm for small-dataset stability
- **R1 gradient penalty** instead of WGAN-GP (more stable with ADA)
- **Adaptive Discriminator Augmentation (ADA)** to prevent D memorization on 747 training layers
- **Spectral normalization** on all discriminator layers
- **Lighting projection head** in D for explicit conditioning enforcement

---

## Tiered Deployment Strategy

| Scenario | Recommended Model | Rationale |
|---|---|---|
| Offline synthetic dataset generation | **LDM** | Highest fidelity (FID 44, KID 0.015); best for training downstream classifiers |
| Real-time in-situ monitoring | **StyleGAN2-ADA** or **WGAN-GP** | Sub-2ms inference; viable for closed-loop control |
| Edge hardware (8GB VRAM) | **WGAN-GP** | Smallest parameter count; fastest fine-tuning (<8 min) |

---

## Citation

```bibtex
@inproceedings{ziad2026lpbf,
  author    = {Ziad, Erfan and Ju, Feng and Lu, Yan},
  title     = {Dual-Conditioned Generative Modeling for {LPBF} Monitoring:
               A Fidelity--Latency Benchmark of Diffusion and {GAN} Architectures},
  booktitle = {Proceedings of the ASME International Manufacturing
               Science and Engineering Conference (MSEC)},
  year      = {2026},
  address   = {State College, PA}
}
```

---

## Acknowledgments

This work was supported in part by the National Institute of Standards and Technology (NIST). The NIST AMMT dataset was provided by B. Lane and H. Yeung (NIST, 2020).

*Disclaimer: Identification of commercial systems does not imply recommendation or endorsement by NIST.*

---

## License

MIT License. See `LICENSE` for details.
