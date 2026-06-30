"""
Dual-Conditioned Generative Modeling for LPBF In-Situ Monitoring
=================================================================
Benchmarks four conditional generative architectures for synthesizing
LPBF layerwise process imagery conditioned on:
  - Lighting condition (LED a / b / c)
  - Binary anomaly masks (bright spot + streak)

Architectures:
  1. StyleGAN2-ADA  (dual-conditioned, with Adaptive Discriminator Augmentation)
  2. WGAN-GP        (dual-conditioned, lightweight)
  3. VAE-DDPM       (two-stage latent diffusion baseline)
  4. LDM            (ControlNet-driven Stable Diffusion, proposed)

Evaluation axes: FID, KID, LPIPS (fidelity / diversity) + inference latency.

Reference paper:
  Ziad, Erfan, Ju, Feng, Lu, Yan.
  "Dual-Conditioned Generative Modeling for LPBF Monitoring:
   A Fidelity-Latency Benchmark of Diffusion and GAN Architectures."
  ASME MSEC 2026, State College, PA.

Author: Erfan Ziad
"""

# ============================================================
# 0) Imports
# ============================================================
import os
import time
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


# ============================================================
# 1) Configuration
# ============================================================

# --- Data paths (update to your environment) ---
DATA_ROOT     = Path("data")
LABELED_PATH  = DATA_ROOT / "labeled_training_set.pkl"
UNLABELED_PATH = DATA_ROOT / "unlabeled_training_set.pkl"

# --- Image geometry ---
H, W = 139, 250                      # Raw ROI resolution
PAD_T, PAD_B, PAD_L, PAD_R = 2, 3, 3, 3   # → padded to 144×256
LIGHT_TO_ID = {"a": 0, "b": 1, "c": 2}

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2) Dataset
# ============================================================

def _as_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure array is H×W×3 uint8."""
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.clip(arr, 0, 255).astype(np.uint8)


def to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """Convert H×W×3 uint8 → padded C×H×W float tensor in [-1, 1]."""
    x = torch.from_numpy(img_np).float() / 255.0
    x = x.permute(2, 0, 1) * 2 - 1
    return F.pad(x.unsqueeze(0), (PAD_L, PAD_R, PAD_T, PAD_B),
                 mode="reflect").squeeze(0)


def _extract_images_from_layer(layer: dict) -> list[tuple[np.ndarray, int]]:
    """Return [(image_uint8, lighting_id), ...] for all lighting variants in a layer."""
    out = []
    imgs = layer.get("images", {})
    if not isinstance(imgs, dict):
        return out
    for k, v in imgs.items():
        if v is None:
            continue
        letter = str(k).strip().lower()[-1:]
        if letter in LIGHT_TO_ID:
            out.append((_as_hwc_uint8(v), LIGHT_TO_ID[letter]))
    return out


class LPBFDualCondDataset(Dataset):
    """
    Loads NIST AMMT layerwise images with dual conditioning signals:
      - Lighting condition (a / b / c)
      - Stacked binary anomaly mask (bright spot + streak)

    Data split: part01–03 → training, part04 → test.
    """

    def __init__(self, pkl_paths: list[Path], part_ids: list[int]):
        self.images, self.lighting, self.masks = [], [], []

        for p in pkl_paths:
            is_labeled = "labeled" in str(p).lower()
            with open(p, "rb") as f:
                data = pickle.load(f)

            items = data.items() if isinstance(data, dict) else [(None, data)]
            for part_key, layers in items:
                if part_key is not None:
                    try:
                        pid = int("".join(filter(str.isdigit, part_key)))
                        if pid not in part_ids:
                            continue
                    except ValueError:
                        continue

                for layer in layers:
                    if is_labeled:
                        spot   = layer.get("spot_label")   or np.zeros((H, W))
                        streak = layer.get("streak_label") or np.zeros((H, W))
                        mask   = np.stack([spot, streak], axis=0)
                    else:
                        mask = np.zeros((2, H, W))

                    for img_np, cid in _extract_images_from_layer(layer):
                        self.images.append(img_np)
                        self.lighting.append(cid)
                        self.masks.append(mask)

        print(f"Loaded {len(self.images)} images from parts {part_ids}.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):
        img   = to_tensor(self.images[i])
        light = torch.tensor(self.lighting[i], dtype=torch.long)
        mask  = torch.from_numpy(self.masks[i]).float()
        mask  = F.pad(mask.unsqueeze(0), (PAD_L, PAD_R, PAD_T, PAD_B),
                      mode="constant", value=0).squeeze(0)
        return img, light, mask


def build_dataloaders(batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) with part-wise split."""
    train_ds = LPBFDualCondDataset([LABELED_PATH, UNLABELED_PATH], part_ids=[1, 2, 3])
    test_ds  = LPBFDualCondDataset([LABELED_PATH],                  part_ids=[4])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_dl, test_dl


# ============================================================
# 3) Shared Building Blocks
# ============================================================

class AdaptiveAugmentation(nn.Module):
    """
    Stochastic noise augmentation applied to discriminator inputs.
    Probability p is tuned externally by ADAController.
    """

    def __init__(self):
        super().__init__()
        self.p = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p > 0 and np.random.rand() < self.p:
            x = x + torch.randn_like(x) * 0.05
        return x


class ADAController:
    """
    Tunes augmentation probability based on discriminator confidence.
    target: desired mean D(real) logit value (0.6 ≈ moderate confidence).
    """

    def __init__(self, target: float = 0.6, step: float = 0.005):
        self.p      = 0.0
        self.target = target
        self.step   = step

    def tune(self, d_real_logits: torch.Tensor):
        sign    = torch.sign(torch.tanh(d_real_logits.mean()) - self.target)
        self.p  = max(0.0, min(1.0, self.p + sign.item() * self.step))


class ResBlock(nn.Module):
    """
    StyleGAN2-style residual block with:
      - InstanceNorm for training stability on small datasets
      - Lighting-conditioned scale modulation (γ)
      - Optional 2× nearest-neighbor upsampling
      - 1/√2 skip scaling to prevent feature explosion
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1    = nn.InstanceNorm2d(in_ch)
        self.norm2    = nn.InstanceNorm2d(out_ch)
        self.light_emb = nn.Embedding(3, in_ch)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) \
                        if (upsample or in_ch != out_ch) else nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gamma = self.light_emb(y).view(-1, x.size(1), 1, 1)
        h     = self.norm1(x) * (1 + gamma)

        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        h = F.leaky_relu(self.conv1(h), 0.2)
        h = F.leaky_relu(self.norm2(self.conv2(h)), 0.2)
        return (h + self.skip(x)) * 0.707   # 1/√2 scaling


# ============================================================
# 4) StyleGAN2-ADA (Dual-Conditioned)
# ============================================================

class MaskEncoder(nn.Module):
    """Encodes a 2-channel binary mask to a 512-channel spatial feature map (18×32)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64,  3, padding=1),              nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  nn.LeakyReLU(0.2),  # 72×128
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  nn.LeakyReLU(0.2),  # 36×64
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  nn.LeakyReLU(0.2),  # 18×32
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m)


class StyleGAN2Generator(nn.Module):
    """
    Dual-conditioned StyleGAN2-ADA generator.
    Conditions: lighting (a/b/c) injected via learned embedding modulation;
                anomaly mask fused at the 18×32 spatial feature stage.
    """

    def __init__(self, z_dim: int = 512, base: int = 128):
        super().__init__()
        self.mapping      = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(),
                                          nn.Linear(z_dim, z_dim))
        self.light_emb    = nn.Embedding(3, z_dim)
        self.mask_encoder = MaskEncoder()
        self.const        = nn.Parameter(torch.randn(1, base * 8, 9, 16))   # 1024 ch

        self.up1 = ResBlock(base * 8, base * 4, upsample=True)  # 18×32, 512 ch
        self.up2 = ResBlock(base * 4, base * 2, upsample=True)  # 36×64
        self.up3 = ResBlock(base * 2, base,     upsample=True)  # 72×128
        self.up4 = ResBlock(base,     base,     upsample=True)  # 144×256

        self.to_rgb = nn.Sequential(nn.Conv2d(base, 3, 3, padding=1), nn.Tanh())

    def forward(self, z: torch.Tensor, y: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
        w   = self.mapping(z) + self.light_emb(y)
        m_f = self.mask_encoder(m)

        x = self.const.repeat(z.size(0), 1, 1, 1)
        x = self.up1(x, y) + m_f     # Spatial mask fusion at 18×32
        x = self.up2(x, y)
        x = self.up3(x, y)
        x = self.up4(x, y)
        return self.to_rgb(x)


class StyleGAN2Discriminator(nn.Module):
    """
    Dual-conditioned discriminator with:
      - Spectral normalization for Lipschitz stability
      - ADA augmentation module
      - Lighting projection head (inner-product conditioning)
    """

    def __init__(self, base: int = 64):
        super().__init__()
        self.ada = AdaptiveAugmentation()

        def sn_block(c1, c2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(c1, c2, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
            )

        self.main      = nn.Sequential(
            sn_block(3,        base),
            sn_block(base,     base * 2),
            sn_block(base * 2, base * 4),
            sn_block(base * 4, base * 8),
        )
        self.avg       = nn.AdaptiveAvgPool2d(1)
        self.fc        = nn.utils.spectral_norm(nn.Linear(base * 8, 1))
        self.light_emb = nn.Embedding(3, base * 8)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
        x = self.ada(x)
        h = self.avg(self.main(x)).flatten(1)
        return self.fc(h) + (h * self.light_emb(y)).sum(dim=1, keepdim=True)


# ============================================================
# 5) Training — StyleGAN2-ADA
# ============================================================

@dataclass
class StyleGANConfig:
    batch:      int   = 16
    total_kimg: int   = 150
    z_dim:      int   = 512
    lr_G:       float = 1e-4
    lr_D:       float = 1e-4
    r1_gamma:   float = 10.0
    ada_target: float = 0.6
    ckpt_dir:   Path  = Path("checkpoints/stylegan")


def compute_r1_penalty(D: nn.Module, real_img: torch.Tensor,
                        y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty: penalizes D's gradients on real data."""
    real_img = real_img.requires_grad_(True)
    logits   = D(real_img, y, m)
    grads    = torch.autograd.grad(
        outputs=logits.sum(), inputs=real_img, create_graph=True
    )[0]
    return grads.pow(2).view(grads.size(0), -1).sum(1).mean()


def train_stylegan(G: nn.Module, D: nn.Module, train_loader: DataLoader,
                   cfg: StyleGANConfig, device: torch.device):
    """StyleGAN2-ADA training loop with R1 regularization and ADA tuning."""
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    opt_G = optim.Adam(G.parameters(), lr=cfg.lr_G, betas=(0.0, 0.99))
    opt_D = optim.Adam(D.parameters(), lr=cfg.lr_D, betas=(0.0, 0.99))
    ada   = ADAController(target=cfg.ada_target)

    best_d_loss = float("inf")
    cur_kimg    = 0
    start       = time.time()

    while cur_kimg < cfg.total_kimg:
        pbar = tqdm(train_loader, desc=f"StyleGAN  {int(cur_kimg)}/{cfg.total_kimg} kimg")
        for real_img, y, m in pbar:
            real_img, y, m = real_img.to(device), y.to(device), m.to(device)
            bs = real_img.size(0)

            # --- Discriminator step ---
            opt_D.zero_grad()
            d_real  = D(real_img, y, m)
            z       = torch.randn(bs, cfg.z_dim, device=device)
            d_fake  = D(G(z, y, m).detach(), y, m)
            r1      = compute_r1_penalty(D, real_img, y, m)
            loss_D  = (F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
                       + (cfg.r1_gamma / 2) * r1)
            loss_D.backward()
            opt_D.step()

            # --- Generator step ---
            opt_G.zero_grad()
            z       = torch.randn(bs, cfg.z_dim, device=device)
            loss_G  = F.softplus(-D(G(z, y, m), y, m)).mean()
            loss_G.backward()
            opt_G.step()

            # ADA update
            ada.tune(d_real)
            D.ada.p  = ada.p
            cur_kimg += bs / 1000
            pbar.set_postfix(D=f"{loss_D.item():.3f}",
                             G=f"{loss_G.item():.3f}",
                             ADA=f"{ada.p:.2f}")

        if loss_D.item() < best_d_loss:
            best_d_loss = loss_D.item()
            torch.save(G.state_dict(), cfg.ckpt_dir / "G_best.pth")

    mins = (time.time() - start) / 60
    print(f"\nStyleGAN training complete in {mins:.1f} min. "
          f"Best D-loss: {best_d_loss:.4f}")


# ============================================================
# 6) LDM (ControlNet-driven Stable Diffusion)
# ============================================================

def build_ldm_pipeline(device: torch.device):
    """Load the SD-Inpainting + ControlNet pipeline for LDM benchmarking."""
    from diffusers import (StableDiffusionControlNetInpaintPipeline,
                           ControlNetModel)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float16,
        use_safetensors=False,
    ).to(device)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=False,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_ldm(pipe, init_pil: Image.Image, mask_pil: Image.Image,
                  num_steps: int = 30,
                  prompt: str = "high resolution metallic powder bed, "
                                "laser powder bed fusion, bright spot defect"
                  ) -> tuple[Image.Image, float]:
    """Run one LDM inference pass and return (output_image, latency_seconds)."""
    t0 = time.time()
    out = pipe(
        prompt=prompt,
        image=init_pil,
        mask_image=mask_pil,
        control_image=init_pil,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
    ).images[0]
    return out, time.time() - t0


# ============================================================
# 7) Evaluation
# ============================================================

def build_metrics(device: torch.device, kid_subset: int = 50):
    """Initialize FID, KID, and LPIPS metric objects."""
    return (
        FrechetInceptionDistance(feature=2048).to(device),
        KernelInceptionDistance(subset_size=kid_subset).to(device),
        LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device),
    )


def evaluate_generator(G: nn.Module, test_loader: DataLoader,
                        device: torch.device, z_dim: int = 512,
                        label: str = "Model") -> dict:
    """
    Compute FID, KID, and LPIPS for a GAN generator on the test set.
    Returns a dict with keys: fid, kid_mean, kid_std, lpips.
    """
    G.eval()
    fid_m, kid_m, lpips_m = build_metrics(device)

    with torch.no_grad():
        for real_img, y, m in tqdm(test_loader, desc=f"Evaluating {label}"):
            real_img, y, m = real_img.to(device), y.to(device), m.to(device)
            z        = torch.randn(real_img.size(0), z_dim, device=device)
            fake_img = G(z, y, m)

            real_01 = (real_img + 1) / 2
            fake_01 = (fake_img + 1) / 2
            real_u8 = (real_01 * 255).clamp(0, 255).byte()
            fake_u8 = (fake_01 * 255).clamp(0, 255).byte()

            fid_m.update(real_u8, real=True);  fid_m.update(fake_u8, real=False)
            kid_m.update(real_u8, real=True);  kid_m.update(fake_u8, real=False)
            lpips_m.update(fake_01, real_01)

    fid               = fid_m.compute().item()
    kid_mean, kid_std = kid_m.compute()
    lpips             = lpips_m.compute().item()

    print(f"\n{'='*40}")
    print(f"{label} — Part 04 Results")
    print(f"  FID  : {fid:.2f}")
    print(f"  KID  : {kid_mean.item():.4f} ± {kid_std.item():.4f}")
    print(f"  LPIPS: {lpips:.4f}")
    print(f"{'='*40}")
    return {"fid": fid, "kid_mean": kid_mean.item(),
            "kid_std": kid_std.item(), "lpips": lpips}


def measure_inference_latency(G: nn.Module, device: torch.device,
                               z_dim: int = 512, num_trials: int = 100) -> float:
    """
    Benchmark single-image inference latency (ms) via GPU-synchronized timing.
    Runs a warm-up of 10 passes before measuring.
    """
    G.eval()
    z = torch.randn(1, z_dim, device=device)
    y = torch.tensor([0], device=device)
    m = torch.zeros(1, 2, 144, 256, device=device)

    for _ in range(10):           # warm-up
        G(z, y, m)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_trials):
        G(z, y, m)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / num_trials * 1000

    print(f"Inference latency: {latency_ms:.2f} ms  "
          f"({1000 / latency_ms:.1f} Hz)")
    return latency_ms


# ============================================================
# 8) Qualitative Visualization
# ============================================================

def plot_validation_grid(G: nn.Module, test_loader: DataLoader,
                          device: torch.device, z_dim: int = 512,
                          num_samples: int = 6,
                          save_path: str = "validation_grid.png"):
    """
    Side-by-side grid: real Part 04 images (left) vs. generated images (right).
    Anomaly-seeded samples are annotated.
    """
    G.eval()
    real_imgs, y_lights, m_masks = next(iter(test_loader))
    real_imgs = real_imgs[:num_samples].to(device)
    y_lights  = y_lights[:num_samples].to(device)
    m_masks   = m_masks[:num_samples].to(device)

    with torch.no_grad():
        z         = torch.randn(num_samples, z_dim, device=device)
        fake_imgs = G(z, y_lights, m_masks)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    fig.suptitle("Part 04 (held-out):  Real  vs.  Generated", fontsize=14)

    def to_np(t):
        return np.clip((t.permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)

    for i in range(num_samples):
        light_label = {0: "a", 1: "b", 2: "c"}[y_lights[i].item()]
        axes[i, 0].imshow(to_np(real_imgs[i]))
        axes[i, 0].set_title(f"Real — Light {light_label}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(to_np(fake_imgs[i]))
        axes[i, 1].set_title("StyleGAN2-ADA")
        axes[i, 1].axis("off")

        if m_masks[i].sum() > 0:
            axes[i, 1].text(5, 15, "ANOMALY SEED", color="red", weight="bold",
                            bbox=dict(facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved: {save_path}")


# ============================================================
# 9) Main Pipeline
# ============================================================

if __name__ == "__main__":
    device = DEVICE
    print(f"Device: {device}")

    # --- Step 1: Data ---
    train_dl, test_dl = build_dataloaders(batch_size=16)

    # --- Step 2: Initialize StyleGAN2-ADA ---
    cfg = StyleGANConfig()
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    G = StyleGAN2Generator(z_dim=cfg.z_dim).to(device)
    D = StyleGAN2Discriminator().to(device)

    # --- Step 3: Train ---
    train_stylegan(G, D, train_dl, cfg, device)

    # --- Step 4: Load best checkpoint and evaluate ---
    best_path = cfg.ckpt_dir / "G_best.pth"
    if best_path.exists():
        G.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best checkpoint: {best_path}")

    results = evaluate_generator(G, test_dl, device, label="StyleGAN2-ADA")
    latency = measure_inference_latency(G, device)

    # --- Step 5: Qualitative grid ---
    plot_validation_grid(G, test_dl, device,
                         save_path=str(cfg.ckpt_dir / "validation_grid.png"))

    # --- Step 6: LDM baseline (optional — requires diffusers) ---
    # Uncomment to run LDM inference latency benchmark:
    #
    # pipe = build_ldm_pipeline(device)
    # real_batch, y_batch, m_batch = next(iter(test_dl))
    # sample_pil = Image.fromarray(
    #     ((real_batch[0].permute(1, 2, 0).numpy() + 1) / 2 * 255).astype("uint8")
    # )
    # mask_pil = Image.fromarray(
    #     (m_batch[0, 0].numpy() * 255).astype("uint8")
    # )
    # _, ldm_latency = generate_ldm(pipe, sample_pil, mask_pil)
    # print(f"LDM latency: {ldm_latency:.2f} s  "
    #       f"({ldm_latency / (latency / 1000):.0f}× slower than StyleGAN2-ADA)")
