

## ✅ `README.md`

```markdown
# DDPM-DIV2K

This repository contains the implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on the DIV2K dataset for high-quality image generation.

---

## 📌 Overview

DDPM is a generative model that learns to reverse a diffusion process that gradually adds noise to the data. This project adapts DDPM architecture using a custom lightweight UNet and trains it on the high-resolution DIV2K dataset.

---

## 📂 Directory Structure

.
├── Checkpoints/             # Stores model weights (.pt files)
├── Diffusion/
│   ├── Train.py             # Training loop
│   ├── Unet.py              # Custom UNet architecture
│   ├── Sampler.py           # Diffusion sampling code
│   └── div2k_dataloader.py  # Custom dataset loader
├── SampledImgs/             # Generated samples from the model
├── data/                    # Contains DIV2K training images
├── Main.py                  # Main script to initiate training and sampling
├── evaluation/              # Scripts to calculate PSNR, SSIM, LPIPS, FID, IS, KID
└── README.md                # Project documentation
```

---

## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/nikhilchatta/ddpm-div2k.git
   cd ddpm-div2k
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Git LFS (if not already installed)**
   ```bash
   git lfs install
   ```

---

## 📦 Training

Run the training script:
```bash
python Main.py
```

Update your training configurations from `modelConfig` dictionary inside `Main.py`.

---

## 🖼 Sampling

After training, generate images:
```python
generate_one_batch(
    model_path="./Checkpoints/final_model.pt",
    save_dir="./SampledImgs/batch_1",
    batch_size=64,
    image_size=128
)
```

---

## 📊 Evaluation Metrics

The following image quality metrics have been implemented:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MS-SSIM (Multi-Scale SSIM)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Perceptual Similarity (VGG Feature Distance)
- FID (Fréchet Inception Distance)
- IS (Inception Score)
- KID (Kernel Inception Distance)

All metrics are evaluated over 20 rounds and averaged.

---

## ⚙ Tools & Frameworks

- Python
- PyTorch
- torchvision
- scikit-image
- scipy
- pytorch-msssim
- lpips
- Git LFS

---

## 📌 Note

Some files such as sample images and data may be large. Git LFS is used for storing large files. Ensure `git lfs pull` is used after cloning.

---

## 🧠 Acknowledgements

Inspired by the original DDPM and UNet implementations. DIV2K dataset © CVPR NTIRE challenge.

---

## 📬 Contact

For questions or collaborations, reach out at: [nikhilchatta@domain.com]
```
