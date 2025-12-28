# DRBD-Mamba: Dual-Resolution Bi-directional Mamba for Efficient Brain Tumor Segmentation

This repository provides the official PyTorch implementation of **DRBD-Mamba**, a dual-resolution bi-directional Mamba-based architecture for 3D brain tumor segmentation from multimodal MRI scans.

The code is released to support **reproducibility, transparency, and further research**, and corresponds to the experiments reported in our journal submission to *Biomedical Signal Processing and Control*.

---

## 📌 Key Features
- Dual-resolution bi-directional Mamba blocks integrated into a 3D encoder–decoder architecture
- Morton space-filling curve (SFC) for locality-preserving voxel sequence modeling
- Vector Quantization (VQ) block for latent regularization
- Systematic five-fold cross-validation protocol for fair evaluation
- Detailed per-case evaluation with Dice and HD95 metrics
- Computational efficiency analysis (FLOPs, parameters, runtime)

---

## 📂 Repository Structure

