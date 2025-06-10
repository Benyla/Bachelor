# Enhanced Variational Autoencoder

**Author**: Bertram Nyvold Larsen  
**Supervisor**: Bjørn Sand Jensen  
**Degree**: Bachelor of Artificial Intelligence and Data, DTU  
**Date**: June 2025  

---

## 📘 Overview

This repository contains the full implementation and evaluation of **VAE+**, an enhanced Variational Autoencoder architecture with adversarial feature matching, trained on the **BBBC021** single-cell biomedical dataset. The project compares the performance of VAE+ to a standard VAE across various metrics such as reconstruction quality, FID score, and latent space structure.

The project investigates:

- Improvements in reconstruction quality using adversarial feedback  
- Effects of latent dimensionality (128 vs 256)  
- Biological interpretability and organization of the latent space  
- Evaluation through classifiers and interpolation in latent space  

---

## 🧠 Main Concepts

- **Variational Autoencoder (VAE)**: Probabilistic generative model for learning latent representations  
- **VAE+**: An improved variant that uses a discriminator network for perceptual feature matching  
- **BBBC021 Dataset**: Single-cell images labeled by Mechanism of Action (MOA), preprocessed into (3, 64, 64) tensors  
- **Latent Space Analysis**: Interpolation, t-SNE visualizations, and classification experiments on the learned embeddings  

---

## 📊 Results

| Metric                | VAE (128) | VAE (256) | VAE+ (128) | VAE+ (256) |
|-----------------------|-----------|-----------|------------|------------|
| Reconstruction Loss   | 55        | 56        | 33         | 30         |
| FID Score             | 126.98    | 125.72    | 70.08      | 59.09      |

VAE+ significantly improves visual and perceptual reconstruction quality compared to the standard VAE. Increasing latent dimensionality (128 → 256) shows modest improvements in both reconstruction loss and FID score.

---

## 🔍 Analysis Tools

The `experiments/` folder contains tools for visual and quantitative analysis:

- `plot_recons.py` — Visualizes input vs reconstructed images
- `latent_traversal.py` — Explores how individual latent dimensions affect image outputs
- `t-SNE.py` — Reduces latent space to 2D for visual clustering
- `KNN_and_NN_classifier.py` — Trains simple classifiers on latent vectors to evaluate structure
- `compute_fid.py` — Computes Fréchet Inception Distance (FID) between real and generated images

All plots and results can be found in `experiments/plots/`.

---

## 🧬 Dataset

The project uses the **BBBC021** dataset (Broad Bioimage Benchmark Collection), which contains:

- ~490,000 single-cell RGB fluorescence images of breast cancer cells
- Images preprocessed using `data/process_data.py`

**Preprocessing steps:**

1. Channel-wise normalization  
2. Resizing to 64×64  
3. Conversion to PyTorch `.pt` format  
4. Metadata linkage retained via filenames  
5. Data split into train (70%), validation (15%), and test (15%)

---

## 🙏 Acknowledgements

- **Supervisor**: Bjørn Sand Jensen  
- **Dataset**: [BBBC021 from the Broad Institute](https://bbbc.broadinstitute.org/BBBC021)  
- **Core Model Inspiration**:  
  - Maxime W. Lafarge et al. – _Capturing Single-Cell Phenotypic Variation via Unsupervised Representation Learning_  

This project was completed as part of the BSc in Artificial Intelligence and Data at DTU.

---
