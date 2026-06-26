# SGLN-LRQMC

This repository is the official code repository for **SGLN-LRQMC**, a stabilized global-local-nonlocal low-rank quaternion matrix completion framework for color image inpainting.

The source code will be released after the paper is accepted. The current repository is prepared as a public landing page for the method, including the method summary, expected usage, experimental settings, and citation information.

## Overview

Low-rank quaternion matrix completion (LRQMC) is effective for color image inpainting because it encodes the RGB channels as a pure quaternion matrix and preserves cross-channel correlations. However, a single global low-rank prior is often insufficient for recovering complex local structures and repeated nonlocal textures, especially under high missing ratios.

SGLN-LRQMC addresses this issue by integrating three complementary priors within a unified ADMM-based optimization framework:

- **Global prior:** QNMF-based quaternion low-rank regularization for global structure and RGB-channel coupling.
- **Local prior:** FFDNet-based plug-and-play denoising for local details and fine-scale structures.
- **Nonlocal prior:** CBM3D-based plug-and-play denoising for nonlocal self-similarity and repeated texture patterns.
- **Stabilization:** Equivariant plug-and-play updates are embedded into the denoising subproblems to improve iterative stability.

Unless otherwise specified, **SGLN-LRQMC** refers to the default implementation using **QNMF + FFDNet + CBM3D + Equivariant PnP**.

## Code Availability

The implementation is currently under preparation for public release.

After acceptance, this repository will provide:

- The full implementation of SGLN-LRQMC.
- Scripts for reproducing the main random-mask experiments.
- Scripts for structured-mask experiments, including text, scratch, and block masks.
- Parameter settings used in the paper.
- Evaluation code for PSNR and SSIM.
- Example commands for running ablation studies and denoiser-replacement experiments.

## Expected Repository Structure

The released version is expected to follow a structure similar to:

```text
SGLN-LRQMC/
|-- README.md
|-- main_random_mask.*
|-- main_structured_mask.*
|-- main_ablation.*
|-- configs/
|   |-- default.*
|   `-- ablation.*
|-- data/
|   `-- CSet12/
|-- denoisers/
|   |-- FFDNet/
|   |-- CBM3D/
|   `-- DRUNet/
|-- masks/
|   |-- random/
|   `-- structured/
|-- metrics/
|-- utils/
`-- results/
```

The exact file names may be adjusted before release.

## Experimental Settings

The experiments in the paper evaluate SGLN-LRQMC on color image inpainting tasks.

### Dataset

The main experiments use ten representative color images of size `256 x 256` from the CSet12 benchmark.

### Missing Patterns

The method is evaluated under:

- Random masks with missing ratios `0.50`, `0.65`, `0.75`, and `0.85`.
- Structured masks, including text, scratch, and block masks.

### Evaluation Metrics

The reconstruction quality is evaluated using:

- Peak signal-to-noise ratio (PSNR)
- Structural similarity index (SSIM)

### Default Parameters

The main experiments use the following default settings:

```text
normalized low-rank parameter:  lambda_tilde = 0.01
QNMF balance parameter:         alpha = 0.1
local PnP parameter:            gamma in [20, 100]
nonlocal PnP parameter:         eta in [20, 100]
initial ADMM penalty:           mu0 = 1e-3
ADMM continuation factor:       rho = 1.1
stopping tolerance:             tol = 1e-3
```

The regularization parameter is defined as:

```text
lambda = lambda_tilde * sqrt(mn)
```

where `m x n` is the spatial size of the input image.

## Method Components

The default SGLN-LRQMC implementation contains the following components.

| Component | Role | Default choice |
| --- | --- | --- |
| Global low-rank branch | Captures global image structures and RGB-channel correlations | QNMF |
| Local PnP branch | Preserves edges and fine-scale local details | FFDNet |
| Nonlocal PnP branch | Exploits repeated textures and nonlocal self-similarity | CBM3D |
| Stabilization strategy | Improves iterative stability of PnP updates | Equivariant PnP |

The framework also supports replacing the local denoising branch with another deep CNN denoiser. In the paper, a DRUNet-based variant is evaluated to examine this flexibility.

## Usage

The runnable code and exact commands will be added after acceptance.

The released version will include examples such as:

```bash
# Random-mask color image inpainting
python main_random_mask.py --image Airplane --mr 0.75 --config configs/default.yaml

# Structured-mask color image inpainting
python main_structured_mask.py --image Peppers --mask scratch --config configs/default.yaml

# Ablation study
python main_ablation.py --image Airplane --mr 0.85 --config configs/ablation.yaml
```

These commands are placeholders and will be updated to match the final released implementation.

## Citation

If this work is useful for your research, please cite the paper. The BibTeX entry will be updated after publication.

```bibtex
@article{SGLN_LRQMC,
  title   = {SGLN-LRQMC: Stabilized Global-Local-Nonlocal Low-Rank Quaternion Matrix Completion for Color Image Inpainting},
  author  = {To be updated},
  journal = {To be updated},
  year    = {To be updated}
}
```

## License

The license will be specified when the source code is released.

## Contact

For questions about the paper or future code release, please open an issue in this repository.
