"""Command-line entry point for SGLN-LRQMC.

Examples
--------
python main_SGLN_LRQMC.py
python main_SGLN_LRQMC.py --dataset CSet12 --image Peppers --mr 0.85
python main_SGLN_LRQMC.py --img-path C:\\data\\image.png --mask-path C:\\data\\mask.png \
    --lambda 1.256 --iterations 100 --output recovery.png
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from config import SGLNLRQMCConfig


ROOT = Path(__file__).resolve().parent


def set_deterministic_seed(seed):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False


def load_binary_mask(mask_path):
    """Load a mask and retain only pixels whose normalized value is exactly one."""
    if not mask_path.exists():
        raise FileNotFoundError(f'Mask file not found: {mask_path}')
    mask = np.asarray(Image.open(mask_path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask /= 255.0
    mask[mask < 1.0] = 0.0
    return mask


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run SGLN-LRQMC for color image completion.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    image_group = parser.add_argument_group('input image')
    image_group.add_argument('--dataset', default='CSet12',
                             help='Dataset folder relative to this script.')
    image_group.add_argument('--image', default='Airplane',
                             help='Image stem used with --dataset (without extension).')
    image_group.add_argument('--img-path', type=Path, default=None,
                             help='Explicit image path; overrides --dataset and --image.')
    image_group.add_argument('--mr', type=float, default=0.85,
                             help='Missing ratio used to infer the default mask filename.')
    image_group.add_argument('--mask-path', type=Path, default=None,
                             help='Explicit binary mask path; overrides the mask inferred from --mr.')

    parameter_group = parser.add_argument_group('SGLN-LRQMC parameters')
    parameter_group.add_argument('--lambda', dest='lam', type=float, default=1.0,
                                 help='Quaternion low-rank regularization parameter.')
    parameter_group.add_argument('--alpha', type=float, default=0.1,
                                 help='QNMF F_norm weight parameter.')
    parameter_group.add_argument('--gamma', type=float, default=20.0,
                                 help='Local PnP regularization parameter.')
    parameter_group.add_argument('--eta', type=float, default=20.0,
                                 help='Nonlocal CBM3D PnP parameter.')
    parameter_group.add_argument('--mu', type=float, default=1e-3,
                                 help='Initial shared ADMM penalty parameter.')
    parameter_group.add_argument('--rho', type=float, default=1.1,
                                 help='ADMM penalty growth factor.')
    parameter_group.add_argument('--mu-max', type=float, default=1e20,
                                 help='Maximum ADMM penalty parameter.')
    parameter_group.add_argument('--iterations', type=int, default=100,
                                 help='Maximum number of iterations.')
    parameter_group.add_argument('--tol', type=float, default=1e-3,
                                 help='Relative-change stopping tolerance; set 0 to disable.')

    run_group = parser.add_argument_group('run options')
    run_group.add_argument('--seed', type=int, default=0, help='Random seed.')
    run_group.add_argument('--local-denoiser', choices=('ffdnet', 'drunet'), default='ffdnet',
                           help='Denoiser used by the local PnP branch.')
    run_group.add_argument('--equivariant-pnp', action=argparse.BooleanOptionalAction,
                           default=True, help='Enable or disable equivariant PnP.')
    run_group.add_argument('--eval-ssim', action=argparse.BooleanOptionalAction,
                           default=True, help='Compute SSIM in addition to PSNR.')
    run_group.add_argument('--output', type=Path, default=None,
                           help='Optional path for saving the selected reconstruction PNG.')
    return parser


def resolve_paths(args):
    img_path = args.img_path or ROOT / 'Dataset' / args.dataset / f'{args.image}.png'
    mask_path = args.mask_path or ROOT / 'Dataset' / f'Mask_{int(round(100 * args.mr))}.png'
    img_path = img_path.resolve()
    mask_path = mask_path.resolve()
    if not img_path.exists():
        raise FileNotFoundError(f'Image file not found: {img_path}')
    return img_path, mask_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    from src.sgln_lrqmc import sgln_lrqmc

    img_path, mask_path = resolve_paths(args)
    set_deterministic_seed(args.seed)

    config = SGLNLRQMCConfig(
        image_path=img_path,
        observed_mask=load_binary_mask(mask_path),
        lam=args.lam,
        alpha=args.alpha,
        gamma=args.gamma,
        eta=args.eta,
        mu=args.mu,
        rho=args.rho,
        mu_max=args.mu_max,
        iterations=args.iterations,
        tolerance=args.tol,
        seed=args.seed,
        equivariant_pnp=args.equivariant_pnp,
        evaluate_ssim=args.eval_ssim,
        local_denoiser=args.local_denoiser,
    )

    print('=' * 70)
    print('PyTorch CUDA SGLN-LRQMC - Color Image Completion')
    print('=' * 70)
    print(f'Image: {img_path}')
    print(f'Mask: {mask_path} (MR setting: {args.mr:.2f})')
    print(f'Iterations: {args.iterations}, lambda: {args.lam}, alpha: {args.alpha}')
    print(f'gamma: {args.gamma}, eta: {args.eta}, mu: {args.mu}, rho: {args.rho}')
    print(f'local denoiser: {args.local_denoiser}, equivariant pnp: {args.equivariant_pnp}, seed: {args.seed}')
    print('=' * 70)

    img_recovery, psnr_max, ssim_max, *_ = sgln_lrqmc(config)
    print(f'Done. peak PSNR = {psnr_max:.4f}, peak SSIM = {ssim_max:.4f}')

    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_recovery).save(output_path)
        print(f'Reconstruction saved to: {output_path}')


if __name__ == '__main__':
    main()
