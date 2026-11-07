"""PyTorch CUDA implementation of the SGLN-LRQMC main method."""

import time

import numpy as np
import torch
from PIL import Image

from .cbm3d_cupy import denoise_cbm3d_torch
from config import SGLNLRQMCConfig
from utils.utils_torch import (
    compare_ssim_matlab_torch,
    compute_psnr_torch,
    denoise_drunet_torch,
    denoise_ffdnet_torch,
    equivariant_denoise_torch,
    load_drunet_model,
    load_ffdnet_model,
    qcomplex_to_rgb,
    rgb_to_qcomplex,
    update_lowrank_qcomplex,
    update_X
)


def _normalize_omega(omega, img_shape, device):
    """Return Omega as HWC with one or three channels on CUDA."""
    h, w, c = img_shape
    omega = torch.as_tensor(omega, dtype=torch.float32, device=device)
    if omega.ndim == 2:
        if tuple(omega.shape) != (h, w):
            raise ValueError(f'2D Omega shape must be {(h, w)}, got {tuple(omega.shape)}')
        return omega.unsqueeze(2)
    if omega.ndim == 3:
        if tuple(omega.shape) != (h, w, c):
            raise ValueError(f'3D Omega shape must be {(h, w, c)}, got {tuple(omega.shape)}')
        return omega
    raise ValueError(f'Omega must be 2D or 3D, got {tuple(omega.shape)}')


def sgln_lrqmc(config: SGLNLRQMCConfig):
    """Run SGLN-LRQMC with a PyTorch CUDA core and a CuPy-only CBM3D branch.

    Returns:
        img_recovery: HWC uint8 NumPy array
        psnr_max: PSNR of the final iterate
        ssim_max: SSIM of the final iterate
        psnr_list: per-iteration float32 NumPy array
        err_list: per-iteration float32 NumPy array
        time_list: cumulative per-iteration float64 NumPy array
        ssim_list: per-iteration float32 NumPy array
    """
    if not torch.cuda.is_available():
        raise RuntimeError('SGLN-LRQMC requires a CUDA-capable PyTorch environment.')

    img_path = config.image_path
    rho = config.rho
    alpha = config.alpha
    lam = config.lam
    gamma = config.gamma
    eta = config.eta
    equivariant_pnp = config.equivariant_pnp
    tol = config.tolerance
    # Match the notation in the paper: one shared ADMM penalty parameter mu.
    mu = config.mu
    k_max = config.iterations
    t = config.truncation
    seed = config.seed
    eval_ssim = config.evaluate_ssim
    cbm3d_kwargs = config.cbm3d_kwargs
    local_denoiser = config.local_denoiser

    device = torch.device('cuda:0')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    img_np = np.asarray(Image.open(img_path), dtype=np.float32)[:, :, :3].copy()
    img_real = torch.from_numpy(img_np).to(device=device)
    img_eval = img_real / 255.0
    m, n, channel = img_real.shape
    omega_mask = _normalize_omega(config.observed_mask, img_real.shape, device)
    missing_mask = 1.0 - omega_mask
    y_real = omega_mask * img_real
    y_q = rgb_to_qcomplex(y_real)

    psnr_observed = compute_psnr_torch(y_real / 255.0, img_eval)
    ssim_observed = compare_ssim_matlab_torch(y_real / 255.0, img_eval) if eval_ssim else 0.0
    print(
        'PSNR and SSIM of the observed Image is '
        f'psnr: {psnr_observed:.2f}dB | ssim: {ssim_observed:.4f}'
    )

    x = y_q.clone()
    # Paper notation: dot{Z} is the quaternion low-rank variable, while
    # S and T are the RGB local and nonlocal PnP variables, respectively.
    z_q = y_q.clone()
    s = y_real.clone()
    t_pnp = y_real.clone()
    x_real = y_real.clone()
    x_prev_real = y_real.clone()
    lambda_q = torch.zeros_like(y_q)
    u = torch.zeros((m, n, channel), dtype=torch.float32, device=device)
    v = torch.zeros((m, n, channel), dtype=torch.float32, device=device)

    if local_denoiser == 'ffdnet':
        local_net = load_ffdnet_model(device)

        def local_denoise_handle(img_noisy, sigma):
            return denoise_ffdnet_torch(img_noisy, sigma, local_net)
    else:
        local_net = load_drunet_model(device)

        def local_denoise_handle(img_noisy, sigma):
            return denoise_drunet_torch(img_noisy, sigma, local_net)

    def cbm3d_handle(img_noisy, sigma):
        return denoise_cbm3d_torch(img_noisy, sigma, **cbm3d_kwargs)

    psnr_list = torch.zeros(k_max, dtype=torch.float32, device=device)
    ssim_list = torch.zeros(k_max, dtype=torch.float32, device=device)
    err_list = torch.zeros(k_max, dtype=torch.float32, device=device)
    time_list = np.zeros(k_max, dtype=np.float64)
    psnr_max = 0.0
    ssim_max = 0.0

    mode_generator = torch.Generator(device='cpu').manual_seed(seed)

    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    for k in range(k_max):
        b = x + lambda_q / mu
        z_q = update_lowrank_qcomplex(b, lam, alpha, mu, t)
        e = z_q - lambda_q / mu
        m_local = s - u / mu
        n_nonlocal = t_pnp - v / mu
        x_real = update_X(e, m_local, n_nonlocal, mu, y_real, missing_mask)
        x = rgb_to_qcomplex(x_real)

        sigma_s = np.sqrt(gamma / mu)
        sigma_t = np.sqrt(eta / mu)
        local_input = x_real + u / mu
        cbm3d_input = x_real + v / mu
        s = equivariant_denoise_torch(
            local_input,
            sigma_s,
            local_denoise_handle,
            equivariant_pnp,
            mode=int(torch.randint(1, 9, (), generator=mode_generator).item()),
        )
        t_pnp = equivariant_denoise_torch(
            cbm3d_input,
            sigma_t,
            cbm3d_handle,
            equivariant_pnp,
            mode=int(torch.randint(1, 9, (), generator=mode_generator).item()),
        )

        lambda_q.add_(mu * (x - z_q))
        u.add_(mu * (x_real - s))
        v.add_(mu * (x_real - t_pnp))
        mu = rho * mu

        x_eval = missing_mask * torch.clamp(x_real / 255.0, 0.0, 1.0) + y_real/255
        if k > 0 and tol>0:
            diff = x_real - x_prev_real
            rel_change = torch.linalg.vector_norm(diff) / (
                1e-7 + torch.linalg.vector_norm(x_prev_real)
            )
            if rel_change > tol:
                err_list[k] = rel_change
            else:
                break
        psnr_test = compute_psnr_torch(x_eval, img_eval)
        psnr_list[k] = psnr_test
        ssim_test = compare_ssim_matlab_torch(x_eval, img_eval) if eval_ssim else 0.0
        ssim_list[k] = ssim_test
        psnr_max = psnr_test
        ssim_max = ssim_test
        time_list[k] = time.perf_counter() - t_start

        if (k + 1) % 20 == 0:
            print(
                f'Iter {k + 1:3d} | psnr: {psnr_test:.4f} | '
                f'ssim: {ssim_test:.4f}'
            )
        x_prev_real.copy_(x_real)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t_start
    print(f'Total Running Time: {elapsed:.2f} seconds')

    img_recovery = torch.clamp(x_real, 0.0, 255.0).round().to(torch.uint8)
    return (
        img_recovery.cpu().numpy(),
        psnr_max,
        ssim_max,
        psnr_list.cpu().numpy(),
        err_list.cpu().numpy(),
        time_list,
        ssim_list.cpu().numpy(),
    )
