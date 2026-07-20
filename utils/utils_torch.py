"""PyTorch CUDA utilities for the SGLN-LRQMC main method."""

from pathlib import Path

import torch
import torch.nn.functional as F


def rgb_to_qcomplex(img):
    """Map an HWC RGB tensor to its 2H x 2W complex quaternion representation."""
    img = torch.as_tensor(img, dtype=torch.float32, device=img.device)
    r, g, b = img.unbind(dim=2)
    a_block = torch.complex(torch.zeros_like(r), r)
    b_block = torch.complex(g, b)
    top = torch.cat((a_block, b_block), dim=1)
    bottom = torch.cat((-torch.conj(b_block), torch.conj(a_block)), dim=1)
    return torch.cat((top, bottom), dim=0).to(torch.complex64)


def qcomplex_to_rgb(qmat):
    """Recover an HWC RGB tensor from a structured complex matrix."""
    h = qmat.shape[0] // 2
    w = qmat.shape[1] // 2
    a_block = qmat[:h, :w]
    b_block = qmat[:h, w:]
    return torch.stack((a_block.imag, b_block.real, b_block.imag), dim=2).float()


def zeros_qcomplex(h, w, device):
    return torch.zeros((2 * h, 2 * w), dtype=torch.complex64, device=device)


def project_qcomplex_structure(qmat):
    """Project a complex matrix back to quaternion block structure."""
    h = qmat.shape[0] // 2
    w = qmat.shape[1] // 2
    a1 = qmat[:h, :w]
    b1 = qmat[:h, w:]
    a2 = torch.conj(qmat[h:, w:])
    b2 = -torch.conj(qmat[h:, :w])
    a_block = 0.5 * (a1 + a2)
    b_block = 0.5 * (b1 + b2)
    top = torch.cat((a_block, b_block), dim=1)
    bottom = torch.cat((-torch.conj(b_block), torch.conj(a_block)), dim=1)
    return torch.cat((top, bottom), dim=0).to(torch.complex64)


def closed_tl12_torch(sigma_y, alpha, coeff, t=0):
    """Apply the TL12 singular-value proximal update on CUDA."""
    threshold = torch.as_tensor(coeff, dtype=sigma_y.dtype, device=sigma_y.device)
    alpha_t = torch.as_tensor(alpha, dtype=sigma_y.dtype, device=sigma_y.device)
    z = torch.clamp(sigma_y[t:] - threshold, min=0.0)
    norm_2 = torch.linalg.vector_norm(z) + torch.finfo(sigma_y.dtype).eps
    scale = 1.0 + alpha_t * threshold / norm_2
    sigma_z = torch.empty_like(sigma_y)
    sigma_z[:t] = sigma_y[:t]
    sigma_z[t:] = scale * z
    sigma_z = torch.minimum(sigma_z, sigma_y)
    svp = int(torch.count_nonzero(z > 0).item()) + t
    return sigma_z[:svp], svp


def update_lowrank_qcomplex(qmat, lam, alpha, beta, t):
    """Update the low-rank variable with PyTorch's CUDA complex SVD."""
    h = qmat.shape[0] // 2
    w = qmat.shape[1] // 2
    u, s_all, vh = torch.linalg.svd(qmat, full_matrices=False)
    s_q = s_all[::2][:min(h, w)].float()
    s_thr, svp = closed_tl12_torch(s_q, alpha, lam / beta, t)
    if svp <= 0:
        return zeros_qcomplex(h, w, qmat.device)
    rank_c = min(2 * svp, s_all.numel())
    s_pair = torch.repeat_interleave(s_thr, 2)[:rank_c].float()
    lowrank = (u[:, :rank_c] * s_pair.unsqueeze(0)) @ vh[:rank_c, :]
    return project_qcomplex_structure(lowrank)


def compute_psnr_torch(a, b):
    """Compute PSNR on CUDA for images in [0, 1]."""
    a = a.float()
    b = b.float()
    mse = torch.mean((a - b).square())
    if mse.item() == 0.0:
        return float('inf')
    return float((10.0 * torch.log10(1.0 / mse)).item())


def _gaussian_kernel(device, dtype, size=11, sigma=1.5):
    radius = size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords.square()) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _convolve_axis_nearest(img, kernel, axis):
    """Match cupyx.scipy.ndimage.convolve1d(..., mode='nearest')."""
    order = [dim for dim in range(img.ndim) if dim != axis] + [axis]
    inverse = [order.index(dim) for dim in range(img.ndim)]
    moved = img.permute(order).contiguous()
    moved_shape = moved.shape
    lines = moved.reshape(-1, 1, moved_shape[-1])
    radius = kernel.numel() // 2
    lines = F.pad(lines, (radius, radius), mode='replicate')
    filtered = F.conv1d(lines, kernel.reshape(1, 1, -1))
    return filtered.reshape(moved_shape).permute(inverse).contiguous()


def _gaussian_filter_torch(img, sigma=1.5, filt_size=11):
    kernel = _gaussian_kernel(img.device, torch.float32, filt_size, sigma)
    out = img.float()
    for axis in range(img.ndim):
        out = _convolve_axis_nearest(out, kernel, axis)
    return out


def compare_ssim_matlab_torch(img1, img2, dynamic_range=1.0):
    """Compute the existing three-axis MATLAB-style SSIM definition on CUDA."""
    if img1.shape != img2.shape:
        raise ValueError('img1 and img2 must have the same shape')
    if img1.ndim not in (2, 3):
        raise ValueError('compare_ssim_matlab_torch expects a 2D or 3D image')
    img1 = img1.float()
    img2 = img2.float()
    c1 = torch.as_tensor((0.01 * dynamic_range) ** 2, device=img1.device, dtype=torch.float32)
    c2 = torch.as_tensor((0.03 * dynamic_range) ** 2, device=img1.device, dtype=torch.float32)
    mu1 = _gaussian_filter_torch(img1)
    mu2 = _gaussian_filter_torch(img2)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _gaussian_filter_torch(img1.square()) - mu1_sq
    sigma2_sq = _gaussian_filter_torch(img2.square()) - mu2_sq
    sigma12 = _gaussian_filter_torch(img1 * img2) - mu1_mu2
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(torch.mean(numerator / denominator).item())


def apply_d4_transform_torch(img, mode):
    if mode == 1:
        return img
    if mode == 2:
        return torch.flip(img, (0,))
    if mode == 3:
        return torch.rot90(img, 1, (0, 1))
    if mode == 4:
        return torch.flip(torch.rot90(img, 1, (0, 1)), (0,))
    if mode == 5:
        return torch.rot90(img, 2, (0, 1))
    if mode == 6:
        return torch.flip(torch.rot90(img, 2, (0, 1)), (0,))
    if mode == 7:
        return torch.rot90(img, 3, (0, 1))
    if mode == 8:
        return torch.flip(torch.rot90(img, 3, (0, 1)), (0,))
    raise ValueError(f'Unsupported D4 transform mode: {mode}')


def invert_d4_transform_torch(img, mode):
    if mode == 1:
        return img
    if mode == 2:
        return torch.flip(img, (0,))
    if mode == 3:
        return torch.rot90(img, 3, (0, 1))
    if mode == 4:
        return torch.rot90(torch.flip(img, (0,)), 3, (0, 1))
    if mode == 5:
        return torch.rot90(img, 2, (0, 1))
    if mode == 6:
        return torch.rot90(torch.flip(img, (0,)), 2, (0, 1))
    if mode == 7:
        return torch.rot90(img, 1, (0, 1))
    if mode == 8:
        return torch.rot90(torch.flip(img, (0,)), 1, (0, 1))
    raise ValueError(f'Unsupported D4 transform mode: {mode}')


def equivariant_denoise_torch(img, sigma, denoise_handle, equivariant_pnp, mode=1):
    if not equivariant_pnp:
        mode = 1
    else:
        sigma = max(sigma, 10) # Prevent the strength from being too low when using a random transform according to DPIR.
    transformed = apply_d4_transform_torch(img, mode)
    denoised = denoise_handle(transformed, sigma)
    return invert_d4_transform_torch(denoised, mode).contiguous()


_FFDNET_MODEL_CACHE = {}
_DRUNET_MODEL_CACHE = {}


def load_ffdnet_model(device):
    """Load and cache the single-GPU FFDNet model."""
    from FFDNet_pytorch.MODEL import FFDNet

    device = torch.device(device)
    cache_key = str(device)
    if cache_key in _FFDNET_MODEL_CACHE:
        return _FFDNET_MODEL_CACHE[cache_key]
    model_path = Path(__file__).resolve().parents[1] / 'FFDNet_pytorch' / 'models' / 'net_rgb.pth'
    net = FFDNet(num_input_channels=3)
    state_dict = torch.load(str(model_path), map_location=device)
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    net.load_state_dict(state_dict)
    net = net.to(device).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    _FFDNET_MODEL_CACHE[cache_key] = net
    return net


def denoise_ffdnet_torch(img_noisy, sigma, net):
    """Run FFDNet directly on an HWC PyTorch CUDA tensor in [0, 255]."""
    img_tensor = img_noisy.permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0)
    noise_map = torch.full((1, 1, 1, 1), float(sigma) / 255.0, device=img_tensor.device)
    with torch.inference_mode():
        noise = net(x=img_tensor, noise_sigma=noise_map)
        denoised = (img_tensor - noise).clamp_(0.0, 1.0).mul_(255.0).squeeze(0)
    return denoised.permute(1, 2, 0).contiguous().float()


def load_drunet_model(device, model_name='drunet_color'):
    """Load and cache the color DRUNet model used by the local PnP branch."""
    from DRUNet_pytorch.models.network_unet import UNetRes

    if model_name != 'drunet_color':
        raise ValueError(f'Only the RGB DRUNet model is supported, got: {model_name}')
    device = torch.device(device)
    cache_key = (str(device), model_name)
    if cache_key in _DRUNET_MODEL_CACHE:
        return _DRUNET_MODEL_CACHE[cache_key]

    model_path = Path(__file__).resolve().parents[1] / 'DRUNet_pytorch' / 'model_zoo' / f'{model_name}.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'DRUNet model file not found: {model_path}')
    net = UNetRes(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R',
        downsample_mode='strideconv',
        upsample_mode='convtranspose',
    )
    state_dict = torch.load(str(model_path), map_location=device)
    net.load_state_dict(state_dict, strict=True)
    net = net.to(device).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    _DRUNET_MODEL_CACHE[cache_key] = net
    return net


def denoise_drunet_torch(img_noisy, sigma, net):
    """Run color DRUNet on an HWC CUDA tensor in [0, 255]."""
    img_tensor = img_noisy.permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0)
    _, _, height, width = img_tensor.shape
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    if pad_h or pad_w:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
    noise_map = torch.full(
        (1, 1, img_tensor.shape[2], img_tensor.shape[3]),
        float(sigma) / 255.0,
        dtype=img_tensor.dtype,
        device=img_tensor.device,
    )
    with torch.inference_mode():
        denoised = net(torch.cat((img_tensor, noise_map), dim=1))
        denoised = denoised[:, :, :height, :width].clamp_(0.0, 1.0).mul_(255.0).squeeze(0)
    return denoised.permute(1, 2, 0).contiguous().float()
