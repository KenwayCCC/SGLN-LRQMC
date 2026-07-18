"""The only CuPy-dependent component: CBM3D with PyTorch DLPack bridging."""

from functools import lru_cache

import cupy as cp
import torch

import bm3d_cupy


_A_OPP = cp.asarray([
    [1 / 3, 1 / 3, 1 / 3],
    [0.5, 0, -0.5],
    [0.25, -0.5, 0.25],
], dtype=cp.float32)

_B_OPP = cp.asarray([
    [1, 1, 2 / 3],
    [1, 0, -4 / 3],
    [1, -1, 2 / 3],
], dtype=cp.float32)

_A_ROW_NORM2 = cp.sum(_A_OPP ** 2, axis=1)


_ORIGINAL_GET_TRANSFORM_MATRICES = bm3d_cupy.get_transform_matrices


@lru_cache(maxsize=None)
def _cached_transform_matrices(n, transform_name, device_id):
    """Build fixed BM3D transform matrices once per CUDA device."""
    with cp.cuda.Device(device_id):
        return _ORIGINAL_GET_TRANSFORM_MATRICES(n, transform_name)


def _get_cached_transform_matrices(n, transform_name):
    return _cached_transform_matrices(
        int(n),
        str(transform_name).lower(),
        cp.cuda.Device().id,
    )


# bm3d_gpu resolves this module-level function on every call. Replacing it with
# a device-aware cache avoids rebuilding identical wavelet/DCT/Haar matrices at
# every SGLN-LRQMC iteration without changing any BM3D numerical operation.
bm3d_cupy.get_transform_matrices = _get_cached_transform_matrices


def _rgb_to_opp_cupy(img):
    h, w, _ = img.shape
    opp = img.reshape(-1, 3) @ _A_OPP.T
    opp_max = cp.max(opp, axis=0)
    opp_min = cp.min(opp, axis=0)
    scale = opp_max - opp_min + cp.float32(1e-16)
    opp_norm = (opp - opp_min) / scale
    sigma_scale = cp.sqrt(_A_ROW_NORM2) / scale
    return opp_norm.reshape(h, w, 3), opp_max, opp_min, sigma_scale


def _opp_to_rgb_cupy(opp_norm, opp_max, opp_min):
    h, w, _ = opp_norm.shape
    opp = opp_norm.reshape(-1, 3) * (opp_max - opp_min) + opp_min
    return (opp @ _B_OPP.T).reshape(h, w, 3)


def denoise_cbm3d_torch(img_noisy, sigma, **kwargs):
    """Run CuPy CBM3D and return a PyTorch CUDA tensor without CPU copies."""
    params = {
        'patch_size': 8,
        'ht_group_size': 16,
        'wiener_group_size': 32,
        'search_radius': 19,
        'search_step': 1,
        'ref_stride': 3,
        'hard_threshold': 2.7,
        'wiener_mu2': 0.4,
        'chunk_size': 1024,
    }
    params.update(kwargs)

    torch_input = img_noisy.contiguous().float()
    cupy_input = cp.from_dlpack(torch_input)
    opp, opp_max, opp_min, sigma_scale = _rgb_to_opp_cupy(cupy_input)
    sigma_opp = cp.float32(sigma) * sigma_scale
    denoised_opp = bm3d_cupy.bm3d_gpu(opp, sigma_opp, **params)
    cupy_output = _opp_to_rgb_cupy(
        cp.asarray(denoised_opp, dtype=cp.float32),
        opp_max,
        opp_min,
    ).astype(cp.float32, copy=False)
    return torch.utils.dlpack.from_dlpack(cupy_output).contiguous()
