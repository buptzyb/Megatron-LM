# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused mHC operations for the per-op GLM path (mhc_fused_ops).

The fixed-launch Triton Sinkhorn: fused_mhc_kernels' Sinkhorn kernels launched with a fixed
num_warps instead of their timing autotune, so every process runs the same compiled kernel.
The kernels are looked up through fused_mhc_kernels at call time: they exist only when Triton
is available, so not under MHC_FORCE_BACKEND=native, MHC_FORCE_BACKEND=cutile or
MHC_DISABLE_TRITON=1.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.fusions import fused_mhc_kernels as fmk

SINKHORN_NUM_WARPS = 1
"""Warps per program of both Sinkhorn kernels; the per-rank fused-ops line prints this value."""


def _sinkhorn_kernels():
    """Return the forward and backward JIT functions under fused_mhc_kernels' autotuners."""
    if not fmk.is_triton_available():
        raise RuntimeError("The fixed-launch mHC Sinkhorn needs Triton, which is unavailable.")
    return fmk._triton_sinkhorn_fwd_kernel.fn, fmk._triton_sinkhorn_bwd_kernel.fn


def _require_cuda(tensor: Tensor) -> None:
    """Fail closed on a non-CUDA tensor instead of falling back to the native Sinkhorn."""
    if not tensor.is_cuda:
        raise RuntimeError(
            f"The fixed-launch mHC Sinkhorn needs CUDA tensors, got device {tensor.device}. "
            "Drop 'sinkhorn' from mhc_fused_ops or set mhc_fused_backend='native'."
        )


def _sinkhorn_fwd(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-6, *, num_warps: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """fused_mhc_kernels' _triton_sinkhorn_fwd with a fixed-warp launch; returns (out, M_init)."""
    _require_cuda(input_logits)
    fwd_kernel, _ = _sinkhorn_kernels()
    if num_warps is None:
        num_warps = SINKHORN_NUM_WARPS  # read at launch time, never bound at definition
    original_shape = input_logits.shape
    hc = original_shape[-1]
    N_batch = input_logits.numel() // (hc * hc)
    dev = input_logits.device
    out = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    M_init = torch.empty(N_batch, hc, hc, dtype=input_logits.dtype, device=dev)
    inp = input_logits.contiguous().view(N_batch, hc, hc)
    fwd_kernel[(N_batch,)](inp, out, M_init, N_batch, eps, hc, num_iterations, num_warps=num_warps)
    return out.view(original_shape), M_init.view(original_shape)


def _sinkhorn_bwd(
    grad_output: Tensor,
    M_init: Tensor,
    num_iterations: int,
    eps: float = 1e-6,
    *,
    num_warps: Optional[int] = None,
) -> Tensor:
    """fused_mhc_kernels' _triton_sinkhorn_bwd with a fixed-warp launch; fp32 workspaces."""
    _require_cuda(grad_output)
    _, bwd_kernel = _sinkhorn_kernels()
    if num_warps is None:
        num_warps = SINKHORN_NUM_WARPS  # read at launch time, never bound at definition
    original_shape = grad_output.shape
    hc = original_shape[-1]
    N_batch = grad_output.numel() // (hc * hc)
    dev = grad_output.device
    grad_input = torch.empty(N_batch, hc, hc, dtype=grad_output.dtype, device=dev)
    go = grad_output.contiguous().view(N_batch, hc, hc)
    mi = M_init.contiguous().view(N_batch, hc, hc)
    ws_M = torch.empty(N_batch * 2 * num_iterations * hc * hc, dtype=torch.float32, device=dev)
    ws_rs = torch.empty(N_batch * num_iterations * hc, dtype=torch.float32, device=dev)
    ws_cs = torch.empty(N_batch * num_iterations * hc, dtype=torch.float32, device=dev)
    bwd_kernel[(N_batch,)](
        go,
        mi,
        grad_input,
        ws_M,
        ws_rs,
        ws_cs,
        N_batch,
        eps,
        hc,
        num_iterations,
        num_warps=num_warps,
    )
    return grad_input.view(original_shape)


class FixedLaunchSinkhorn(torch.autograd.Function):
    """Sinkhorn with fused_mhc_kernels' Triton kernels at SINKHORN_NUM_WARPS warps.

    Saves M_init for the backward, as fused_mhc_kernels' TritonFusedSinkhorn does. It has no
    deterministic-mode branch: one fixed launch serves both modes.
    """

    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        """Run the Sinkhorn forward and save the initial matrix for the backward."""
        out, M_init = _sinkhorn_fwd(input_logits, num_iterations, eps)
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Run the Sinkhorn backward from the saved initial matrix."""
        (M_init,) = ctx.saved_tensors
        # Through the module global, so that a test can count the backward launches.
        grad_input = _sinkhorn_bwd(grad_output, M_init, ctx.num_iterations, ctx.eps)
        return grad_input, None, None


def fixed_launch_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Apply the fixed-launch Triton Sinkhorn with autograd support.

    The per-op ``sinkhorn`` operation of HyperConnectionModule; it takes the arguments of
    hyper_connection.native_sinkhorn.
    """
    return FixedLaunchSinkhorn.apply(input_logits, num_iterations, eps)
