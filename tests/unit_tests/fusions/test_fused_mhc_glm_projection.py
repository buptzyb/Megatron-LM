# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The fixed-launch Triton Sinkhorn that per-op ``sinkhorn`` binds (fused_mhc_glm_projection).

The CPU tests pin the warp constant, the signatures and the fail-closed checks. The GPU tests
check the accuracy against float64, the bf16 path, the num_warps that each launch applies, a NaN
smoke test at the production shape under torch's deterministic memory fill, and the shared
backward kernel's barrier at n = 8. CI runs every test on several ranks at once, each on its
own GPU.
"""

import contextlib
import inspect
import os

import pytest
import torch

from megatron.core.fusions import fused_mhc_glm_projection as glm
from megatron.core.fusions import fused_mhc_kernels as fmk
from tests.unit_tests.transformer.glm_mhc_emulator import rel_rms, sinkhorn_f64

_NEEDS_CUDA_TRITON = pytest.mark.skipif(
    not (torch.cuda.is_available() and fmk.is_triton_available()), reason="needs CUDA and Triton"
)

# The production shape (T = 4096 tokens, n = 4) and layouts: HyperConnectionModule passes the
# logits as a view at offset 8 into the [T, 1, 24] mapping projection, and the upstream
# gradient arrives transposed.
T, N, ITERS, EPS = 4096, 4, 20, 1e-6
_SHAPE = (T, 1, N, N)
_LOGITS_LAYOUT = ((24, 16, 4, 1), 8, T * 24)  # strides, storage offset, storage size
_GRAD_LAYOUT = ((16, 4, 1, 4), 0, T * 16)
# rel-RMS against float64, for the output and the logits gradient.
REL_RMS_FLOOR = 1e-5


@pytest.fixture(autouse=True)
def _bind_local_cuda_device():
    """CI torchrun workers are independent replicas, each on its own local GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


@contextlib.contextmanager
def _det_fill():
    """torch's deterministic mode with fill_uninitialized_memory: every torch.empty is NaN.

    Both settings are restored on exit.
    """
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = was_fill


def _production_inputs(logits_seed, grad_seed, scale=1.0):
    """fp32 logits (a leaf) and a bf16-rounded upstream gradient, in the production layouts."""
    logits = torch.randn(_SHAPE, generator=torch.Generator().manual_seed(logits_seed)) * scale
    grad = torch.randn(_SHAPE, generator=torch.Generator().manual_seed(grad_seed))
    grad = grad.bfloat16().float()  # production's gradient is the fp32 upcast of a bf16 one

    def place(values, layout):
        stride, offset, numel = layout
        view = torch.zeros(numel, device="cuda").as_strided(_SHAPE, stride, offset)
        return view.copy_(values.cuda()).detach()

    return place(logits, _LOGITS_LAYOUT).requires_grad_(), place(grad, _GRAD_LAYOUT)


def _float64_reference(logits, grad, iterations=ITERS):
    """The Sinkhorn output and logits gradient in float64 on CPU, in hyper_connection's order."""
    x64 = logits.detach().cpu().double().requires_grad_()
    out64 = sinkhorn_f64(x64, iterations, EPS)
    (grad64,) = torch.autograd.grad(out64, x64, grad.detach().cpu().double())
    return out64.detach(), grad64


def _record_launches(monkeypatch):
    """Record (kernel, num_warps) for every launch of the two shared Sinkhorn JIT functions.

    The patch point is the JIT function's run method, which fused_mhc_kernels' deterministic
    Sinkhorn launch test patches too.
    """
    launches = []
    for name, tuner in (
        ("fwd", fmk._triton_sinkhorn_fwd_kernel),
        ("bwd", fmk._triton_sinkhorn_bwd_kernel),
    ):

        def record(*args, _name=name, _original=tuner.fn.run, **kwargs):
            launches.append((_name, kwargs.get("num_warps")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(tuner.fn, "run", record)
    return launches


# --- CPU ------------------------------------------------------------------------------------


def test_constants_and_signature():
    """The warp constant, the public signature and the helpers' keyword-only num_warps."""
    assert glm.SINKHORN_NUM_WARPS == 1
    parameters = inspect.signature(glm.fixed_launch_sinkhorn).parameters
    assert list(parameters) == ["input_logits", "num_iterations", "eps"]
    assert parameters["eps"].default == 1e-6
    # The helpers resolve num_warps at each launch. A default bound when the function is
    # defined would not follow a changed SINKHORN_NUM_WARPS, while the per-rank line would.
    for helper in (glm._sinkhorn_fwd, glm._sinkhorn_bwd):
        num_warps = inspect.signature(helper).parameters["num_warps"]
        assert num_warps.kind is inspect.Parameter.KEYWORD_ONLY
        assert num_warps.default is None


def test_kernel_lookup_requires_triton(monkeypatch):
    """Without Triton the kernel lookup raises instead of returning missing kernels."""
    monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs Triton"):
        glm._sinkhorn_kernels()


def test_rejects_non_cuda_input(monkeypatch):
    """A CPU tensor fails closed instead of running the native Sinkhorn, with or without Triton."""
    logits = torch.zeros(2, 1, N, N, requires_grad=True)
    for available in (True, False):
        monkeypatch.setattr(fmk, "is_triton_available", lambda available=available: available)
        with pytest.raises(RuntimeError, match="needs CUDA tensors"):
            glm.fixed_launch_sinkhorn(logits, ITERS)
        with pytest.raises(RuntimeError, match="needs CUDA tensors"):
            glm._sinkhorn_bwd(logits.detach(), logits.detach(), ITERS)


# --- GPU ------------------------------------------------------------------------------------


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("scale", [1.0, 10.0], ids=["x1", "x10"])
def test_fp32_bar_against_float64(scale):
    """The output and the logits gradient are within the rel-RMS floor of float64."""
    logits, grad = _production_inputs(1234, 4321, scale)
    out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
    (logits_grad,) = torch.autograd.grad(out, logits, grad)
    out64, grad64 = _float64_reference(logits, grad)
    for name, actual, reference in (
        ("output", out, out64),
        ("logits gradient", logits_grad, grad64),
    ):
        assert torch.isfinite(actual).all(), name
        error = rel_rms(actual.detach().cpu(), reference)
        assert error <= REL_RMS_FLOOR, f"{name}: rel-RMS {error:.3e} > {REL_RMS_FLOOR:g}"


@_NEEDS_CUDA_TRITON
def test_bf16_input_matches_float64_reference():
    """A bf16 input runs the same kernels, with the output and M_init in bf16."""
    shape = (8, 4, N, N)
    logits = torch.randn(shape, generator=torch.Generator().manual_seed(1234)).bfloat16()
    grad = torch.randn(shape, generator=torch.Generator().manual_seed(4321)).bfloat16()
    x = logits.cuda().requires_grad_()
    out = glm.fixed_launch_sinkhorn(x, ITERS, EPS)
    (x_grad,) = torch.autograd.grad(out, x, grad.cuda())
    assert out.dtype == x_grad.dtype == torch.bfloat16
    out64, grad64 = _float64_reference(logits, grad)
    # The bf16 tolerances of fused_mhc_kernels' tests.
    torch.testing.assert_close(out.detach().cpu().double(), out64, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_grad.cpu().double(), grad64, atol=5e-2, rtol=5e-2)


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("case", ["default", "deterministic", "patched_constant"])
def test_launches_use_fixed_num_warps(monkeypatch, case):
    """Both launches apply SINKHORN_NUM_WARPS, read at each launch, in either torch mode.

    fused_mhc_kernels' own wrapper launches 4 warps under torch's deterministic mode; this
    launcher has no deterministic branch. A patched constant reaches the launch, as it reaches
    the per-rank fused-ops line (test_mhc_fused_ops.py).
    """
    expected = 1
    if case == "patched_constant":
        monkeypatch.setattr(glm, "SINKHORN_NUM_WARPS", 2)
        expected = 2
    logits, grad = _production_inputs(1234, 4321)
    launches = _record_launches(monkeypatch)
    mode = _det_fill() if case == "deterministic" else contextlib.nullcontext()
    with mode:
        out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
        (logits_grad,) = torch.autograd.grad(out, logits, grad)
    assert launches == [("fwd", expected), ("bwd", expected)]
    assert torch.isfinite(out).all() and torch.isfinite(logits_grad).all()


@_NEEDS_CUDA_TRITON
def test_det_fill_nan_free_at_production_shape():
    """A NaN smoke test: 100 calls under det-fill stay finite and repeat bit for bit.

    Under det-fill every torch.empty, the backward's three workspaces included, starts as NaN,
    so an output, M_init or gradient element that no program writes stays NaN; the positive
    control shows this for a grid one program short. The calls alternate two input sets, so a
    read of memory the call did not write shows up as a bit change. At one warp and n = 4 no
    race of the reverse pass can be provoked: test_shared_backward_kernel_barrier_n8 covers the
    barrier.
    """
    input_sets = [_production_inputs(1234, 4321), _production_inputs(5678, 8765)]
    first, nonfinite_calls, changed_calls = {}, [], []
    with _det_fill():
        filled = torch.isnan(torch.empty(64, device="cuda")).all()
        assert filled, "R5 precondition: a fresh torch.empty is not all NaN under det-fill"

        # Positive control: the forward JIT function one program short at the production shape.
        fwd_kernel, _ = glm._sinkhorn_kernels()
        out = torch.empty(T, N, N, device="cuda")
        M_init = torch.empty(T, N, N, device="cuda")
        inp = input_sets[0][0].detach().contiguous().view(T, N, N)
        fwd_kernel[(T - 1,)](inp, out, M_init, T, EPS, N, ITERS, num_warps=glm.SINKHORN_NUM_WARPS)
        for name, tensor in (("out", out), ("M_init", M_init)):
            flagged = torch.isnan(tensor[-1]).all() and torch.isfinite(tensor[:-1]).all()
            assert flagged, f"R5 positive control: {name} is not NaN in exactly its last matrix"

        for call in range(100):
            index = call % len(input_sets)
            logits, grad = input_sets[index]
            out = glm.fixed_launch_sinkhorn(logits, ITERS, EPS)
            (logits_grad,) = torch.autograd.grad(out, logits, grad)
            if not (torch.isfinite(out).all() and torch.isfinite(logits_grad).all()):
                nonfinite_calls.append(call)
            bits = [t.detach().contiguous().view(torch.int32) for t in (out, logits_grad)]
            if index not in first:
                first[index] = [b.clone() for b in bits]
            elif not all(torch.equal(b, f) for b, f in zip(bits, first[index])):
                changed_calls.append(call)
    assert not nonfinite_calls, f"non-finite output or gradient in calls {nonfinite_calls}"
    assert not changed_calls, f"bits differ from the input set's first call in {changed_calls}"


@_NEEDS_CUDA_TRITON
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_shared_backward_kernel_barrier_n8(monkeypatch, num_warps):
    """The shared backward kernel at n = 8 under det-fill, at warp counts that can race.

    The backward stores the recomputed matrices and sums in scratch and reads them back in the
    reverse pass; at n = 8 with several warps a read could run before another warp's store
    without the barrier. Inputs and tolerances are those of fused_mhc_kernels' deterministic
    Sinkhorn launch test. The launch record shows the requested warps were applied: at one
    warp the kernel passes without the barrier too.
    """
    generator = torch.Generator().manual_seed(193)
    data = (torch.rand((3, 1, 8, 8), generator=generator) * 0.2 - 0.1).cuda()
    grad_out = (torch.rand((3, 1, 8, 8), generator=generator) * 0.2 - 0.1).cuda()
    out64, grad64 = _float64_reference(data, grad_out, iterations=5)
    launches = _record_launches(monkeypatch)
    with _det_fill():
        out, M_init = glm._sinkhorn_fwd(data, 5, EPS, num_warps=num_warps)
        grad = glm._sinkhorn_bwd(grad_out, M_init, 5, EPS, num_warps=num_warps)
        torch.cuda.synchronize()
    assert launches == [("fwd", num_warps), ("bwd", num_warps)]
    for name, actual, reference in (("output", out, out64), ("input gradient", grad, grad64)):
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual.cpu().double(), reference, atol=2e-6, rtol=2e-5, msg=name)
