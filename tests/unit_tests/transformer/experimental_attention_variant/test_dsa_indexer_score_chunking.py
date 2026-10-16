# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row-chunked cuDNN DSA indexer scoring.

CPU tests drive the scoring paths through a PyTorch model of the cuDNN frontend contract
(dense ReLU-weighted scores, per-segment causal offsets, per-row top-k) and check them
against an independent per-row reference built from global token positions. GPU tests run
the real frontend and require bitwise-identical top-k between chunked and single-launch
scoring.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.dsa_fused_safety import (
    FUSED_INDEXER_MAX_SAFE_ROWS,
)

TOPK = 4


def _dense_scores(q_rows, k_rows, w_rows):
    """``sum_h w[r, h] * relu(q[r, h] . k[c])`` in FP32 for ``(n, h, d)``/``(m, d)``/``(n, h)``.

    Rows are scored one at a time so a row's value never depends on which other rows share
    the call, mirroring the per-element independence of the cuDNN scorer.
    """
    rows = []
    for q_row, w_row in zip(q_rows.float(), w_rows.float()):
        logits = (q_row @ k_rows.float().t()).relu_()
        rows.append((logits * w_row.unsqueeze(-1)).sum(dim=0))
    return torch.stack(rows) if rows else torch.zeros((0, k_rows.size(0)))


class _FrontendModel:
    """PyTorch model of the cuDNN DSA scorer and top-k used by ``dsa_cudnn_kernels``.

    ``default_alignment`` selects the causal alignment used when ``q_causal_offsets`` is
    omitted: bottom-right up to cudnn-frontend 1.25, top-left from 1.26 on.
    """

    def __init__(self, default_alignment="bottom_right", pad_output=False, max_vector_topk=None):
        self.default_alignment = default_alignment
        # cudnn-frontend <= 1.26 pads the THD score width to a multiple of 4 for TMA and
        # returns a row-strided view of the padded buffer.
        self.pad_output = pad_output
        # cudnn-frontend <= 1.26 top-k cannot compile an odd K above its vector width.
        self.max_vector_topk = max_vector_topk
        self.forward_calls = []
        self.topk_calls = []

    def _default_offset(self, seqlen_q, seqlen_k):
        return seqlen_k - seqlen_q if self.default_alignment == "bottom_right" else 0

    def indexer_forward_wrapper(
        self,
        q,
        k,
        w,
        ratio=1,
        sm_scale=1.0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
    ):
        assert ratio == 1 and sm_scale == 1.0
        offsets = None if q_causal_offsets is None else q_causal_offsets.tolist()
        if cu_seqlens_q is None:
            b, sq, _h, _d = q.shape
            sk = k.size(1)
            self.forward_calls.append({"rows": b * sq, "width": sk, "offsets": offsets})
            padded_width = -(-sk // 4) * 4 if self.pad_output else sk
            out = torch.full((b, sq, padded_width), float("-inf"))[:, :, :sk]
            self.forward_calls[-1]["storage"] = out.untyped_storage().data_ptr()
            for bi in range(b):
                offset = self._default_offset(sq, sk) if offsets is None else offsets[bi]
                limit = offset + torch.arange(sq) + 1
                visible = torch.arange(sk).view(1, -1) < limit.view(-1, 1)
                scores = _dense_scores(q[bi], k[bi, :, 0], w[bi])
                out[bi] = torch.where(visible, scores, out[bi])
            return {"scores": out}

        cu_q = cu_seqlens_q.tolist()
        cu_k = cu_seqlens_k.tolist()
        segment_q_lengths = [hi - lo for lo, hi in zip(cu_q[:-1], cu_q[1:])]
        assert q.size(0) == cu_q[-1] and max(segment_q_lengths) <= max_seqlen_q
        self.forward_calls.append(
            {"rows": q.size(0), "width": max_seqlen_k, "offsets": offsets, "cu_q": cu_q}
        )
        padded_width = -(-max_seqlen_k // 4) * 4 if self.pad_output else max_seqlen_k
        out = torch.full((q.size(0), padded_width), float("-inf"))[:, :max_seqlen_k]
        self.forward_calls[-1]["storage"] = out.untyped_storage().data_ptr()
        for segment in range(len(cu_q) - 1):
            q0, q1, k0, k1 = cu_q[segment], cu_q[segment + 1], cu_k[segment], cu_k[segment + 1]
            if q1 == q0:
                continue
            offset = self._default_offset(q1 - q0, k1 - k0) if offsets is None else offsets[segment]
            limit = offset + torch.arange(q1 - q0) + 1
            visible = torch.arange(k1 - k0).view(1, -1) < limit.view(-1, 1)
            scores = _dense_scores(q[q0:q1], k[k0:k1, 0], w[q0:q1])
            out[q0:q1, : k1 - k0] = torch.where(visible, scores, out[q0:q1, : k1 - k0])
        return {"scores": out}

    def indexer_top_k_wrapper(self, scores, seq_lens, top_k, next_n, return_val):
        assert next_n == 1
        if self.max_vector_topk is not None and top_k > self.max_vector_topk:
            assert top_k % 2 == 0, f"frontend top-k cannot compile odd K={top_k}"
        self.topk_calls.append(
            {
                "storage": scores.untyped_storage().data_ptr(),
                "contiguous": scores.is_contiguous(),
                "rows": scores.size(0),
                "k": top_k,
                # The legacy per-head PyTorch scorer never calls the frontend scorer.
                "forward": self.forward_calls[-1] if self.forward_calls else None,
            }
        )
        rows, cols = scores.shape
        indices = torch.full((rows, top_k), -1, dtype=torch.int32)
        values = torch.full((rows, top_k), float("-inf"))
        for row, seq_len in enumerate(seq_lens.tolist()):
            length = min(seq_len, cols)
            if length <= top_k:
                indices[row, :length] = torch.arange(length, dtype=torch.int32)
                values[row, :length] = scores[row, :length]
            else:
                row_values, row_indices = scores[row, :length].topk(top_k)
                indices[row] = row_indices.int()
                values[row] = row_values
        return {"indices": indices, "values": values if return_val else None}


class _LegacyFrontendModel(_FrontendModel):
    """Frontend with THD metadata but without per-segment Q offsets."""

    def indexer_forward_wrapper(
        self,
        q,
        k,
        w,
        ratio=1,
        sm_scale=1.0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):
        return super().indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=ratio,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )


@pytest.fixture
def frontend(monkeypatch):
    model = _FrontendModel()
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", model)
    return model


def _zigzag_row_bounds(sequence_lengths, cp_size, cp_rank):
    """Global ``[start, end)`` key bounds of each local query row under zigzag CP."""
    starts, ends = [], []
    sequence_start = 0
    for length in sequence_lengths:
        if cp_size == 1:
            chunks, chunk_len = (0,), length
        else:
            chunk_len = length // (2 * cp_size)
            chunks = (cp_rank, 2 * cp_size - 1 - cp_rank)
        for chunk in chunks:
            for row in range(chunk_len):
                starts.append(sequence_start)
                ends.append(sequence_start + chunk * chunk_len + row + 1)
        sequence_start += length
    return torch.tensor(starts), torch.tensor(ends)


def _reference_topk(q_rows, k_keys, w_rows, starts, ends, topk):
    """Per-row top-k over global keys ``[start, end)``, as sorted index lists and values."""
    indices, values = [], []
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        scores = _dense_scores(q_rows[row : row + 1], k_keys[start:end], w_rows[row : row + 1])[0]
        count = min(topk, end - start)
        row_values, row_indices = scores.topk(count)
        order = (row_indices + start).argsort()
        indices.append((row_indices + start)[order].tolist())
        values.append(row_values[order])
    return indices, values


def _assert_matches_reference(topk_indices, topk_values, reference):
    ref_indices, ref_values = reference
    for row, (expected, expected_values) in enumerate(zip(ref_indices, ref_values)):
        selected = topk_indices[0, row]
        valid = selected >= 0
        order = selected[valid].argsort()
        assert selected[valid][order].tolist() == expected, f"row {row}"
        if topk_values is not None:
            torch.testing.assert_close(
                topk_values[0, row][valid][order], expected_values, rtol=0, atol=1e-5
            )


def _packed_inputs(sequence_lengths, cp_size, cp_rank, n_heads=4, head_dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total = sum(sequence_lengths)
    local_rows = total // cp_size
    q = torch.randn((1, local_rows, n_heads, head_dim), generator=generator)
    k = torch.randn((1, total, 1, head_dim), generator=generator)
    w = torch.rand((1, local_rows, n_heads), generator=generator)
    starts, ends = _zigzag_row_bounds(sequence_lengths, cp_size, cp_rank)
    cu_seqlens = torch.tensor([0, *torch.tensor(sequence_lengths).cumsum(0).tolist()])
    return q, k, w, starts, ends, cu_seqlens.to(torch.int32), max(sequence_lengths)


def _run_packed_thd(inputs, cp_size, cp_rank, return_topk_scores, **kwargs):
    q, k, w, starts, ends, cu_seqlens, max_seqlen = inputs
    return dsa_cudnn_kernels._indexer_topk_packed_thd(
        q,
        k,
        w,
        TOPK,
        return_topk_scores,
        starts,
        ends,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        cp_size,
        cp_rank,
        0,
        None,
        **kwargs,
    )


PACKED_LAYOUTS = [
    pytest.param([5, 7, 4], 1, 0, id="cp1-three-sequences"),
    pytest.param([24, 8], 2, 0, id="cp2-rank0"),
    pytest.param([24, 8], 2, 1, id="cp2-rank1"),
    pytest.param([32, 16, 16], 4, 3, id="cp4-rank3-with-tail"),
]


@pytest.mark.parametrize("sequence_lengths,cp_size,cp_rank", PACKED_LAYOUTS)
@pytest.mark.parametrize("launch_rows", [1, 3, 5, 64])
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_packed_thd_row_chunks_match_single_launch_and_reference(
    frontend, monkeypatch, sequence_lengths, cp_size, cp_rank, launch_rows, return_topk_scores
):
    inputs = _packed_inputs(sequence_lengths, cp_size, cp_rank)
    single = _run_packed_thd(inputs, cp_size, cp_rank, return_topk_scores)
    assert len(frontend.forward_calls) == 1

    frontend.forward_calls.clear()
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_kwargs: launch_rows
    )
    chunked = _run_packed_thd(inputs, cp_size, cp_rank, return_topk_scores, score_chunk_max_bytes=1)

    local_rows = inputs[0].size(1)
    expected_launches = -(-local_rows // launch_rows)
    assert len(frontend.forward_calls) == expected_launches
    assert max(call["rows"] for call in frontend.forward_calls) <= launch_rows
    if expected_launches > 1:
        assert all(call["offsets"] is not None for call in frontend.forward_calls)
    torch.testing.assert_close(chunked[0], single[0], rtol=0, atol=0)
    if return_topk_scores:
        torch.testing.assert_close(chunked[1], single[1], rtol=0, atol=0)
    else:
        assert chunked[1] is None and single[1] is None

    q, k, w, starts, ends, _cu, _max = inputs
    reference = _reference_topk(q[0], k[0, :, 0], w[0], starts, ends, TOPK)
    _assert_matches_reference(chunked[0], chunked[1], reference)


def test_packed_thd_without_bound_scores_in_one_launch(frontend, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("an unset bound must not consult the launch-row policy")

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_launch_rows", unexpected)
    _run_packed_thd(_packed_inputs([24, 8], 2, 1), 2, 1, False)
    assert len(frontend.forward_calls) == 1


@pytest.mark.parametrize("sequence_lengths,cp_size,cp_rank", PACKED_LAYOUTS)
def test_packed_thd_unbounded_launch_keeps_original_frontend_arguments(
    monkeypatch, sequence_lengths, cp_size, cp_rank
):
    """Without a bound the scorer is called once with the pre-chunking argument contract."""
    seen = []

    class RecordingFrontend(_FrontendModel):
        def indexer_forward_wrapper(self, q, k, w, **kwargs):
            seen.append({"q": q, "k": k, "w": w, **kwargs})
            return super().indexer_forward_wrapper(q, k, w, **kwargs)

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", RecordingFrontend())
    inputs = _packed_inputs(sequence_lengths, cp_size, cp_rank)
    q, k, w, _starts, _ends, cu_seqlens, max_seqlen = inputs
    _run_packed_thd(inputs, cp_size, cp_rank, False)

    assert len(seen) == 1
    call = seen[0]
    assert call["q"].data_ptr() == q.data_ptr() and call["q"].shape == q[0].shape
    assert call["w"].data_ptr() == w.data_ptr() and call["w"].shape == w[0].shape
    assert call["ratio"] == 1 and call["sm_scale"] == 1.0
    if cp_size == 1:
        assert set(call) == {
            "q",
            "k",
            "w",
            "ratio",
            "sm_scale",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "max_seqlen_q",
            "max_seqlen_k",
        }
        assert call["cu_seqlens_q"] is cu_seqlens and call["cu_seqlens_k"] is cu_seqlens
        assert call["max_seqlen_q"] == max_seqlen and call["max_seqlen_k"] == max_seqlen
        assert call["k"].data_ptr() == k.data_ptr()
        return
    layout = dsa_cudnn_kernels.dsa_layout.build_packed_cp_indexer_layout(
        cu_seqlens, cu_seqlens, cp_size=cp_size, cp_rank=cp_rank, key_size=k.size(1)
    )
    torch.testing.assert_close(call["cu_seqlens_q"], layout.segment_cu_q.to(torch.int32))
    torch.testing.assert_close(call["cu_seqlens_k"], layout.segment_cu_k.to(torch.int32))
    torch.testing.assert_close(
        call["q_causal_offsets"],
        (layout.segment_k_lengths - layout.segment_q_lengths).to(torch.int32),
    )
    half = max_seqlen // (2 * cp_size)
    assert call["max_seqlen_q"] == half
    assert call["max_seqlen_k"] == max((cp_rank + 1) * half, max_seqlen - cp_rank * half)


def test_packed_thd_chunking_needs_q_offsets(monkeypatch):
    legacy = _LegacyFrontendModel()
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", legacy)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_k: 2)
    inputs = _packed_inputs([5, 7, 4], 1, 0)
    _run_packed_thd(inputs, 1, 0, False, score_chunk_max_bytes=1)
    assert len(legacy.forward_calls) == 1


@pytest.mark.parametrize("default_alignment", ["bottom_right", "top_left"])
@pytest.mark.parametrize("launch_rows", [2, 5, 64])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_single_sequence_segments_use_explicit_offsets(
    monkeypatch, default_alignment, launch_rows, cp_rank
):
    frontend = _FrontendModel(default_alignment)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", frontend)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_kwargs: launch_rows
    )
    cp_size, sequence_length = 2, 24
    q, k, w, starts, ends, _cu, _max = _packed_inputs([sequence_length], cp_size, cp_rank)

    topk_indices, topk_values = dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k, w, TOPK, True, cp_size, cp_rank, 0, q.size(1)
    )

    assert all(call["offsets"] is not None for call in frontend.forward_calls)
    reference = _reference_topk(q[0], k[0, :, 0], w[0], starts, ends, TOPK)
    _assert_matches_reference(topk_indices, topk_values, reference)


@pytest.mark.parametrize("default_alignment", ["bottom_right", "top_left"])
@pytest.mark.parametrize("launch_rows", [3, 64])
def test_plain_causal_uses_explicit_offsets(monkeypatch, default_alignment, launch_rows):
    frontend = _FrontendModel(default_alignment)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", frontend)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_kwargs: launch_rows
    )
    generator = torch.Generator().manual_seed(1)
    b, sq, heads, dim = 2, 11, 4, 8
    q = torch.randn((b, sq, heads, dim), generator=generator)
    k = torch.randn((b, sq, dim), generator=generator)
    w = torch.rand((b, sq, heads), generator=generator)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        q, k, w, TOPK, return_scores=False
    )

    assert all(call["offsets"] is not None for call in frontend.forward_calls)
    for batch in range(b):
        starts = torch.zeros(sq, dtype=torch.long)
        ends = torch.arange(1, sq + 1)
        ref_indices, _ = _reference_topk(q[batch], k[batch], w[batch], starts, ends, TOPK)
        for row, expected in enumerate(ref_indices):
            length = int(topk_length[batch, row])
            assert sorted(topk_indices[batch, row, :length].tolist()) == expected


def test_unfused_bounds_scores_every_key_and_masks(frontend, monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_k: 4)
    generator = torch.Generator().manual_seed(2)
    sq, sk, heads, dim = 9, 12, 4, 8
    q = torch.randn((1, sq, heads, dim), generator=generator)
    k = torch.randn((1, sk, dim), generator=generator)
    w = torch.rand((1, sq, heads), generator=generator)
    starts = torch.tensor([0, 2, 2, 5, 0, 7, 1, 3, 4])
    ends = torch.tensor([3, 6, 9, 12, 1, 12, 2, 10, 8])

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        q, k, w, TOPK, varlen_starts=starts, varlen_ends=ends, return_scores=False
    )

    assert all(call["offsets"] == [sk] for call in frontend.forward_calls)
    ref_indices, _ = _reference_topk(q[0], k[0], w[0], starts, ends, TOPK)
    for row, expected in enumerate(ref_indices):
        length = int(topk_length[0, row])
        assert sorted(topk_indices[0, row, :length].tolist()) == expected


def test_launch_rows_policy():
    cpu = torch.device("cpu")
    heads = 32
    tokens_per_tile = dsa_cudnn_kernels._indexer_score_tokens_per_tile(heads)
    assert tokens_per_tile == 8

    # An explicit bound is a hard cap, rounded down to whole tiles.
    width = 1 << 20
    max_bytes = 100 * width * 4
    rows = dsa_cudnn_kernels._indexer_score_launch_rows(
        16384, batch=1, score_width=width, n_heads=heads, device=cpu, max_bytes=max_bytes
    )
    assert rows == 96

    # Small layouts stay in one launch.
    assert (
        dsa_cudnn_kernels._indexer_score_launch_rows(
            4096, batch=1, score_width=4096, n_heads=heads, device=cpu, max_bytes=1 << 30
        )
        == 4096
    )

    # One launch never exceeds the fused kernel's verified-safe row count.
    rows = dsa_cudnn_kernels._indexer_score_launch_rows(
        4 * FUSED_INDEXER_MAX_SAFE_ROWS,
        batch=2,
        score_width=8,
        n_heads=heads,
        device=cpu,
        max_bytes=1 << 40,
    )
    assert rows == FUSED_INDEXER_MAX_SAFE_ROWS // 2

    # Without a bound the built-in budget applies.
    rows = dsa_cudnn_kernels._indexer_score_launch_rows(
        16384, batch=1, score_width=width, n_heads=heads, device=cpu, max_bytes=None
    )
    assert rows == dsa_cudnn_kernels._INDEXER_SCORE_CHUNK_MAX_BYTES // (width * 4)


def test_hook_forwards_score_chunk_bound_only_when_configured():
    def backend_fn(indexer_score_chunk_max_bytes=None, deterministic=False):
        del indexer_score_chunk_max_bytes, deterministic

    unset = SimpleNamespace(dsa_indexer_score_chunk_max_gb=None, deterministic_mode=False)
    assert dsa_kernels._indexer_hook_kwargs(backend_fn, unset) == {"deterministic": False}
    assert dsa_kernels._indexer_hook_kwargs(backend_fn, unset, deterministic=False) == {}

    configured = SimpleNamespace(dsa_indexer_score_chunk_max_gb=1.5, deterministic_mode=False)
    assert dsa_kernels._indexer_hook_kwargs(backend_fn, configured, deterministic=False) == {
        "indexer_score_chunk_max_bytes": 3 << 29
    }


def test_packed_thd_passes_padded_frontend_view_to_topk_without_copy(monkeypatch):
    """A padded (row-strided) score view goes to top-k as is, never copied whole."""
    frontend = _FrontendModel(pad_output=True)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", frontend)
    inputs = _packed_inputs([7, 5, 3], 1, 0)
    _run_packed_thd(inputs, 1, 0, True)
    assert frontend.forward_calls[0]["width"] % 4 != 0
    assert frontend.topk_calls[0]["storage"] == frontend.topk_calls[0]["forward"]["storage"]
    assert not frontend.topk_calls[0]["contiguous"]

    frontend.forward_calls.clear()
    frontend.topk_calls.clear()
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_k: 4)
    _run_packed_thd(inputs, 1, 0, True, score_chunk_max_bytes=1)
    assert len(frontend.topk_calls) == len(frontend.forward_calls) > 1
    for topk in frontend.topk_calls:
        assert topk["storage"] == topk["forward"]["storage"]


def test_prefix_windows_never_ask_frontend_for_covered_odd_topk(monkeypatch):
    """Windows whose width is covered by top-k keep their valid prefix without ranking."""
    frontend = _FrontendModel("top_left", max_vector_topk=256)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", frontend)
    generator = torch.Generator().manual_seed(3)
    sq, topk = 601, 2048
    q = torch.randn((1, sq, 4, 8), generator=generator)
    k = torch.randn((1, sq, 8), generator=generator)
    w = torch.rand((1, sq, 4), generator=generator)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        q, k, w, topk, return_scores=False
    )

    assert not frontend.topk_calls
    torch.testing.assert_close(topk_length[0], torch.arange(1, sq + 1, dtype=torch.int32))
    for row in (0, 1, 300, sq - 1):
        assert sorted(topk_indices[0, row, : row + 1].tolist()) == list(range(row + 1))


def test_launch_rows_cap_is_applied_before_tile_alignment():
    rows = dsa_cudnn_kernels._indexer_score_launch_rows(
        17 * 2048,
        batch=17,
        score_width=2048,
        n_heads=32,
        device=torch.device("cpu"),
        max_bytes=None,
    )
    assert rows % 8 == 0 and rows * 17 <= FUSED_INDEXER_MAX_SAFE_ROWS


def test_wave_rows_scale_with_batch(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_cuda_sm_count", lambda _index: 148)
    cuda = torch.device("cuda", 0)
    assert dsa_cudnn_kernels._indexer_score_wave_rows(32, cuda) == 8 * 148
    assert dsa_cudnn_kernels._indexer_score_wave_rows(32, cuda, batch=4) == 8 * 37
    rows = dsa_cudnn_kernels._indexer_score_launch_rows(
        8192, batch=4, score_width=1 << 20, n_heads=32, device=cuda, max_bytes=None
    )
    assert rows == 8 * 37


def test_packed_thd_bound_applies_when_topk_covers_the_width(frontend, monkeypatch):
    inputs = _packed_inputs([24, 8], 2, 1)
    wide_topk = 64
    q, k, w, starts, ends, cu_seqlens, max_seqlen = inputs
    args = (q, k, w, wide_topk, True, starts, ends, cu_seqlens, cu_seqlens)
    single = dsa_cudnn_kernels._indexer_topk_packed_thd(
        *args, max_seqlen, max_seqlen, 2, 1, 0, None
    )
    frontend.forward_calls.clear()
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_k: 3)
    chunked = dsa_cudnn_kernels._indexer_topk_packed_thd(
        *args, max_seqlen, max_seqlen, 2, 1, 0, None, score_chunk_max_bytes=1
    )
    assert len(frontend.forward_calls) == -(-q.size(1) // 3)
    torch.testing.assert_close(chunked[0], single[0], rtol=0, atol=0)
    torch.testing.assert_close(chunked[1], single[1], rtol=0, atol=0)


def test_legacy_frontend_keeps_default_launches_and_warns(monkeypatch, caplog):
    """Without q_causal_offsets a bound cannot be honored with the cuDNN scorer: warn once."""
    legacy = _LegacyFrontendModel()
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", legacy)
    monkeypatch.setattr(dsa_cudnn_kernels, "_SCORE_CHUNK_BOUND_UNAVAILABLE_WARNED", False)
    generator = torch.Generator().manual_seed(4)
    sq = 12
    q = torch.randn((1, sq, 4, 8), generator=generator)
    k = torch.randn((1, sq, 8), generator=generator)
    w = torch.rand((1, sq, 4), generator=generator)

    with caplog.at_level("WARNING"):
        for _ in range(2):
            topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
                q, k, w, TOPK, return_scores=False, score_chunk_max_bytes=5 * sq * 4
            )

    assert len(legacy.forward_calls) == 2  # one unchunked cuDNN launch per call
    assert caplog.text.count("does not accept q_causal_offsets") == 1
    starts = torch.zeros(sq, dtype=torch.long)
    ref_indices, _ = _reference_topk(q[0], k[0], w[0], starts, torch.arange(1, sq + 1), TOPK)
    for row, expected in enumerate(ref_indices):
        assert sorted(topk_indices[0, row, : int(topk_length[0, row])].tolist()) == expected


def test_dense_scores_warn_that_the_bound_does_not_apply(frontend, monkeypatch, caplog):
    monkeypatch.setattr(dsa_cudnn_kernels, "_DENSE_SCORES_CHUNK_BOUND_WARNED", False)
    q = torch.randn((1, 6, 4, 8))
    k = torch.randn((1, 6, 8))
    w = torch.rand((1, 6, 4))
    with caplog.at_level("WARNING"):
        dsa_cudnn_kernels._indexer_topk_bshd(q, k, w, TOPK, return_scores=True)
        assert "does not bound the dense indexer loss" not in caplog.text
        for _ in range(2):
            dsa_cudnn_kernels._indexer_topk_bshd(
                q, k, w, TOPK, return_scores=True, score_chunk_max_bytes=1
            )
    assert caplog.text.count("does not bound the dense indexer loss") == 1


def test_unfused_scoring_report_names_the_executor(monkeypatch):
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", _FrontendModel())
    assert dsa_cudnn_kernels._unfused_scoring_context(False).endswith(
        "bounds-masked cuDNN dense scorer"
    )
    assert dsa_cudnn_kernels._unfused_scoring_context(True).endswith("per-head PyTorch loop")
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", _LegacyFrontendModel())
    assert dsa_cudnn_kernels._unfused_scoring_context(False).endswith("per-head PyTorch loop")


@pytest.mark.parametrize("launch_rows", [3, 64])
def test_prefix_launches_compact_padded_scores_before_topk(monkeypatch, launch_rows):
    """BSHD launches keep base's compact top-k input even when the frontend pads its output."""
    frontend = _FrontendModel("top_left", pad_output=True)
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", frontend)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_k: launch_rows
    )
    generator = torch.Generator().manual_seed(5)
    sq = 13
    q = torch.randn((1, sq, 4, 8), generator=generator)
    k = torch.randn((1, sq, 8), generator=generator)
    w = torch.rand((1, sq, 4), generator=generator)

    topk_indices, topk_length, _ = dsa_cudnn_kernels._indexer_topk_bshd(
        q, k, w, TOPK, return_scores=False
    )

    padded_topk = [c for c in frontend.topk_calls if c["forward"] and c["forward"]["width"] % 4]
    assert padded_topk and all(c["contiguous"] for c in padded_topk)
    starts = torch.zeros(sq, dtype=torch.long)
    ref_indices, _ = _reference_topk(q[0], k[0], w[0], starts, torch.arange(1, sq + 1), TOPK)
    for row, expected in enumerate(ref_indices):
        assert sorted(topk_indices[0, row, : int(topk_length[0, row])].tolist()) == expected


def test_topk_with_scores_hook_forwards_the_bound_only_when_configured(monkeypatch):
    """Checkpoint top-k reuse scores through run_fused_qk_topk_with_scores; it is bounded too."""
    seen = []

    def backend_fn(
        q,
        k,
        weights,
        index_topk,
        starts,
        ends,
        block_size,
        use_relu=True,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        packed_seq_params=None,
        cp_size=1,
        deterministic=False,
        query=None,
        key=None,
        config=None,
        indexer_score_chunk_max_bytes=None,
        **_layout_kwargs,
    ):
        seen.append(
            {
                "deterministic": deterministic,
                "query": query,
                "key": key,
                "config": config,
                "bound": indexer_score_chunk_max_bytes,
            }
        )
        return "topk"

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda _config, _name: backend_fn)
    q, k, w = torch.zeros((4, 1, 2, 8)), torch.zeros((4, 1, 8)), torch.zeros((4, 1, 2))
    query, key = torch.zeros((4, 1, 2, 16)), torch.zeros((4, 1, 1, 16))
    bounds = torch.zeros(4, dtype=torch.long)
    for max_gb, expected in ((None, None), (0.5, 1 << 29)):
        config = SimpleNamespace(dsa_indexer_score_chunk_max_gb=max_gb, deterministic_mode=False)
        result = dsa_kernels.run_fused_qk_topk_with_scores(
            config, q, k, w, 2, bounds, bounds, 64, query=query, key=key
        )
        assert result == "topk"
        call = seen[-1]
        assert call["bound"] == expected
        assert call["query"] is query and call["key"] is key and call["config"] is config
        assert call["deterministic"] is False

    def minimal_backend(
        q,
        k,
        weights,
        index_topk,
        starts,
        ends,
        block_size,
        use_relu=True,
        use_local_indexer_varlen=False,
        single_packed_thd_sequence=False,
        local_packed_cp_rank=0,
        local_packed_cp_query_start=0,
        local_packed_cp_query_len=None,
        packed_seq_params=None,
        cp_size=1,
    ):
        """A backend without the optional kwargs: they must be filtered out, not passed."""
        return "minimal"

    monkeypatch.setattr(dsa_kernels, "_resolve_fused_hook", lambda *_a: minimal_backend)
    config = SimpleNamespace(dsa_indexer_score_chunk_max_gb=0.5, deterministic_mode=False)
    assert (
        dsa_kernels.run_fused_qk_topk_with_scores(
            config, q, k, w, 2, bounds, bounds, 64, query=query, key=key
        )
        == "minimal"
    )


def test_topk_with_scores_backend_forwards_the_bound(monkeypatch):
    seen = []

    def fake_topk_bshd(q_bshd, k_bsd, w_bsh, topk, **kwargs):
        seen.append(kwargs)
        b, sq = q_bshd.shape[:2]
        indices = torch.arange(topk, dtype=torch.int32).expand(b, sq, topk).contiguous()
        return indices, torch.full((b, sq), topk, dtype=torch.int32), indices.float()

    monkeypatch.setattr(
        dsa_cudnn_kernels, "_sparse_indexer_loss_layout_supported", lambda *_a: True
    )
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_bshd", fake_topk_bshd)
    q, k, w = torch.zeros((4, 1, 2, 8)), torch.zeros((4, 1, 8)), torch.zeros((4, 1, 2))
    bounds = torch.zeros(4, dtype=torch.long)
    for bound in (None, 12345):
        extra = {} if bound is None else {"indexer_score_chunk_max_bytes": bound}
        indices, length, scores = dsa_cudnn_kernels.run_fused_qk_topk_with_scores(
            q, k, w, 2, bounds, bounds, 64, **extra
        )
        assert indices.shape == (1, 4, 2) and scores.dtype == torch.float32
        if bound is None:
            assert "score_chunk_max_bytes" not in seen[-1]
        else:
            assert seen[-1]["score_chunk_max_bytes"] == bound


@pytest.mark.parametrize("sequence_lengths,cp_size,cp_rank", PACKED_LAYOUTS)
@pytest.mark.parametrize("deterministic", [False, True])
def test_topk_with_scores_bounded_stash_matches_single_launch(
    frontend, monkeypatch, sequence_lengths, cp_size, cp_rank, deterministic
):
    """The checkpoint top-k stash is identical whether its scoring was chunked or not."""
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_sparse_indexer_loss_layout_supported", lambda *_a: True
    )
    q, k, w, starts, ends, cu_seqlens, max_seqlen = _packed_inputs(
        sequence_lengths, cp_size, cp_rank
    )
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )

    def stash(**extra):
        frontend.forward_calls.clear()
        return dsa_cudnn_kernels.run_fused_qk_topk_with_scores(
            q.transpose(0, 1).contiguous(),
            k[:, :, 0].transpose(0, 1).contiguous(),
            w.transpose(0, 1).contiguous(),
            TOPK,
            starts,
            ends,
            64,
            use_local_indexer_varlen=True,
            local_packed_cp_rank=cp_rank,
            packed_seq_params=packed_seq_params,
            cp_size=cp_size,
            deterministic=deterministic,
            **extra,
        )

    single = stash()
    assert len(frontend.forward_calls) == 1
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_launch_rows", lambda rows, **_k: 3)
    chunked = stash(indexer_score_chunk_max_bytes=1)
    assert len(frontend.forward_calls) > 1
    for got, expected in zip(chunked, single):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# Real cuDNN frontend
# --------------------------------------------------------------------------------------


def _real_frontend_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the cuDNN DSA indexer")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the cuDNN DSA indexer forward requires SM100+")
    try:
        dsa_cudnn_kernels._ensure_dsa_namespace()
    except ImportError as error:
        pytest.skip(f"cudnn-frontend DSA namespace unavailable: {error}")
    if not dsa_cudnn_kernels._supports_indexer_q_causal_offsets():
        pytest.skip("installed cudnn-frontend lacks q_causal_offsets")


def _cuda_packed_inputs(sequence_lengths, cp_size, cp_rank, n_heads=32, head_dim=128, seed=0):
    q, k, w, starts, ends, cu_seqlens, max_seqlen = _packed_inputs(
        sequence_lengths, cp_size, cp_rank, n_heads=n_heads, head_dim=head_dim, seed=seed
    )
    return (
        q.to("cuda", torch.bfloat16),
        k.to("cuda", torch.bfloat16),
        w.to("cuda", torch.bfloat16),
        starts.cuda(),
        ends.cuda(),
        cu_seqlens.cuda(),
        max_seqlen,
    )


GPU_PACKED_LAYOUTS = [
    pytest.param([3000, 5192], 1, 0, id="cp1"),
    pytest.param([12288, 4096], 4, 0, id="cp4-rank0"),
    pytest.param([12288, 4096], 4, 3, id="cp4-rank3"),
    pytest.param([61440, 4096], 8, 5, id="cp8-rank5-long-back"),
    # Score widths that are not a multiple of 4 make the frontend return a padded view.
    pytest.param([3000, 5193], 1, 0, id="cp1-padded-width"),
    pytest.param([4104, 2056], 2, 1, id="cp2-rank1-padded-width"),
]


@pytest.mark.parametrize("sequence_lengths,cp_size,cp_rank", GPU_PACKED_LAYOUTS)
@pytest.mark.parametrize("return_topk_scores", [False, True])
def test_real_frontend_packed_thd_row_chunks_are_bitwise_identical(
    sequence_lengths, cp_size, cp_rank, return_topk_scores
):
    _real_frontend_or_skip()
    inputs = _cuda_packed_inputs(sequence_lengths, cp_size, cp_rank)
    local_rows = inputs[0].size(1)
    max_score_k = max(sequence_lengths)
    # About five launches, with window boundaries inside segments.
    max_bytes = (local_rows // 5 + 3) * max_score_k * 4

    single = _run_packed_thd(inputs, cp_size, cp_rank, return_topk_scores, deterministic=True)
    chunked = _run_packed_thd(
        inputs,
        cp_size,
        cp_rank,
        return_topk_scores,
        deterministic=True,
        score_chunk_max_bytes=max_bytes,
    )
    torch.cuda.synchronize()

    # Selected sets are canonical under deterministic mode; slot order is unspecified.
    single_indices, single_order = single[0].sort(dim=-1)
    chunked_indices, chunked_order = chunked[0].sort(dim=-1)
    torch.testing.assert_close(chunked_indices, single_indices, rtol=0, atol=0)
    if return_topk_scores:
        torch.testing.assert_close(
            chunked[1].gather(-1, chunked_order), single[1].gather(-1, single_order), rtol=0, atol=0
        )


@pytest.mark.parametrize("cp_rank", [0, 3])
def test_real_frontend_single_sequence_segments_match_reference(cp_rank):
    _real_frontend_or_skip()
    cp_size, sequence_length, topk = 4, 8192, 64
    q, k, w, starts, ends, _cu, _max = _cuda_packed_inputs([sequence_length], cp_size, cp_rank)

    topk_indices, topk_values = dsa_cudnn_kernels._indexer_topk_single_packed_cp_segments(
        q, k, w, topk, True, cp_size, cp_rank, 0, q.size(1), score_chunk_max_bytes=1 << 20
    )

    rows = torch.arange(0, q.size(1), 97, device="cuda")
    selected_rows = topk_indices[0, rows]
    reference_indices, _ = _reference_topk(
        q[0, rows], k[0, :, 0], w[0, rows], starts[rows].cpu(), ends[rows].cpu(), topk
    )
    overlap = 0
    for row, expected in enumerate(reference_indices):
        got = set(selected_rows[row][selected_rows[row] >= 0].tolist())
        overlap += len(got & set(expected)) / len(expected)
    # BF16 inputs with FP32 accumulation; only near-cutoff reorderings may differ.
    assert overlap / len(reference_indices) > 0.99


def test_real_frontend_full_fusion_with_score_bound_matches_unbounded():
    """The single-pass fused function accepts the bound and returns the same gradients."""
    _real_frontend_or_skip()
    try:
        dsa_cudnn_kernels._ensure_flash_mla()
    except ImportError as error:
        pytest.skip(f"FlashMLA unavailable: {error}")
    torch.manual_seed(1234)
    sq, heads, attn_dim, latent_v, idx_heads, idx_dim = 4096, 64, 576, 512, 64, 128
    inputs = {
        "query": torch.randn((sq, 1, heads, attn_dim), device="cuda", dtype=torch.bfloat16),
        "kv_full": torch.randn((sq, 1, attn_dim), device="cuda", dtype=torch.bfloat16),
        "q_indexer": torch.randn((sq, 1, idx_heads, idx_dim), device="cuda", dtype=torch.bfloat16),
        "k_indexer": torch.randn((sq, 1, idx_dim), device="cuda", dtype=torch.bfloat16),
        "weights": torch.randn((sq, 1, idx_heads), device="cuda", dtype=torch.bfloat16),
    }
    grad_output = None
    results = []
    for bound in (None, 1 << 22):
        leaves = {name: t.clone().requires_grad_(True) for name, t in inputs.items()}
        extra = {} if bound is None else {"indexer_score_chunk_max_bytes": bound}
        output, indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
            leaves["query"],
            leaves["kv_full"],
            leaves["q_indexer"],
            leaves["k_indexer"],
            leaves["weights"],
            indexer_topk=2048,
            softmax_scale=attn_dim**-0.5,
            loss_coeff=0.01,
            sparse_loss=True,
            d_v=latent_v,
            **extra,
        )
        if grad_output is None:
            grad_output = torch.randn_like(output)
        (output.float().mul(grad_output.float()).sum() + indexer_loss).backward()
        results.append((output.detach(), {n: t.grad for n, t in leaves.items()}))

    (output_ref, grads_ref), (output_bounded, grads_bounded) = results
    torch.testing.assert_close(output_bounded, output_ref)
    for name, grad in grads_ref.items():
        torch.testing.assert_close(grads_bounded[name], grad, msg=f"grad {name}")
