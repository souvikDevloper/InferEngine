from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Laptop/test installations intentionally omit Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_qkv_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        rows: tl.constexpr,
        inner: tl.constexpr,
        columns: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, inner, BLOCK_K):
            x = tl.load(
                x_ptr + offsets_m[:, None] * inner + (start + offsets_k[None, :]),
                mask=(offsets_m[:, None] < rows) & (start + offsets_k[None, :] < inner),
                other=0.0,
            )
            weight = tl.load(
                weight_ptr + (start + offsets_k[:, None]) * columns + offsets_n[None, :],
                mask=(start + offsets_k[:, None] < inner) & (offsets_n[None, :] < columns),
                other=0.0,
            )
            accumulator += tl.dot(x, weight)
        tl.store(
            output_ptr + offsets_m[:, None] * columns + offsets_n[None, :],
            accumulator,
            mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < columns),
        )


def pack_qkv_weights(q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor) -> torch.Tensor:
    """Pack PyTorch Linear weights once as [input, Q+K+V] for one launch."""
    if q_weight.ndim != 2 or k_weight.ndim != 2 or v_weight.ndim != 2:
        raise ValueError("Q, K, and V weights must be matrices")
    if len({q_weight.shape[1], k_weight.shape[1], v_weight.shape[1]}) != 1:
        raise ValueError("Q, K, and V input dimensions must match")
    return torch.cat((q_weight, k_weight, v_weight), dim=0).transpose(0, 1).contiguous()


def fused_qkv(x: torch.Tensor, packed_weight: torch.Tensor, split_sizes: tuple[int, int, int]):
    """Compute Q, K, and V projections with one Triton matmul kernel launch."""
    if triton is None:
        raise RuntimeError("Triton is not installed; use the gpu environment")
    if not x.is_cuda or not packed_weight.is_cuda:
        raise ValueError("fused_qkv requires CUDA tensors")
    if x.ndim != 2 or packed_weight.ndim != 2 or x.shape[1] != packed_weight.shape[0]:
        raise ValueError("expected x=[tokens, hidden] and weight=[hidden, q+k+v]")
    if sum(split_sizes) != packed_weight.shape[1]:
        raise ValueError("split sizes do not match packed output width")
    rows, inner = x.shape
    columns = packed_weight.shape[1]
    output = torch.empty((rows, columns), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(rows, 32), triton.cdiv(columns, 64))
    _fused_qkv_kernel[grid](
        x,
        packed_weight,
        output,
        rows=rows,
        inner=inner,
        columns=columns,
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return output.split(split_sizes, dim=-1)
