"""Exact selected logprobs from a linear head with bounded vocabulary tiles.

The reference is ``linear(x, weight).float().log_softmax(-1).gather(...)``.
Logits and their gradients round to the input dtype, as in a regular linear
layer. No probabilities or gradients are filtered. First-order autograd is
supported for either or both inputs; labels are valid vocabulary indices.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

_VOCAB_TILE = 2048


def linear_selected_logprobs(x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return FP32 logprobs for CUDA FP32/BF16 ``[N,H] @ [V,H].T``.

    Non-contiguous inputs are copied before launch. There is no bias, TP,
    logit softcap, quantization, or higher-order derivative in this primitive.
    Unsupported inputs raise rather than silently changing implementations.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("CCE expects x [N,H] and weight [V,H] with matching H")
    if labels.shape != (x.shape[0],) or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("CCE labels must be int32/int64 with shape [N]")
    if x.shape[1] == 0 or weight.shape[0] == 0:
        raise ValueError("CCE requires positive hidden and vocabulary sizes")
    if x.device.type != "cuda" or weight.device != x.device or labels.device != x.device:
        raise ValueError("CCE inputs must be on the same CUDA device")
    if x.dtype not in (torch.float32, torch.bfloat16) or weight.dtype != x.dtype:
        raise ValueError("CCE supports matching FP32 or BF16 inputs")
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        raise ValueError("CCE requires CUDA compute capability 8.0 or newer")
    # Keep this device-side to avoid synchronizing every training step on .item().
    torch._assert_async(((labels >= 0) & (labels < weight.shape[0])).all(), "CCE label out of range")
    return _LinearSelectedLogprobs.apply(x.contiguous(), weight.contiguous(), labels.contiguous())


class _LinearSelectedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, labels):
        import triton

        from areno.accel.kernels.cce import merge_lse, project_reduce

        n, h = x.shape
        v = weight.shape[0]
        tiles = triton.cdiv(v, 128)
        lse = torch.empty(n, device=x.device, dtype=torch.float32)
        output = torch.empty_like(lse)
        if n:
            partial = torch.empty((n, tiles), device=x.device, dtype=torch.float32)
            target = torch.empty_like(lse)
            with torch.cuda.device(x.device):
                project_reduce[(triton.cdiv(n, 32), tiles)](
                    x,
                    weight,
                    labels,
                    partial,
                    target,
                    lse,
                    output,
                    output,
                    n,
                    h,
                    v,
                    0,
                    v,
                    tiles,
                    False,
                )
                merge_lse[(n,)](partial, target, lse, output, n, tiles, triton.next_power_of_2(tiles))
        ctx.save_for_backward(x, weight, labels, lse)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        import triton

        from areno.accel.kernels.cce import project_reduce

        x, weight, labels, lse = ctx.saved_tensors
        n, h = x.shape
        v = weight.shape[0]
        need_x, need_w = ctx.needs_input_grad[:2]
        # Accumulate the small hidden gradient in FP32. The weight gradient
        # is written one vocab tile at a time, without a second full copy.
        dx = torch.zeros(x.shape, device=x.device, dtype=torch.float32) if need_x else None
        dw = torch.zeros_like(weight) if need_w else None
        upstream = upstream.contiguous()
        if n and (need_x or need_w):
            with torch.cuda.device(x.device):
                for start in range(0, v, _VOCAB_TILE):
                    width = min(_VOCAB_TILE, v - start)
                    grad = torch.empty((n, width), device=x.device, dtype=x.dtype)
                    project_reduce[(triton.cdiv(n, 32), triton.cdiv(width, 128))](
                        x,
                        weight,
                        labels,
                        lse,
                        lse,
                        lse,
                        upstream,
                        grad,
                        n,
                        h,
                        v,
                        start,
                        width,
                        0,
                        True,
                    )
                    if need_x:
                        dx.add_(torch.mm(grad, weight[start : start + width]).float())
                    if need_w:
                        torch.mm(grad.T, x, out=dw[start : start + width])
        return dx.to(x.dtype) if need_x else None, dw, None
