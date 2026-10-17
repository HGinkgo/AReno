"""Tiled vocabulary projection with FP32 log-sum-exp state, without logits IO."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def project_reduce(
    X,
    W,
    Labels,
    Partial,
    Target,
    Lse,
    Upstream,
    Grad,
    N: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    TILES: tl.constexpr,
    BACKWARD: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 32,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    vocab = START + cols
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for k in range(triton.cdiv(H, BK)):
        features = k * BK + ks
        x = tl.load(X + rows[:, None] * H + features[None, :], (rows[:, None] < N) & (features[None, :] < H), 0)
        w = tl.load(W + vocab[None, :] * H + features[:, None], (cols[None, :] < WIDTH) & (features[:, None] < H), 0)
        acc += tl.dot(x, w, input_precision="tf32x3")
    # Match a linear layer's output dtype, including BF16 logits rounding.
    logits = acc.to(X.dtype.element_ty).to(tl.float32)
    labels = tl.load(Labels + rows, rows < N, -1)
    valid = (rows[:, None] < N) & (cols[None, :] < WIDTH)
    if BACKWARD:
        lse = tl.load(Lse + rows, rows < N, 0)
        upstream = tl.load(Upstream + rows, rows < N, 0)
        grad = ((labels[:, None] == vocab[None, :]).to(tl.float32) - tl.exp(logits - lse[:, None])) * upstream[:, None]
        tl.store(Grad + rows[:, None] * WIDTH + cols[None, :], grad, valid)
    else:
        logits = tl.where(cols[None, :] < WIDTH, logits, -float("inf"))
        maximum = tl.max(logits, 1)
        partial_lse = maximum + tl.log(tl.sum(tl.exp(logits - maximum[:, None]), 1))
        tl.store(Partial + rows * TILES + tl.program_id(1), partial_lse, rows < N)
        target = tl.sum(tl.where(labels[:, None] == vocab[None, :], logits, 0), 1)
        owns_label = (labels >= START + tl.program_id(1) * BN) & (
            labels < START + tl.minimum((tl.program_id(1) + 1) * BN, WIDTH)
        )
        tl.store(Target + rows, target, (rows < N) & owns_label)


@triton.jit
def merge_lse(Partial, Target, Lse, Out, N: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    partial = tl.load(Partial + row * TILES + cols, cols < TILES, -float("inf"))
    maximum = tl.max(partial, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(partial - maximum), 0))
    target = tl.load(Target + row)
    tl.store(Lse + row, lse)
    tl.store(Out + row, target - lse)
