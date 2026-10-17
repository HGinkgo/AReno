"""Correctness and full forward/backward memory/timing for a linear head.

Run from the repository root, e.g.:
python examples/cce/benchmark_head.py --output head-results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from areno.accel.cce import linear_selected_logprobs


def reference(x, weight, labels):
    return F.linear(x, weight).float().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)


class AllocationShapes(TorchDispatchMode):
    """Observe ATen allocations, including Python-side Triton scratch buffers."""

    def __init__(self, forbidden):
        super().__init__()
        self.forbidden = forbidden
        self.full_logits_seen = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for tensor in result if isinstance(result, (tuple, list)) else [result]:
            if isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == self.forbidden:
                self.full_logits_seen = True
        return result


def measure(fn, x, weight, labels, upstream, warmup, steps):
    samples = []
    for index in range(warmup + steps):
        x.grad = weight.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        output = fn(x, weight, labels)
        output.backward(upstream)
        torch.cuda.synchronize()
        sample = {
            "seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "incremental_allocated_bytes": torch.cuda.max_memory_allocated() - allocated_before,
        }
        del output
        if index >= warmup:
            samples.append(sample)
    x.grad = weight.grad = None
    return {
        "samples": samples,
        "median_seconds": statistics.median(row["seconds"] for row in samples),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in samples),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in samples),
        "peak_incremental_bytes": max(row["incremental_allocated_bytes"] for row in samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    assert args.warmup >= 2 and args.steps >= 10
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seed": 42,
        "warmup": args.warmup,
        "steps": args.steps,
        "tf32": False,
        "bf16_reduced_precision_reduction": False,
        "cases": [],
        "ok": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype in (torch.float32, torch.bfloat16):
        for vocab in (32768, 65536):
            torch.manual_seed(42)
            n, h = 4096, 1024
            x = (torch.randn(n, h, device="cuda", dtype=dtype) / h**0.5).requires_grad_()
            weight = torch.randn(vocab, h, device="cuda", dtype=dtype).requires_grad_()
            labels = torch.randint(vocab, (n,), device="cuda")
            labels[0], labels[-1] = 0, vocab - 1
            upstream = torch.randn(n, device="cuda") / n
            upstream[::5] = 0
            expected = reference(x, weight, labels)
            expected.backward(upstream)
            reference_values = expected.detach().cpu()
            reference_dx, reference_dw = x.grad.cpu(), weight.grad.cpu()
            del expected
            x.grad = weight.grad = None
            with AllocationShapes((n, vocab)) as allocations:
                actual = linear_selected_logprobs(x, weight, labels)
                actual.backward(upstream)
            tolerance, grad_limit = (1e-5, 1e-4) if dtype == torch.float32 else (1e-2, 0.02)
            torch.testing.assert_close(actual.cpu(), reference_values, atol=tolerance, rtol=tolerance)
            errors = {"logprob_max_abs": (actual.cpu() - reference_values).abs().max().item()}
            for name, tensor, expected_gradient in (
                ("hidden", x.grad, reference_dx),
                ("weight", weight.grad, reference_dw),
            ):
                relative = (tensor.float().cpu() - expected_gradient.float()).norm() / expected_gradient.float().norm()
                errors[name + "_gradient_relative_l2"] = relative.item()
                assert relative <= grad_limit, errors
            assert not allocations.full_logits_seen
            del actual, reference_dx, reference_dw
            x.grad = weight.grad = None
            gc.collect()
            torch.cuda.empty_cache()
            baseline = measure(reference, x, weight, labels, upstream, args.warmup, args.steps)
            gc.collect()
            torch.cuda.empty_cache()
            candidate = measure(linear_selected_logprobs, x, weight, labels, upstream, args.warmup, args.steps)
            assert candidate["peak_allocated_bytes"] < baseline["peak_allocated_bytes"]
            case = {
                "shape": [n, h, vocab],
                "dtype": str(dtype),
                "errors": errors,
                "full_logits_allocation_seen": allocations.full_logits_seen,
                "reference": baseline,
                "cce": candidate,
            }
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(case), flush=True)
            del x, weight, labels, upstream
            gc.collect()
            torch.cuda.empty_cache()
    report["ok"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
