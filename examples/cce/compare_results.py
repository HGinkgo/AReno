"""Check saved CCE benchmarks and TensorBoard extracts; fail on regressions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def compare(root):
    head = read(root / "head.json")
    assert head["ok"] and head["warmup"] >= 2 and head["steps"] >= 10
    expected_cases = {
        ("torch.float32", (4096, 1024, 32768)),
        ("torch.float32", (4096, 1024, 65536)),
        ("torch.bfloat16", (4096, 1024, 32768)),
        ("torch.bfloat16", (4096, 1024, 65536)),
    }
    tolerances = {
        "torch.float32": (1e-5, 1e-4),
        "torch.bfloat16": (1e-2, 0.02),
    }
    seen_cases = set()
    assert len(head["cases"]) >= 4
    for case in head["cases"]:
        case_id = (case["dtype"], tuple(case["shape"]))
        # Keep the four Issue-588 cases mandatory while allowing future
        # benchmark shapes to be added without breaking this check.
        if case_id not in expected_cases:
            continue
        assert len(case_id[1]) == 3 and all(dimension > 0 for dimension in case_id[1])
        assert case_id not in seen_cases, case_id
        seen_cases.add(case_id)
        logprob_tolerance, gradient_tolerance = tolerances[case["dtype"]]
        assert case["errors"]["logprob_max_abs"] <= logprob_tolerance
        for key in ("hidden_gradient_relative_l2", "weight_gradient_relative_l2"):
            assert case["errors"][key] <= gradient_tolerance
        assert not case["full_logits_allocation_seen"]
        assert case["cce"]["peak_allocated_bytes"] < case["reference"]["peak_allocated_bytes"]
        for name in ("reference", "cce"):
            assert len(case[name]["samples"]) == head["steps"]
    assert expected_cases <= seen_cases

    reference, candidate = [read(root / name / "result.json") for name in ("reference", "candidate")]
    for key in (
        "model",
        "seed",
        "dtype",
        "world_size",
        "tp",
        "dp",
        "runtime",
        "optimizer",
        "warmup",
        "steps",
        "length",
        "batch",
    ):
        assert reference[key] == candidate[key], key
    assert reference["runtime"]["keep_rollout_state"]
    assert reference["runtime"]["optimizer_state_offload"] == "none"
    assert reference["warmup"] >= 2 and reference["steps"] >= 10
    assert reference["cce"] == "0" and candidate["cce"] == "1"
    assert read(root / "reference" / "normalized-sequences.json") == read(
        root / "candidate" / "normalized-sequences.json"
    )
    initial = [torch.tensor(read(root / name / "first-train-logprobs.json")) for name in ("reference", "candidate")]
    torch.testing.assert_close(initial[0], initial[1], atol=1e-2, rtol=1e-2)
    native_metrics = {}
    losses = []
    for name, report in (("reference", reference), ("candidate", candidate)):
        assert report["ok"] and report["reload_difference"] == {"max_abs": 0.0, "mean_abs": 0.0}
        assert len(report["samples"]) == report["warmup"] + report["steps"]
        assert all(row["cce_enabled"] == int(report["cce"]) for row in report["samples"])
        assert all(math.isfinite(row["loss"]) and row["grad_norm"] > 0 for row in report["samples"])
        assert report["continued_step"]["grad_norm"] > 0
        train, reloaded = [read(root / name / (stage + "-metrics.json")) for stage in ("train", "reloaded")]
        assert train["ok"] and reloaded["ok"]
        series = train["series"]["train/loss"]
        assert [row["step"] for row in series] == list(range(report["warmup"] + report["steps"]))
        values = torch.tensor([row["value"] for row in series])
        torch.testing.assert_close(values, torch.tensor([row["loss"] for row in report["samples"]]))
        losses.append(values)
        metric = reloaded["series"]["train/logp_abs_diff_mean"]
        assert len(metric) == 1 and metric[0]["step"] == 1
        value = metric[0]["value"]
        assert math.isfinite(value) and 0 <= value < 1e-2
        assert value == report["grpo_step"]["logp_abs_diff_mean"]
        native_metrics[name] = value
    # This bounded smoke trajectory also stays inside the BF16 loss tolerance;
    # it does not establish bitwise agreement or training quality.
    torch.testing.assert_close(losses[0], losses[1], atol=1e-2, rtol=1e-2)
    assert candidate["peak_allocated_bytes"] < reference["peak_allocated_bytes"]
    return {
        "ok": True,
        "initial_logprob_max_abs": (initial[0] - initial[1]).abs().max().item(),
        "initial_logprob_mean_abs": (initial[0] - initial[1]).abs().mean().item(),
        "sft_loss_max_abs": (losses[0] - losses[1]).abs().max().item(),
        "native_logp_abs_diff_mean": native_metrics,
        "allocated_peak_reduction_fraction": 1 - candidate["peak_allocated_bytes"] / reference["peak_allocated_bytes"],
        "reserved_peak_reduction_fraction": 1 - candidate["peak_reserved_bytes"] / reference["peak_reserved_bytes"],
        "median_step_time_change_fraction": candidate["median_step_seconds"] / reference["median_step_seconds"] - 1,
        "checkpoint_scope": "model weights; fresh optimizer on reload",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(compare(args.results), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
