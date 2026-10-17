"""Public Trainer: fixed SFT steps, save/reload, then real rollout + GRPO.

Run reference and CCE in separate processes with ARENO_CCE=0 and 1, using
identical arguments and distinct output directories. No optimizer offload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import torch

from areno import Trainer
from areno.api.algorithms import grpo_loss_fn, sft_loss_fn
from areno.api.config import CudaConfig
from areno.api.models import SamplingParams, TrainSequence
from areno.api.tokenizer import load_tokenizer


def capture_sft_loss(pack, logprobs):
    """Retain the earliest meaningful boundary once, in the worker process."""
    destination = Path(os.environ["ARENO_CCE_CAPTURE"])
    if not destination.exists():
        destination.write_text(json.dumps(logprobs.detach().float().cpu().tolist()) + "\n")
    return sft_loss_fn(pack, logprobs)


def differences(left, right):
    a, b = torch.tensor(left), torch.tensor(right)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    return {"max_abs": (a - b).abs().max().item(), "mean_abs": (a - b).abs().mean().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    assert args.warmup >= 2 and args.steps >= 10
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["ARENO_PROFILE_TRAIN"] = "1"
    os.environ["ARENO_CCE_CAPTURE"] = str((args.output / "first-train-logprobs.json").resolve())
    torch.manual_seed(42)
    tokenizer = load_tokenizer(str(args.model))
    corpus = tokenizer.encode("The capital of France is Paris. Two plus two equals four.\n", add_special_tokens=False)
    rows = []
    for index in range(args.batch):
        repeated = corpus[index:] + corpus[:index]
        tokens = (repeated * (args.length // len(repeated) + 1))[: args.length]
        rows.append(TrainSequence(tokens=tokens, prompt_len=8, eos_token_id=tokenizer.eos_token_id))
    # Inspect raw text and normalized data before training; fail on any schema,
    # length, target count, or vocabulary violation rather than trimming data.
    inspection = {
        "raw_text": tokenizer.decode(corpus),
        "lengths": [len(row.tokens) for row in rows],
        "target_tokens": sum(len(row.tokens) - row.prompt_len for row in rows),
        "ok": False,
    }
    assert all(len(row.tokens) == args.length and row.prompt_len < len(row.tokens) for row in rows)
    assert all(0 <= token < tokenizer.vocab_size for row in rows for token in row.tokens)
    inspection["ok"] = True
    (args.output / "data-inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
    (args.output / "normalized-sequences.json").write_text(json.dumps([row.model_dump() for row in rows]) + "\n")
    runtime = {
        "attn_backend": "native",
        "compile_model": False,
        "activation_checkpointing": True,
        "keep_rollout_state": True,
        "optimizer_state_offload": "none",
        "eager_decode": True,
    }
    optimizer = {"lr": 1e-5, "grad_clip_norm": 1.0}
    report = {
        "ok": False,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cce": os.getenv("ARENO_CCE", "0"),
        "model": str(args.model.resolve()),
        "seed": 42,
        "dtype": "bfloat16",
        "world_size": 1,
        "tp": 1,
        "dp": 1,
        "runtime": runtime,
        "optimizer": optimizer,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "steps": args.steps,
        "length": args.length,
        "batch": args.batch,
        "samples": [],
        "checkpoint_semantics": "model weights only; a fresh optimizer after reload",
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def trainer_for(model, phase):
        return Trainer(
            world_size=1,
            model_path=str(model),
            custom_config=CudaConfig(
                tp_size=1, dp_size=1, devices=[0], runtime=runtime, optimizer=optimizer, max_running_prompts=2
            ),
            metrics_log_dir=str(args.output / phase),
        )

    save()
    trainer = trainer_for(args.model, "train")
    try:
        trainer.init()
        for index in range(args.warmup + args.steps):
            metrics = trainer.train(rows, capture_sft_loss, mini_bs=args.batch)
            assert math.isfinite(metrics["loss"])
            assert int(metrics["cce_enabled"]) == int(report["cce"])
            report["samples"].append({"warmup": index < args.warmup, **metrics})
            save()
            print(json.dumps({"step": index, **metrics}), flush=True)
        measured = report["samples"][args.warmup :]
        report["median_step_seconds"] = statistics.median(row["step_train_time_s"] for row in measured)
        report["input_tokens_per_second"] = args.length * args.batch / report["median_step_seconds"]
        report["target_tokens_per_second"] = inspection["target_tokens"] / report["median_step_seconds"]
        report["peak_allocated_bytes"] = max(row["cuda_peak_allocated_bytes"] for row in measured)
        report["peak_reserved_bytes"] = max(row["cuda_peak_reserved_bytes"] for row in measured)
        check_rows = [row.tokens[:64] for row in rows]
        expected = trainer.score_logprobs("actor", check_rows, microbatch_size=args.batch)
        (args.output / "post-train-logprobs.json").write_text(json.dumps(expected) + "\n")
        checkpoint = trainer.save_checkpoint(str(args.output / "checkpoint"))
        save()
    finally:
        trainer.close()

    resumed = trainer_for(checkpoint, "reloaded")
    try:
        resumed.init()
        restored = resumed.score_logprobs("actor", check_rows, microbatch_size=args.batch)
        report["reload_difference"] = differences(expected, restored)
        report["continued_step"] = resumed.train(rows, sft_loss_fn, mini_bs=args.batch)
        # Native rollout logprobs are measured against native training logprobs
        # on the same current policy. Toy rewards only exercise GRPO plumbing.
        prompt = tokenizer.encode("Complete the sequence: one, two, three,", add_special_tokens=False)
        resumed.begin_rollout_session()
        try:
            rollout = resumed.rollout_token_batch(
                [prompt],
                2,
                SamplingParams(seed=42, max_new_tokens=16, temperature=1.0, top_p=1.0, top_k=-1, ignore_eos=True),
            )
        finally:
            resumed.end_rollout_session()
        (args.output / "rollout.json").write_text(json.dumps([item.model_dump() for item in rollout], indent=2) + "\n")
        rl_rows = []
        for index, sequence in enumerate(rollout[0].sequences):
            assert len(sequence.resp_tokens) == len(sequence.resp_logprobs) > 0
            reward = float(index)
            rl_rows.append(
                TrainSequence(
                    tokens=prompt + sequence.resp_tokens,
                    prompt_len=len(prompt),
                    logprobs=[0.0] * len(prompt) + sequence.resp_logprobs,
                    scalar_advantage=2 * reward - 1,
                    reward=reward,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
        report["grpo_step"] = resumed.train(rl_rows, grpo_loss_fn, mini_bs=2)
        assert math.isfinite(report["grpo_step"]["logp_abs_diff_mean"])
        report["ok"] = True
        save()
    finally:
        resumed.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
