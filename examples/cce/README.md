# CCE reproduction

This implements [Issue #588](https://github.com/inclusionAI/AReno/issues/588)
for packed Qwen3 dense training on one CUDA SM80+ GPU. Enable it with
`ARENO_CCE=1` before starting the existing public `Trainer` or CLI. It is off
by default. CCE is validated only with eager training, so set
`runtime={"compile_model": False}` in the CUDA configuration as well. With the
default `compile_model=True`, the dispatch guard retains the standard head path
and emits the unsupported-path warning. See [the implementation contract](../../docs/concepts/cce.rst).
Benchmark outputs are written to a caller-selected output directory. The PR
description should transcribe the resulting measurements; raw logs and
checkpoints remain local because the repository ignores them.

Use an AReno environment with its native extension built, PyTorch, Triton,
ModelScope, TensorBoard, and pytest installed. These scripts add no dependencies.
The recorded run used native attention, so flash-attn was not needed. Activate
that environment and run the following commands from the repository root.

## Model and input

Set `MODEL_DIR` to a local directory for the ModelScope Qwen3-0.6B snapshot,
and export it. Download the immutable revision used by the report:

```bash
python - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen3-0.6B",
    revision="09b42cad3d112e832108974449ccb5e8e0f5b5d1",
    local_dir=os.environ["MODEL_DIR"],
)
PY
```

The Trainer script creates two fixed text sequences of 1024 tokens with eight
prompt tokens each. It writes raw text inspection and normalized tokens before
training and rejects invalid lengths or token IDs. This synthetic corpus tests
the training path; its loss and toy rewards are not model-quality evaluations.

## Enable CCE from the public SDK

The feature is selected by the environment variable and the existing
`CudaConfig` runtime dictionary; no new CLI flag is required. This is a
minimal initialization example (the normal `Trainer.train(...)` calls follow
`init()`):

```python
import os
from areno import Trainer
from areno.api import CUDA, CudaConfig

os.environ["ARENO_CCE"] = "1"
trainer = Trainer(
    world_size=1,
    model_path="/path/to/Qwen3-checkpoint",
    backend_type=CUDA,
    custom_config=CudaConfig(
        tp_size=1,
        dp_size=1,
        devices=[0],
        runtime={"compile_model": False},
    ),
)
trainer.init()
```

The same settings are used by `benchmark_trainer.py`; its command below is a
copy-pasteable end-to-end update/save/reload/rollout check.

## Tests and head benchmark

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ARENO_CHECK_TOKEN_IDS=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest tests/ -k cpu -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ARENO_CHECK_TOKEN_IDS=1 CUDA_VISIBLE_DEVICES=0 \
  python -m pytest tests/test_cce_gpu.py tests/test_fp32_master_optimizer_cuda.py -q

mkdir -p .cce-run
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python examples/cce/benchmark_head.py --output .cce-run/head.json \
  --warmup 2 --steps 10 > .cce-run/head.log 2>&1
```

The head benchmark uses a separate full-linear/log-softmax reference, fixed seed
42, arbitrary upstream gradients including masked entries, and both trainable
inputs. It asserts the issue's dtype-specific forward and gradient tolerances,
absence of a full `[N,V]` ATen allocation in CCE, and lower allocated peak memory.
CUDA synchronization brackets each complete head forward/backward measurement.
Allocated and reserved peaks, incremental allocations, and every timing sample
are saved. The shape observer covers Python-side scratch allocations; Triton
tiles stay within a program and do not allocate global logits arrays.

## Public Trainer comparison

Use independent processes and output directories. The recorded reference was
the historical commit `8f4ab26499197baa26e7e21505a039809b8e3f19` with CCE
disabled; the historical candidate was
`138dbb00a3f4e4c27e8be267ce0c4c6fd7c06ebf` with CCE enabled. These are
experiment snapshots, not the current submission commit. Their runtime sources
were held constant while the stale test fixtures differed, so the comparison
changes only `ARENO_CCE`.

```bash
git worktree add --detach ../cce-reference 8f4ab26499197baa26e7e21505a039809b8e3f19
git worktree add --detach ../cce-candidate 138dbb00a3f4e4c27e8be267ce0c4c6fd7c06ebf
RESULTS_DIR="$PWD/.cce-run"
mkdir -p "$RESULTS_DIR"
```

Build the native extension in each worktree with
`python -m pip install -e . --no-deps --no-build-isolation` before using it.
An environment with the same already-built compatible extension may instead
provide that extension file to each worktree. Explicit `PYTHONPATH` ensures
worker processes import the intended checkout despite an editable installation.
Run sequentially on an otherwise idle GPU:

```bash
(
  cd ../cce-reference
  PYTHONPATH="$PWD" ARENO_CCE=0 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python examples/cce/benchmark_trainer.py --model "$MODEL_DIR" \
    --output "$RESULTS_DIR/reference" --length 1024 --batch 2 --warmup 2 --steps 10 \
    > "$RESULTS_DIR/reference.log" 2>&1
)
(
  cd ../cce-candidate
  PYTHONPATH="$PWD" ARENO_CCE=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python examples/cce/benchmark_trainer.py --model "$MODEL_DIR" \
    --output "$RESULTS_DIR/candidate" --length 1024 --batch 2 --warmup 2 --steps 10 \
    > "$RESULTS_DIR/candidate.log" 2>&1
)
```

The run uses full parameters, BF16, default FP32-master AdamW, lr `1e-5`, clip
`1`, native attention, activation checkpointing, and world/TP/DP=1. Compilation,
quantization, and optimizer offload are disabled. `keep_rollout_state=True` is
essential here: `False` implicitly enables CPU optimizer-state offload in the
current engine even if the explicit offload option is `none`.

Each path performs two warmup and ten measured SFT optimizer updates, saves model
weights, closes the Trainer, and loads those weights in a new Trainer. It checks
fixed-input logprobs before/after reload, performs another SFT update with a fresh
optimizer, samples two actual 16-token responses, and calls native GRPO with
advantages -1/+1 and toy rewards 0/1. Native `train/logp_abs_diff_mean` therefore
compares rollout and training on the same current policy. It is distinct from
the saved-weight reload check. Balanced advantages can give zero GRPO loss while
the recorded parameter gradient is nonzero.

The supplied script automatically enables `ARENO_PROFILE_TRAIN=1`. Its single
microbatch per update makes worker allocator peaks cover forward, backward,
gradient processing, and the optimizer update. Public Trainer step time also
includes packing and IPC. Peaks are PyTorch allocator statistics, not whole
device utilization; no model/optimizer offload is used. Initialization, saving,
reloading, rollout, and the final GRPO update are outside the ten measured SFT
samples. New CCE shapes may incur substantial JIT compilation latency.

## Read actual training metrics and compare

Read the TensorBoard event files using the repository reader, then compare the
earliest logprobs, bounded SFT losses, native GRPO metric, checkpoint reload,
and memory invariants. `compare_results.py` also checks the numerical and memory
thresholds, the CCE/reference `logp_abs_diff_mean` values, and the checkpoint
scope. Both commands return nonzero on failed checks.

```bash
RESULTS_DIR="${RESULTS_DIR:-$PWD/.cce-run}"
for name in reference candidate; do
  for stage in train reloaded; do
    python .agents/skills/areno-run-training/scripts/read_metrics.py \
      "$RESULTS_DIR/$name/$stage" --last 20 \
      > "$RESULTS_DIR/$name/$stage-metrics.json"
  done
done
python examples/cce/compare_results.py "$RESULTS_DIR" --output "$RESULTS_DIR/comparison.json"

# Recheck a local results directory without a GPU:
python examples/cce/compare_results.py "$RESULTS_DIR"
```

For runs longer than the default 12 updates, increase the reader's `--last`
value to include all updates. TensorBoard extracts are authoritative for loss,
reward, and native rollout/train logprob metrics. The report retains raw timing
samples, logs, normalized inputs, model hashes, and event-file provenance.
