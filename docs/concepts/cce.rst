Fused vocabulary projection (CCE)
=================================

``ARENO_CCE=1`` requests a tiled vocabulary projection for packed Qwen3 dense
training on one CUDA GPU with compute capability 8.0 or newer. It supports
matching FP32 or BF16 hidden states and head weights, and is disabled by
default. Because the normal ``compile_model`` default is ``True``, eager CCE
training must also set ``runtime={"compile_model": False}``; otherwise the
dispatch guard retains the standard LM-head path. Set both before starting the
existing CLI or public ``Trainer`` process.

Validated execution mode
-------------------------

CCE is validated only for eager training. ``RuntimeConfig.compile_model``
defaults to ``True`` for ordinary AReno runs; the CCE dispatch guard therefore
keeps the standard LM-head path when compilation is enabled. Set
``runtime={"compile_model": False}`` explicitly when enabling CCE. For the
public API, the smallest configuration is:

.. code-block:: python

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

This is an opt-in memory path, not a promise of faster execution. Compiled
training remains outside the validated CCE contract; ``ARENO_CCE=1`` alone
does not opt a compiled model into CCE.
The dispatch guard is covered by the CPU dispatch tests.

The training worker defers the ordinary LM head, selects next-token prediction
sites without crossing packed sequence boundaries, and computes selected
logprobs directly. Existing loss functions retain ownership of masks, reductions,
and upstream gradients. Rollout and standalone scoring retain their ordinary
head paths.

Numerical contract
------------------

For hidden states ``X[N,H]``, weights ``W[V,H]``, and labels ``y[N]``, the
reference is:

.. code-block:: python

   logits = torch.nn.functional.linear(X, W).float()
   logprobs = logits.log_softmax(-1).gather(-1, y[:, None]).squeeze(-1)

The forward kernel projects 32 tokens by 128 vocabulary entries at a time. It
rounds logits to the input dtype, computes tile log-sum-exp values in FP32, and
merges those values into a global log-sum-exp per token. Subtracting that value
from the selected logit returns FP32 logprobs. No probabilities or gradients are
filtered or approximated by a cutoff.

Backward recomputes logits in vocabulary tiles of at most 2048 entries. For
arbitrary upstream gradient ``g``, it forms
``G = g * (one_hot(y) - exp(logits - logsumexp))``, rounds ``G`` to the input
dtype, and computes ``dX = G @ W`` and ``dW = G.T @ X``. Hidden-gradient tile
contributions accumulate in FP32 before casting to the input dtype. Either input
may be frozen. A tied embedding and head receive both gradient contributions
through the same Parameter.

Floating-point reduction order differs from the reference. FP32 checks use
``atol=rtol=1e-5`` for logprobs and relative gradient L2 error at most ``1e-4``;
BF16 checks use ``1e-2`` and ``2%`` respectively. These are numerical tolerances,
not a promise of identical training trajectories.

Memory and dispatch
-------------------

Forward uses an ``[N, ceil(V/128)]`` FP32 array of tile reduction states instead
of an ``[N,V]`` logits tensor. Backward uses at most ``[N,2048]`` elements of
logit-gradient scratch. Inputs, model weights, their gradients, and optimizer
state still require storage. At small vocabularies up to 2048, the bounded
backward scratch may span the vocabulary.

The integrated path requires ``runtime.compile_model=False``, an unmodified
``VocabParallelLMHead`` with the full vocabulary on one GPU, model type
``qwen3``, packed token data, and a single worker. Other models (including
Qwen3.5), multiple workers, tensor parallelism, CPU/NPU, unified-memory devices,
unsupported dtypes or GPU capability, and wrapped, biased, softcapped, or
quantized heads retain their existing implementation with a warning when CCE
was requested. Kernel execution errors propagate. The primitive supports
non-contiguous inputs by copying them, empty token batches, and first-order
autograd, including CUDA graph capture after warmup. Higher-order derivatives
are outside its contract. CPU offload does not count as a CCE memory reduction.

Reproduction and profiling
--------------------------

``examples/cce/benchmark_head.py`` validates forward/backward, checks allocation
shapes, and measures both paths at ``N=4096``, ``H=1024``, and vocabulary sizes
32768 and 65536 in FP32 and BF16. ``examples/cce/benchmark_trainer.py`` uses the
public ``Trainer`` for fixed SFT updates, weight save/reload, a further update,
and actual rollout followed by native GRPO. Both use two warmup iterations and
ten measured iterations. See the :download:`reproduction guide
<../../examples/cce/README.md>`; write raw artifacts to a caller-selected output
directory and transcribe the resulting measurements into the PR description.
The repository ignores raw logs, TensorBoard events, and checkpoints, so the
reproduction scripts and their commands are the durable part of the submission.

``ARENO_PROFILE_TRAIN=1`` adds synchronized CUDA worker timing and allocator
peaks as ``train_worker_time_s``, ``cuda_peak_allocated_bytes``,
``cuda_peak_reserved_bytes``, and ``cce_enabled`` metrics. Each measurement
covers one worker microbatch through backward and, when scheduled, its optimizer
update. With gradient accumulation, these are microbatch measurements; do not
interpret their averaged values as a complete accumulated-step peak. The supplied
Trainer benchmark uses one microbatch per update and also records public Trainer
step time. Profiling is disabled by default because synchronization affects timing.

The measured head peak is lower for all four synthetic cases. FP32 head execution
is slower; BF16 head execution is also slightly slower on the measured RTX 3090.
The model benchmark has a modest memory reduction, while the reported warm-step
time is a fixed-workload observation rather than an overall speedup claim. New
shapes incur Triton compilation cost, which can dominate their first update.
Weight checkpoints reload exactly in this experiment, but do not include
optimizer, scheduler, or global-step state. The validation covers one CUDA GPU;
it reports no NPU, multi-GPU, or unified-memory system-memory result.
