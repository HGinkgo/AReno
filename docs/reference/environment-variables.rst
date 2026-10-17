:orphan:

Environment variables
=====================

AReno uses a small number of environment variables for build and runtime
control.

``ARENO_BUILD_EXT``
   Set ``ARENO_BUILD_EXT=0`` to skip CUDA/NPU extension compilation for
   metadata-only installs, docs builds, or CPU-only packaging checks.
   Dependency selection still detects the installed torch_npu package. Use
   ``--no-build-isolation`` so source installs see the existing PyTorch environment.

``TORCH_CUDA_ARCH_LIST``
   Set this when narrowing CUDA extension builds to a target GPU architecture,
   for example ``TORCH_CUDA_ARCH_LIST="9.0"`` for H100/H200-only builds.

``MAX_JOBS``
   Set this to control parallel compilation jobs during editable installs.

``ARENO_CCE``
   Set ``ARENO_CCE=1`` before starting training to enable the opt-in single-GPU
   Qwen3 dense CUDA vocabulary projection. Unsupported configurations emit a
   warning and retain their existing head path. See :doc:`/concepts/cce` for the
   dtype, hardware, autograd, and model constraints. Disabled by default.

``ARENO_PROFILE_TRAIN``
   Set ``ARENO_PROFILE_TRAIN=1`` to record synchronized CUDA worker microbatch
   timing, allocator peaks, and whether CCE was used. Synchronization changes
   timing; profiling is disabled by default. See :doc:`/concepts/cce` for the
   measurement scope and gradient accumulation caveat.

For Ascend, the installed ``torch_npu`` selects NPU dependencies and the native
NPU extension build. Keep the existing PyTorch/torch_npu/CANN versions and use
``--no-build-isolation``. CANN's ``set_env.sh`` supplies the compiler paths.

``ARENO_NPU_SOC``
   Optional exact Ascend SoC override for cross-compilation. With visible NPU
   hardware, CANN detects the SoC automatically; no override is required.

For environment inspection, use :doc:`/cli/diagnostics`.
