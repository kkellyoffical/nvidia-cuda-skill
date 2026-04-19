# nvidia-cuda v0.2.0

Second public release of the `nvidia-cuda` skill package.

## Highlights

- upgrades the repository from text-only guidance to guidance plus reusable tooling
- adds a CUDA environment probe for fast machine inspection
- adds a heuristic training-stack scanner for `.item()`, deprecated AMP, and dataloader configuration issues
- adds attention, training-step, and dataloader benchmarks to help measure underutilization instead of guessing
- keeps the package publishable as a ClawHub-compatible skill while making it more useful as a standalone open-source repository

## New scripts

- `scripts/cuda_env_probe.py`
- `scripts/check_training_stack.py`
- `scripts/benchmark_attention.py`
- `scripts/training_step_benchmark.py`
- `scripts/dataloader_benchmark.py`

## Release notes

This release still focuses on zero-dependency skill packaging and standard-library or PyTorch-only tooling. It does not add an executable framework library or third-party benchmarking dependency.
