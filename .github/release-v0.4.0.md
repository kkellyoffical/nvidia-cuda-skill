# nvidia-cuda v0.4.0

Fourth public release of the `nvidia-cuda` skill package.

## Highlights

- adds a DDP/FSDP smoke-test helper for distributed training validation
- expands the training-stack scanner beyond Python-only coverage into Triton, CUDA, and C++ review surfaces
- extends the attention benchmark with official flash implementation listing and activation hooks
- keeps all new guidance grounded in official PyTorch, Triton, and NVIDIA sources

## New files

- `scripts/ddp_fsdp_smoke.py`

## Updated files

- `scripts/benchmark_attention.py`
- `scripts/check_training_stack.py`
- `references/official-notes.md`
- `README.md`
- `SKILL.md`
- `CHANGELOG.md`

## Remaining gaps

- FA3/FA4 activation still depends on the local provider module actually being installed
- the scanner remains heuristic and does not perform full semantic analysis of Triton or CUDA kernels
- profiler capture scaffolding is still documentation-first rather than a dedicated script
