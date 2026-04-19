# nvidia-cuda v0.3.0

Third public release of the `nvidia-cuda` skill package.

## Highlights

- adds a current NVIDIA GPU recommendation matrix as of 2026-04-19
- adds sample planning and environment configs for workstation, single-node training, rack-scale serving, and edge inference
- adds a runtime support note so users can see the intended Python, PyTorch, and CUDA surface
- adds a minimal NCCL smoke test and a basic GitHub Actions smoke workflow
- keeps the package compatible with GitHub and ClawHub publication

## New recommendation coverage

- RTX 5090
- RTX PRO 6000 Blackwell Workstation Edition
- DGX Station
- RTX PRO 6000 Blackwell Server Edition
- H200
- DGX B300
- DGX B200
- GB200 NVL72
- GB300 NVL72
- L4

## New files

- `references/latest-gpu-recommendations-2026-04.md`
- `references/runtime-support.md`
- `examples/dgx-b300-training.yaml`
- `examples/l4-edge-serving.yaml`
- `scripts/nccl_smoke.py`
- `.github/workflows/smoke.yml`

## Remaining gaps

- the scanner is still Python-focused and does not yet inspect Triton, C++, or TensorRT-LLM config surfaces
- the attention benchmark still does not directly exercise FA3/FA4 registration workflows
- H200/B200-only paths are documented from official sources but not locally benchmarked on this H100 host
