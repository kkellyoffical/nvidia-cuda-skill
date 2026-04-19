# nvidia-cuda

`nvidia-cuda` is an open-source skill package for Codex, OpenClaw, and similar agent systems that need to write or review deep learning code for NVIDIA GPUs.

It is opinionated on purpose. The package encodes modern AI infra practice for H100, H200, and B200-class systems:

- BF16 and FP8-aware precision policy
- `torch.compile` as a benchmarked option, not a taboo
- attention backend selection that does not stop at default SDPA
- FSDP and ZeRO before using gradient checkpointing as a crutch
- `.item()` and host sync anti-pattern detection
- dataloader, pinning, prefetch, and persistent worker checks
- NCCL, TP/SP/CP/EP, TensorRT-LLM, speculative decoding, and TE guidance

## Why this exists

A lot of GPU training code still leaves major performance on the floor because it inherits conservative defaults:

- full FP32 everywhere
- no compile pass
- no attention backend evaluation on newer hardware
- per-step host synchronization for metrics
- checkpointing used before state sharding
- dataloaders that starve the GPU

Those mistakes compound. This skill gives an agent a clear optimization order and a concrete review checklist.

## What is inside

- [SKILL.md](SKILL.md): the main operational policy
- [references/official-notes.md](references/official-notes.md): official PyTorch and NVIDIA references
- [agents/openai.yaml](agents/openai.yaml): optional UI metadata for skill-aware clients

## Install locally

### Codex

Copy this directory to your skills path as `~/.codex/skills/nvidia-cuda/`.

### OpenClaw / ClawHub-compatible agents

The repository root is already a skill package:

- `SKILL.md` at the root
- supporting files under `references/` and `agents/`

That means you can publish this repository directly as a skill package after review.

## Triggering

Use the skill by name:

```text
$nvidia-cuda optimize this H100 training loop
```

Example prompts:

- `$nvidia-cuda review this multi-GPU training stack for H200 underutilization`
- `$nvidia-cuda optimize this transformer inference path on B200`
- `$nvidia-cuda audit this dataloader, precision policy, and logging loop`

## Hardware and framework stance

The skill is most opinionated for:

- H100 / Hopper
- H200
- B200 / Blackwell-class GPUs

The primary software target is PyTorch on CUDA, with side coverage for:

- NCCL
- Triton
- Transformer Engine
- TensorRT-LLM
- Megatron Core style large-model sharding

## Release policy

This repository uses semver. The first public release is `0.1.0`.

Version `0.1.0` establishes:

- the initial optimization doctrine
- the anti-pattern blocklist
- H100/H200/B200 attention and precision guidance
- training and inference optimization ladders
- ClawHub-ready repository packaging

## ClawHub publication note

ClawHub supports publishing skill packages from a local path via its CLI.

At the time this repository was prepared:

- the upstream ClawHub project documents `clawhub skill publish <path>`
- the ClawHub web UI also supports signed-in publication

Because community skill marketplaces can carry supply-chain risk, publish only reviewed, minimal, instruction-focused packages and inspect any scanner output before making a listing public.

## License

[MIT-0](LICENSE)
