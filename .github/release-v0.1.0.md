# nvidia-cuda v0.1.0

Initial public release of the `nvidia-cuda` skill package.

## Highlights

- establishes a concrete optimization order for NVIDIA GPU deep learning work
- bans common underutilization patterns such as full FP32-by-default, per-step `.item()`, and untuned dataloaders
- treats `torch.compile` as a benchmarkable optimization step instead of a blanket rejection
- requires attention backend evaluation on H200/B200 rather than stopping at default SDPA
- prioritizes FSDP and ZeRO-style sharding before gradient checkpointing when state replication is the real memory problem
- adds frontier guidance for Transformer Engine FP8, paged KV, chunked prefill, inflight batching, and speculative decoding

## Included files

- `SKILL.md`
- `references/official-notes.md`
- `agents/openai.yaml`
- `README.md`
- `CHANGELOG.md`

## Release notes

This is a documentation and instruction release. There is no executable library in this repository yet.
