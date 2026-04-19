# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [0.1.0] - 2026-04-19

### Added

- initial `nvidia-cuda` skill package for NVIDIA deep learning optimization and review
- BF16, FP8, TF32, and mixed precision policy guidance for H100, H200, and B200-class hardware
- `torch.compile` and CUDA Graphs decision rules
- attention backend policy covering SDPA, FA3/FA4 registration, cuDNN attention, Transformer Engine, and varlen attention
- large-model training guidance for FSDP2, ZeRO-style sharding, distributed optimizer, TP, SP, CP, PP, and EP
- LLM serving guidance for paged KV, chunked prefill, inflight batching, speculative decoding, and TensorRT-LLM
- hard anti-pattern rules for per-step `.item()`, untuned dataloaders, and premature checkpointing
- official reference notes with PyTorch and NVIDIA documentation links
- ClawHub-ready project layout with root `SKILL.md`
