#!/usr/bin/env python3
"""Benchmark scaled_dot_product_attention across CUDA backends."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


BACKENDS = {
    "auto": None,
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def backend_context(name: str):
    backend = BACKENDS[name]
    if backend is None:
        return contextlib.nullcontext()
    return sdpa_kernel(backend)


def timed_run(fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start
    avg_ms = start_evt.elapsed_time(end_evt) / iters
    return avg_ms, wall_elapsed / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="auto")
    parser.add_argument("--compile", dest="use_compile", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")

    dtype = DTYPES[args.dtype]
    shape = (args.batch, args.heads, args.seq, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    def run() -> torch.Tensor:
        with backend_context(args.backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)

    fn = torch.compile(run, mode="max-autotune") if args.use_compile else run
    avg_ms, avg_wall_s = timed_run(fn, args.warmup, args.iters)
    tokens_per_s = (args.batch * args.seq) / avg_wall_s

    result: dict[str, Any] = {
        "backend": args.backend,
        "dtype": args.dtype,
        "compile": args.use_compile,
        "batch": args.batch,
        "heads": args.heads,
        "seq": args.seq,
        "head_dim": args.head_dim,
        "causal": args.causal,
        "avg_ms": round(avg_ms, 4),
        "tokens_per_s": round(tokens_per_s, 2),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
