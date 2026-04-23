from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from open_titans.modules.attention import (
    AttentionalBias,
    BiasType,
    l2_bias,
    huber_bias,
    lp_bias,
    kl_bias,
)


@dataclass
class BiasTestResult:
    bias_type: str
    device: str
    input_shape: tuple
    output_shape: tuple
    output_non_negative: bool
    gradient_flows: bool
    latency_ms: float
    peak_memory_mb: Optional[float]


def _reset_memory(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return None


def run_correctness_check(device: torch.device):
    print(f"\n── Correctness [{device}] ──────────────────────────────────────────")

    batch, seq_len, dim = 2, 16, 64
    pred = torch.randn(batch, seq_len, dim, device=device)
    target = torch.randn(batch, seq_len, dim, device=device)

    test_cases = [
        ("l2", {}),
        ("huber", {"delta": 1.0}),
        ("lp", {"p": 3.0}),
        ("kl", {}),
    ]

    for bias_name, kwargs in test_cases:
        module = AttentionalBias(bias_type=bias_name, **kwargs).to(device)
        out = module(pred, target)

        assert out.shape == (batch, seq_len), (
            f"[{bias_name}] shape mismatch: {out.shape} != {(batch, seq_len)}"
        )
        assert torch.isfinite(out).all(), f"[{bias_name}] non-finite values"
        assert (out >= 0).all(), f"[{bias_name}] negative loss values"
        print(f"  [{bias_name}] shape={out.shape} min={out.min():.4f} max={out.max():.4f} ✓")

    print(f"  [correctness/{device}] all bias types passed ✓")


def run_gradient_check(device: torch.device):
    print(f"\n── Gradient Check [{device}] ──────────────────────────────────────")

    batch, seq_len, dim = 2, 8, 32

    for bias_type in BiasType:
        pred = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
        target = torch.randn(batch, seq_len, dim, device=device)

        kwargs = {}
        if bias_type == BiasType.HUBER:
            kwargs["delta"] = 1.0
        elif bias_type == BiasType.LP:
            kwargs["p"] = 3.0

        module = AttentionalBias(bias_type=bias_type, **kwargs).to(device)
        out = module(pred, target)
        out.sum().backward()

        assert pred.grad is not None, f"[{bias_type.value}] no gradient computed"
        assert torch.isfinite(pred.grad).all(), f"[{bias_type.value}] non-finite gradients"
        print(f"  [{bias_type.value}] grad_norm={pred.grad.norm():.4f} ✓")

    print(f"  [gradient/{device}] all bias types passed ✓")


def run_functional_api_check(device: torch.device):
    print(f"\n── Functional API [{device}] ──────────────────────────────────────")

    batch, seq_len, dim = 2, 8, 32
    pred = torch.randn(batch, seq_len, dim, device=device)
    target = torch.randn(batch, seq_len, dim, device=device)

    functions = {
        "l2_bias": l2_bias,
        "huber_bias": lambda p, t: huber_bias(p, t, delta=1.0),
        "lp_bias": lambda p, t: lp_bias(p, t, p=3.0),
        "kl_bias": kl_bias,
    }

    for name, fn in functions.items():
        out = fn(pred, target)
        assert out.shape == (batch, seq_len), f"[{name}] shape mismatch: {out.shape}"
        assert torch.isfinite(out).all(), f"[{name}] non-finite values"
        print(f"  [{name}] shape={out.shape} ✓")

    print(f"  [functional/{device}] all functions passed ✓")


def run_neural_memory_integration(device: torch.device):
    print(f"\n── NeuralMemory Integration [{device}] ───────────────────────────")

    from open_titans.modules import NeuralMemory

    for bias_type in BiasType:
        kwargs = {}
        if bias_type == BiasType.HUBER:
            kwargs["delta"] = 1.0
        elif bias_type == BiasType.LP:
            kwargs["p"] = 3.0

        bias = AttentionalBias(bias_type=bias_type, **kwargs)

        model = NeuralMemory(
            dim=64,
            heads=2,
            chunk_size=4,
            store_memory_loss_fn=bias,
        ).to(device)

        model.eval()
        x = torch.randn(1, 16, 64, device=device)

        with torch.no_grad():
            out, state = model(x)

        assert out.shape == x.shape, f"[{bias_type.value}] shape mismatch: {out.shape} != {x.shape}"
        print(f"  [{bias_type.value}] output_shape={out.shape} ✓")

    print(f"  [integration/{device}] all bias types work with NeuralMemory ✓")


def run_benchmark(device: torch.device):
    print(f"\n── Benchmark [{device}] ────────────────────────────────────────")

    batch, seq_len, dim = 4, 1024, 256
    pred = torch.randn(batch, seq_len, dim, device=device)
    target = torch.randn(batch, seq_len, dim, device=device)

    warmup = 3
    runs = 10

    for bias_type in BiasType:
        kwargs = {}
        if bias_type == BiasType.HUBER:
            kwargs["delta"] = 1.0
        elif bias_type == BiasType.LP:
            kwargs["p"] = 3.0

        module = AttentionalBias(bias_type=bias_type, **kwargs).to(device)

        for _ in range(warmup):
            module(pred, target)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        _reset_memory(device)

        start = time.perf_counter()
        for _ in range(runs):
            module(pred, target)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / runs * 1000

        peak_mb = _peak_memory_mb(device)
        mem_str = f"{peak_mb:.1f} MB" if peak_mb is not None else "N/A"

        print(
            f"  [{bias_type.value}]"
            f"  latency={elapsed:.3f} ms"
            f"  peak_mem={mem_str}"
        )


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("\n" + "=" * 70)
    print(" OpenTitans — Attentional Bias Test Suite")
    print("=" * 70)

    for device in devices:
        run_correctness_check(device)
        run_gradient_check(device)
        run_functional_api_check(device)
        run_neural_memory_integration(device)
        run_benchmark(device)

    print("\n" + "=" * 70)
    print(" All tests passed ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
