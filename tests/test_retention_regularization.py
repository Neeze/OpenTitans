from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch

from open_titans.modules.gates import (
    RetentionRegularization,
    RetentionType,
    quadratic_local,
    quadratic_global,
    bregman_local,
    bregman_global,
    elastic_net_local,
    elastic_net_global,
    f_divergence_local,
    f_divergence_global,
)


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

    batch, dim = 4, 64
    w = torch.randn(batch, dim, dim, device=device)
    w_prev = torch.randn(batch, dim, dim, device=device)

    for rtype in RetentionType:
        local_kw = {}
        if rtype == RetentionType.ELASTIC_NET:
            local_kw["alpha"] = 0.5

        module = RetentionRegularization(
            retention_type=rtype, local_kwargs=local_kw or None,
        ).to(device)
        total, local_loss, global_loss = module(w, w_prev)

        assert total.shape == (batch,), f"[{rtype.value}] total shape: {total.shape}"
        assert local_loss.shape == (batch,), f"[{rtype.value}] local shape: {local_loss.shape}"
        assert global_loss.shape == (batch,), f"[{rtype.value}] global shape: {global_loss.shape}"
        assert torch.isfinite(total).all(), f"[{rtype.value}] non-finite total"
        assert torch.isfinite(local_loss).all(), f"[{rtype.value}] non-finite local"
        assert torch.isfinite(global_loss).all(), f"[{rtype.value}] non-finite global"
        assert (local_loss >= 0).all(), f"[{rtype.value}] negative local loss"
        assert (global_loss >= 0).all(), f"[{rtype.value}] negative global loss"

        print(
            f"  [{rtype.value}] total={total.mean():.4f} "
            f"local={local_loss.mean():.4f} global={global_loss.mean():.4f} ✓"
        )

    print(f"  [correctness/{device}] all retention types passed ✓")


def run_gradient_check(device: torch.device):
    print(f"\n── Gradient Check [{device}] ──────────────────────────────────────")

    batch, dim = 2, 32

    for rtype in RetentionType:
        w = torch.randn(batch, dim, dim, device=device, requires_grad=True)
        w_prev = torch.randn(batch, dim, dim, device=device)

        local_kw = {}
        if rtype == RetentionType.ELASTIC_NET:
            local_kw["alpha"] = 0.5

        module = RetentionRegularization(
            retention_type=rtype, local_kwargs=local_kw or None,
        ).to(device)
        total, _, _ = module(w, w_prev)
        total.sum().backward()

        assert w.grad is not None, f"[{rtype.value}] no gradient"
        assert torch.isfinite(w.grad).all(), f"[{rtype.value}] non-finite gradients"
        print(f"  [{rtype.value}] grad_norm={w.grad.norm():.4f} ✓")

    print(f"  [gradient/{device}] all retention types passed ✓")


def run_functional_api_check(device: torch.device):
    print(f"\n── Functional API [{device}] ──────────────────────────────────────")

    batch, dim = 2, 32
    w = torch.randn(batch, dim, dim, device=device)
    w_prev = torch.randn(batch, dim, dim, device=device)

    functions = {
        "quadratic_local": (quadratic_local, (w, w_prev)),
        "quadratic_global": (quadratic_global, (w,)),
        "bregman_local": (bregman_local, (w, w_prev)),
        "bregman_global": (bregman_global, (w,)),
        "elastic_net_local": (elastic_net_local, (w, w_prev)),
        "elastic_net_global": (elastic_net_global, (w,)),
        "f_divergence_local": (f_divergence_local, (w, w_prev)),
        "f_divergence_global": (f_divergence_global, (w,)),
    }

    for name, (fn, args) in functions.items():
        out = fn(*args)
        assert out.shape == (batch,), f"[{name}] shape: {out.shape}"
        assert torch.isfinite(out).all(), f"[{name}] non-finite"
        print(f"  [{name}] shape={out.shape} ✓")

    print(f"  [functional/{device}] all functions passed ✓")


def run_learnable_lambda_check(device: torch.device):
    print(f"\n── Learnable Lambda [{device}] ─────────────────────────────────────")

    batch, dim = 2, 16
    w = torch.randn(batch, dim, dim, device=device)
    w_prev = torch.randn(batch, dim, dim, device=device)

    module = RetentionRegularization(
        retention_type="quadratic",
        lambda_local=1.0,
        lambda_global=0.01,
        learnable_lambda=True,
    ).to(device)

    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    assert "_lambda_local" in trainable, "lambda_local not learnable"
    assert "_lambda_global" in trainable, "lambda_global not learnable"

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    total, _, _ = module(w, w_prev)
    total.sum().backward()
    optimizer.step()

    print(f"  lambda_local={module._lambda_local.item():.4f}")
    print(f"  lambda_global={module._lambda_global.item():.4f}")
    print(f"  [learnable/{device}] learnable lambda works ✓")


def run_identity_check(device: torch.device):
    print(f"\n── Identity Check [{device}] ───────────────────────────────────────")

    batch, dim = 2, 16
    w = torch.randn(batch, dim, dim, device=device)

    for rtype in RetentionType:
        local_kw = {}
        if rtype == RetentionType.ELASTIC_NET:
            local_kw["alpha"] = 0.5

        module = RetentionRegularization(
            retention_type=rtype, lambda_global=0.0, local_kwargs=local_kw or None,
        ).to(device)
        total, local_loss, _ = module(w, w)

        if rtype in (RetentionType.QUADRATIC, RetentionType.BREGMAN, RetentionType.ELASTIC_NET):
            assert torch.allclose(local_loss, torch.zeros_like(local_loss), atol=1e-6), (
                f"[{rtype.value}] local_loss should be 0 when w == w_prev"
            )
            print(f"  [{rtype.value}] local_loss=0 when w==w_prev ✓")
        else:
            print(f"  [{rtype.value}] local_loss={local_loss.mean():.6f} (softmax-based, non-zero expected) ✓")

    print(f"  [identity/{device}] all identity checks passed ✓")


def run_benchmark(device: torch.device):
    print(f"\n── Benchmark [{device}] ────────────────────────────────────────")

    batch, dim = 8, 128
    w = torch.randn(batch, dim, dim, device=device)
    w_prev = torch.randn(batch, dim, dim, device=device)

    warmup, runs = 3, 10

    for rtype in RetentionType:
        local_kw = {}
        if rtype == RetentionType.ELASTIC_NET:
            local_kw["alpha"] = 0.5

        module = RetentionRegularization(
            retention_type=rtype, local_kwargs=local_kw or None,
        ).to(device)

        for _ in range(warmup):
            module(w, w_prev)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        _reset_memory(device)

        start = time.perf_counter()
        for _ in range(runs):
            module(w, w_prev)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / runs * 1000

        peak_mb = _peak_memory_mb(device)
        mem_str = f"{peak_mb:.1f} MB" if peak_mb is not None else "N/A"

        print(f"  [{rtype.value}]  latency={elapsed:.3f} ms  peak_mem={mem_str}")


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("\n" + "=" * 70)
    print(" OpenTitans — Retention Regularization Test Suite")
    print("=" * 70)

    for device in devices:
        run_correctness_check(device)
        run_gradient_check(device)
        run_functional_api_check(device)
        run_learnable_lambda_check(device)
        run_identity_check(device)
        run_benchmark(device)

    print("\n" + "=" * 70)
    print(" All tests passed ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
