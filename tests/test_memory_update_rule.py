from __future__ import annotations

import gc
import time
from typing import Optional

import torch
import torch.nn.functional as F

from open_titans.modules.memory import (
    MemoryUpdateRule,
    UpdateRuleType,
    linear_update,
    yaad_update,
    memora_update,
    moneta_update,
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

    batch, dim = 4, 32
    w_prev = torch.randn(batch, dim, dim, device=device)
    grad = torch.randn(batch, dim, dim, device=device)
    error_norm = torch.rand(batch, device=device) * 2.0

    for rule_type in UpdateRuleType:
        kwargs = {}
        error = None
        if rule_type == UpdateRuleType.YAAD:
            error = error_norm
            kwargs["delta"] = 1.0
        elif rule_type == UpdateRuleType.MONETA:
            kwargs["l1_strength"] = 0.01

        module = MemoryUpdateRule(
            dim=dim, rule_type=rule_type, **kwargs,
        ).to(device)

        w_new = module(w_prev, grad, error_norm=error)

        assert w_new.shape == w_prev.shape, (
            f"[{rule_type.value}] shape mismatch: {w_new.shape} != {w_prev.shape}"
        )
        assert torch.isfinite(w_new).all(), f"[{rule_type.value}] non-finite output"
        print(f"  [{rule_type.value}] shape={w_new.shape} norm={w_new.norm():.4f} ✓")

    print(f"  [correctness/{device}] all update rules passed ✓")


def run_gradient_check(device: torch.device):
    print(f"\n── Gradient Check [{device}] ──────────────────────────────────────")

    batch, dim = 2, 16

    for rule_type in UpdateRuleType:
        w_prev = torch.randn(batch, dim, dim, device=device, requires_grad=True)
        grad_input = torch.randn(batch, dim, dim, device=device)
        error_norm = torch.rand(batch, device=device) * 2.0

        kwargs = {}
        error = None
        if rule_type == UpdateRuleType.YAAD:
            error = error_norm
            kwargs["delta"] = 1.0
        elif rule_type == UpdateRuleType.MONETA:
            kwargs["l1_strength"] = 0.01

        module = MemoryUpdateRule(
            dim=dim, rule_type=rule_type, **kwargs,
        ).to(device)

        w_new = module(w_prev, grad_input, error_norm=error)
        w_new.sum().backward()

        assert w_prev.grad is not None, f"[{rule_type.value}] no gradient"
        assert torch.isfinite(w_prev.grad).all(), f"[{rule_type.value}] non-finite grads"
        print(f"  [{rule_type.value}] grad_norm={w_prev.grad.norm():.4f} ✓")

    print(f"  [gradient/{device}] all update rules passed ✓")


def run_functional_api_check(device: torch.device):
    print(f"\n── Functional API [{device}] ──────────────────────────────────────")

    batch, dim = 2, 16
    w_prev = torch.randn(batch, dim, dim, device=device)
    grad = torch.randn(batch, dim, dim, device=device)
    alpha = torch.tensor(0.9, device=device)
    eta = torch.tensor(0.01, device=device)
    error_norm = torch.rand(batch, device=device)

    functions = {
        "linear_update": lambda: linear_update(w_prev, grad, alpha, eta),
        "yaad_update": lambda: yaad_update(w_prev, grad, alpha, eta, error_norm),
        "memora_update": lambda: memora_update(F.softmax(w_prev, dim=-1), grad, alpha, eta),
        "moneta_update": lambda: moneta_update(w_prev, grad, alpha, eta),
    }

    for name, fn in functions.items():
        out = fn()
        assert out.shape == w_prev.shape, f"[{name}] shape mismatch"
        assert torch.isfinite(out).all(), f"[{name}] non-finite"
        print(f"  [{name}] shape={out.shape} ✓")

    print(f"  [functional/{device}] all functions passed ✓")


def run_learnable_params_check(device: torch.device):
    print(f"\n── Learnable Params [{device}] ─────────────────────────────────────")

    batch, dim = 2, 8
    w_prev = torch.randn(batch, dim, dim, device=device)
    grad = torch.randn(batch, dim, dim, device=device)

    module = MemoryUpdateRule(
        dim=dim, rule_type="linear",
        learnable_eta=True, learnable_alpha=True,
    ).to(device)

    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    assert "_eta_logit" in trainable, "eta not learnable"
    assert "_alpha_logit" in trainable, "alpha not learnable"

    eta_before = module.eta.item()
    alpha_before = module.alpha.item()

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    w_new = module(w_prev, grad)
    w_new.sum().backward()
    optimizer.step()

    print(f"  η: {eta_before:.4f} → {module.eta.item():.4f}")
    print(f"  α: {alpha_before:.4f} → {module.alpha.item():.4f}")
    print(f"  [learnable/{device}] learnable params work ✓")


def run_decay_property_check(device: torch.device):
    print(f"\n── Decay Property [{device}] ───────────────────────────────────────")

    batch, dim = 2, 8
    w_prev = torch.randn(batch, dim, dim, device=device)
    zero_grad = torch.zeros_like(w_prev)

    module = MemoryUpdateRule(
        dim=dim, rule_type="linear",
        learnable_eta=False, learnable_alpha=False,
        init_alpha=0.5, init_eta=0.01,
    ).to(device)

    w_new = module(w_prev, zero_grad)
    expected = module.alpha * w_prev
    assert torch.allclose(w_new, expected, atol=1e-6), "linear decay failed with zero grad"
    print(f"  [linear] zero-gradient decay correct ✓")

    module_yaad = MemoryUpdateRule(
        dim=dim, rule_type="yaad",
        learnable_eta=False, learnable_alpha=False,
        init_alpha=0.5, init_eta=0.01,
    ).to(device)

    error_norm = torch.zeros(batch, device=device)
    w_new_yaad = module_yaad(w_prev, zero_grad, error_norm=error_norm)
    assert torch.allclose(w_new_yaad, expected, atol=1e-6), "yaad decay failed with zero grad"
    print(f"  [yaad] zero-gradient decay correct ✓")

    print(f"  [decay/{device}] decay property verified ✓")


def run_sparsity_check(device: torch.device):
    print(f"\n── Sparsity Check [{device}] ───────────────────────────────────────")

    batch, dim = 2, 16
    w_prev = torch.zeros(batch, dim, dim, device=device)
    small_grad = torch.randn(batch, dim, dim, device=device) * 0.001

    module = MemoryUpdateRule(
        dim=dim, rule_type="moneta",
        learnable_eta=False, learnable_alpha=False,
        init_alpha=0.9, init_eta=1.0,
        l1_strength=0.01,
    ).to(device)

    w_new = module(w_prev, small_grad)
    sparsity = (w_new == 0).float().mean()
    print(f"  [moneta] sparsity={sparsity.item():.2%} (small gradients thresholded to zero)")
    assert sparsity > 0.5, "MONETA should produce sparse updates with small gradients"
    print(f"  [sparsity/{device}] sparsity check passed ✓")


def run_normalization_check(device: torch.device):
    print(f"\n── Normalization Check [{device}] ──────────────────────────────────")

    batch, dim = 2, 8
    w_prev = F.softmax(torch.randn(batch, dim, dim, device=device), dim=-1)
    grad = torch.randn(batch, dim, dim, device=device)

    module = MemoryUpdateRule(
        dim=dim, rule_type="memora",
        learnable_eta=False, learnable_alpha=False,
        init_alpha=0.9, init_eta=0.01,
    ).to(device)

    w_new = module(w_prev, grad)
    row_sums = w_new.flatten(-2).sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"MEMORA output not normalized: {row_sums}"
    )
    print(f"  [memora] output sums to 1.0 ✓")
    assert (w_new >= 0).all(), "MEMORA output has negative values"
    print(f"  [memora] output non-negative ✓")
    print(f"  [normalization/{device}] check passed ✓")


def run_benchmark(device: torch.device):
    print(f"\n── Benchmark [{device}] ────────────────────────────────────────")

    batch, dim = 8, 128
    w_prev = torch.randn(batch, dim, dim, device=device)
    grad = torch.randn(batch, dim, dim, device=device)
    error_norm = torch.rand(batch, device=device)
    warmup, runs = 3, 10

    for rule_type in UpdateRuleType:
        kwargs = {}
        error = None
        if rule_type == UpdateRuleType.YAAD:
            error = error_norm
        elif rule_type == UpdateRuleType.MONETA:
            kwargs["l1_strength"] = 0.01

        module = MemoryUpdateRule(dim=dim, rule_type=rule_type, **kwargs).to(device)

        for _ in range(warmup):
            module(w_prev, grad, error_norm=error)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _reset_memory(device)

        start = time.perf_counter()
        for _ in range(runs):
            module(w_prev, grad, error_norm=error)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / runs * 1000

        peak_mb = _peak_memory_mb(device)
        mem_str = f"{peak_mb:.1f} MB" if peak_mb is not None else "N/A"
        print(f"  [{rule_type.value}]  latency={elapsed:.3f} ms  peak_mem={mem_str}")


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("\n" + "=" * 70)
    print(" OpenTitans — Memory Update Rule Test Suite")
    print("=" * 70)

    for device in devices:
        run_correctness_check(device)
        run_gradient_check(device)
        run_functional_api_check(device)
        run_learnable_params_check(device)
        run_decay_property_check(device)
        run_sparsity_check(device)
        run_normalization_check(device)
        run_benchmark(device)

    print("\n" + "=" * 70)
    print(" All tests passed ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
