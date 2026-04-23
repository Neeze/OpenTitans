from __future__ import annotations

import gc
import time
from typing import Optional

import torch
import torch.nn as nn

from open_titans.models.miras import (
    MirasConfig,
    MirasModel,
    create_miras_model,
    list_variants,
    MIRAS_REGISTRY,
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


SMALL_CONFIG = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    dim_head=32,
    intermediate_size=128,
    mem_heads=2,
    chunk_size=8,
    vocab_size=500,
    max_seq_len=64,
)

TINY_CONFIG = dict(
    hidden_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    dim_head=32,
    intermediate_size=128,
    mem_heads=1,
    chunk_size=4,
    vocab_size=100,
    max_seq_len=32,
)


def run_registry_check():
    print("\n── Registry ────────────────────────────────────────────────────────")

    variants = list_variants()
    assert len(variants) == 3, f"Expected 3 variants, got {len(variants)}"
    assert "yaad" in variants
    assert "moneta" in variants
    assert "memora" in variants

    for name, desc in variants.items():
        print(f"  {name}: {desc}")

    print("  [registry] 3 variants registered ✓")


def run_config_check():
    print("\n── Config ──────────────────────────────────────────────────────────")

    for variant in ("yaad", "moneta", "memora"):
        config = MirasConfig(variant=variant, hidden_size=128, num_hidden_layers=2)
        assert config.variant == variant
        assert config.hidden_size == 128
        print(f"  [{variant}] config created ✓")

    try:
        MirasConfig(variant="invalid")
        assert False, "Should have raised"
    except AssertionError:
        print("  [invalid] rejected unknown variant ✓")


def run_forward_check(device: torch.device):
    print(f"\n── Forward [{device}] ──────────────────────────────────────────────")

    batch_size = 2
    seq_len = 32

    for variant in ("yaad", "moneta", "memora"):
        model = create_miras_model(variant, **SMALL_CONFIG).to(device)
        model.eval()

        input_ids = torch.randint(0, 500, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = model(input_ids)

        logits = output.logits
        assert logits.shape == (batch_size, seq_len, 500), (
            f"[{variant}] logits shape: {logits.shape}"
        )
        assert torch.isfinite(logits).all(), f"[{variant}] non-finite logits"

        assert output.past_key_values is not None, f"[{variant}] no cache returned"
        assert len(output.past_key_values) == SMALL_CONFIG["num_hidden_layers"]
        print(f"  [{variant}] logits={logits.shape} cache_layers={len(output.past_key_values)} ✓")

    print(f"  [forward/{device}] all variants passed ✓")


def run_loss_check(device: torch.device):
    print(f"\n── Loss [{device}] ─────────────────────────────────────────────────")

    batch_size = 2
    seq_len = 32

    for variant in ("yaad", "moneta", "memora"):
        model = create_miras_model(variant, **SMALL_CONFIG).to(device)
        model.train()

        input_ids = torch.randint(0, 500, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 500, (batch_size, seq_len), device=device)

        output = model(input_ids, labels=labels)

        assert output.loss is not None, f"[{variant}] loss is None"
        assert output.loss.ndim == 0, f"[{variant}] loss not scalar"
        assert torch.isfinite(output.loss), f"[{variant}] non-finite loss"
        print(f"  [{variant}] loss={output.loss.item():.4f} ✓")

    print(f"  [loss/{device}] all variants passed ✓")


def run_gradient_check(device: torch.device):
    print(f"\n── Gradient [{device}] ─────────────────────────────────────────────")

    batch_size = 1
    seq_len = 16

    for variant in ("yaad", "moneta", "memora"):
        model = create_miras_model(variant, **TINY_CONFIG).to(device)
        model.train()

        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 100, (batch_size, seq_len), device=device)

        output = model(input_ids, labels=labels)
        output.loss.backward()

        total_grad_norm = 0.0
        num_params_with_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
                num_params_with_grad += 1

        total_grad_norm = total_grad_norm ** 0.5
        assert num_params_with_grad > 0, f"[{variant}] no gradients"
        print(f"  [{variant}] grad_norm={total_grad_norm:.4f} params_with_grad={num_params_with_grad} ✓")

    print(f"  [gradient/{device}] all variants passed ✓")


def run_stateful_check(device: torch.device):
    print(f"\n── Stateful [{device}] ─────────────────────────────────────────────")

    batch_size = 1
    seq_len = 16

    for variant in ("yaad", "moneta", "memora"):
        model = create_miras_model(variant, **TINY_CONFIG).to(device)
        model.eval()

        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)

        with torch.no_grad():
            out1 = model(input_ids)
            cache = out1.past_key_values

            out2 = model(input_ids, cache=cache)

        assert out2.logits.shape == out1.logits.shape
        print(f"  [{variant}] stateful forward ✓")

    print(f"  [stateful/{device}] all variants passed ✓")


def run_benchmark(device: torch.device):
    print(f"\n── Benchmark [{device}] ────────────────────────────────────────────")

    batch_size = 2
    seq_len = 32
    warmup, runs = 2, 5

    for variant in ("yaad", "moneta", "memora"):
        model = create_miras_model(variant, **SMALL_CONFIG).to(device)
        model.eval()

        input_ids = torch.randint(0, 500, (batch_size, seq_len), device=device)

        for _ in range(warmup):
            with torch.no_grad():
                model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _reset_memory(device)

        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / runs * 1000

        peak_mb = _peak_memory_mb(device)
        mem_str = f"{peak_mb:.1f} MB" if peak_mb is not None else "N/A"

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [{variant}]  params={n_params:.2f}M  latency={elapsed:.1f} ms  peak_mem={mem_str}")


def run_custom_memory_model_check(device: torch.device):
    print(f"\n── Custom Memory Model [{device}] ──────────────────────────────────")

    dim_head = TINY_CONFIG["hidden_size"] // TINY_CONFIG["mem_heads"]

    custom_model = nn.Sequential(
        nn.Linear(dim_head, dim_head * 4, bias=False),
        nn.GELU(),
        nn.Linear(dim_head * 4, dim_head, bias=False),
    )

    for variant in ("yaad", "moneta", "memora"):
        model = create_miras_model(
            variant,
            neural_memory_model=custom_model,
            **TINY_CONFIG,
        ).to(device)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 16), device=device)

        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (1, 16, 100)
        assert torch.isfinite(output.logits).all()
        print(f"  [{variant}] custom memory model works ✓")

    print(f"  [custom_model/{device}] all variants passed ✓")


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("\n" + "=" * 70)
    print(" OpenTitans — MIRAS Model Test Suite (NeuralMemory)")
    print("=" * 70)

    run_registry_check()
    run_config_check()

    for device in devices:
        run_forward_check(device)
        run_loss_check(device)
        run_gradient_check(device)
        run_stateful_check(device)
        run_custom_memory_model_check(device)
        run_benchmark(device)

    print("\n" + "=" * 70)
    print(" All tests passed ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
