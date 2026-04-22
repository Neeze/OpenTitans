from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BenchResult:
    label: str
    device: str
    batch: int
    seq_len: int
    dim: int
    latency_ms: float
    throughput_tok_per_sec: float
    peak_memory_mb: Optional[float]
    warmup_runs: int
    timed_runs: int


def _reset_memory(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return None


def _time_fn(fn, warmup: int, runs: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed / runs * 1000


def benchmark_neural_memory(
    dim: int = 128,
    seq_len: int = 512,
    batch: int = 2,
    heads: int = 4,
    chunk_size: int = 8,
    device: torch.device = torch.device("cpu"),
    warmup: int = 3,
    runs: int = 10,
    label: str = "",
) -> BenchResult:
    from open_titans.modules import NeuralMemory

    model = NeuralMemory(
        dim=dim,
        heads=heads,
        chunk_size=chunk_size,
        momentum=True,
        momentum_order=1,
        pre_rmsnorm=True,
    ).to(device)

    model.eval()
    x = torch.randn(batch, seq_len, dim, device=device)

    _reset_memory(device)

    def run():
        with torch.no_grad():
            model(x)

    latency_ms = _time_fn(run, warmup, runs, device)
    peak_mb = _peak_memory_mb(device)
    throughput = (batch * seq_len) / (latency_ms / 1000)

    return BenchResult(
        label=label or f"NeuralMemory_{device.type}",
        device=str(device),
        batch=batch,
        seq_len=seq_len,
        dim=dim,
        latency_ms=latency_ms,
        throughput_tok_per_sec=throughput,
        peak_memory_mb=peak_mb,
        warmup_runs=warmup,
        timed_runs=runs,
    )


def benchmark_forward_and_backward(
    dim: int = 128,
    seq_len: int = 512,
    batch: int = 2,
    heads: int = 4,
    chunk_size: int = 8,
    device: torch.device = torch.device("cpu"),
    warmup: int = 3,
    runs: int = 10,
    label: str = "",
) -> BenchResult:
    from open_titans.modules import NeuralMemory

    model = NeuralMemory(
        dim=dim,
        heads=heads,
        chunk_size=chunk_size,
        momentum=True,
        momentum_order=1,
        pre_rmsnorm=True,
    ).to(device)

    model.train()
    x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)

    _reset_memory(device)

    def run():
        out, _ = model(x)
        out.mean().backward()
        model.zero_grad(set_to_none=True)

    latency_ms = _time_fn(run, warmup, runs, device)
    peak_mb = _peak_memory_mb(device)
    throughput = (batch * seq_len) / (latency_ms / 1000)

    return BenchResult(
        label=label or f"NeuralMemory_fwd_bwd_{device.type}",
        device=str(device),
        batch=batch,
        seq_len=seq_len,
        dim=dim,
        latency_ms=latency_ms,
        throughput_tok_per_sec=throughput,
        peak_memory_mb=peak_mb,
        warmup_runs=warmup,
        timed_runs=runs,
    )


def print_result(r: BenchResult):
    mem_str = f"{r.peak_memory_mb:.1f} MB" if r.peak_memory_mb is not None else "N/A"
    print(
        f"  [{r.label}]"
        f"  device={r.device}"
        f"  batch={r.batch}  seq={r.seq_len}  dim={r.dim}"
        f"  latency={r.latency_ms:.2f} ms"
        f"  throughput={r.throughput_tok_per_sec:,.0f} tok/s"
        f"  peak_mem={mem_str}"
    )


def run_correctness_check(device: torch.device):
    from open_titans.modules import NeuralMemory, NeuralMemState

    model = NeuralMemory(dim=64, heads=2, chunk_size=4).to(device)
    model.eval()

    x = torch.randn(1, 16, 64, device=device)

    with torch.no_grad():
        out, state = model(x)

    assert isinstance(out, torch.Tensor), "output must be a Tensor"
    assert out.shape == x.shape, f"shape mismatch: {out.shape} != {x.shape}"
    assert isinstance(state, NeuralMemState), "state must be NeuralMemState"
    print(f"  [correctness/{device}] output shape {out.shape} ✓")

    with torch.no_grad():
        out2, state2 = model(x, state=state)
    assert out2.shape == x.shape, f"stateful shape mismatch: {out2.shape}"
    print(f"  [correctness/{device}] stateful forward shape {out2.shape} ✓")


CPU_CONFIGS = [
    dict(batch=1, seq_len=64, dim=64, heads=2, chunk_size=4),
    dict(batch=1, seq_len=128, dim=64, heads=2, chunk_size=4),
]

CUDA_CONFIGS = [
    dict(batch=1, seq_len=128, dim=64, heads=2, chunk_size=4),
    dict(batch=2, seq_len=512, dim=128, heads=4, chunk_size=8),
    dict(batch=4, seq_len=1024, dim=256, heads=8, chunk_size=16),
]


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("\n" + "=" * 70)
    print(" OpenTitans — NeuralMemory Benchmark")
    print("=" * 70)

    for device in devices:
        print(f"\n── Correctness [{device}] ──────────────────────────────────────────")
        run_correctness_check(device)

    for device in devices:
        configs = CUDA_CONFIGS if device.type == "cuda" else CPU_CONFIGS
        warmup = 2 if device.type == "cpu" else 3
        runs = 5 if device.type == "cpu" else 10

        print(f"\n── Forward-only [{device}] ────────────────────────────────────────")
        for cfg in configs:
            r = benchmark_neural_memory(**cfg, device=device, warmup=warmup, runs=runs, label=f"fwd_{device.type}")
            print_result(r)

        print(f"\n── Forward + Backward [{device}] ─────────────────────────────────")
        for cfg in configs:
            r = benchmark_forward_and_backward(**cfg, device=device, warmup=warmup, runs=runs, label=f"fwd_bwd_{device.type}")
            print_result(r)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
