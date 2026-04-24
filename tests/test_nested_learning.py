from __future__ import annotations

import gc
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_titans.modules.memory import (
    ExpressiveUpdateRule,
    sherman_morrison_step,
)
from open_titans.trainer import nested_train_step


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

    batch, dim_in, dim_out = 4, 32, 32
    W_t = torch.randn(batch, dim_out, dim_in, device=device)
    x_t = F.layer_norm(torch.randn(batch, dim_in, device=device), [dim_in])
    grad_l_in = torch.randn(batch, dim_out, device=device)

    module = ExpressiveUpdateRule(dim_in=dim_in).to(device)
    W_next = module(W_t, x_t, grad_l_in)

    assert W_next.shape == W_t.shape, f"shape mismatch: {W_next.shape} != {W_t.shape}"
    assert torch.isfinite(W_next).all(), "non-finite output"
    print(f"  shape={W_next.shape} norm={W_next.norm():.4f} ✓")
    print(f"  [correctness/{device}] passed ✓")


def run_sherman_morrison_inverse_check(device: torch.device):
    print(f"\n── Sherman-Morrison Inverse [{device}] ─────────────────────────────")

    dim = 16
    x = torch.randn(dim, device=device)
    eta = torch.tensor(0.5, device=device)

    naive_inv = torch.linalg.inv(torch.outer(x, x) + eta * torch.eye(dim, device=device))

    lambda_sq = x.pow(2).sum()
    scale = 1.0 / (lambda_sq + eta)
    sm_inv = (1.0 / eta) * (torch.eye(dim, device=device) - scale * torch.outer(x, x))

    diff = (naive_inv - sm_inv).abs().max().item()
    assert diff < 1e-4, f"SM inverse mismatch: max_diff={diff}"
    print(f"  max_abs_diff={diff:.2e} ✓")
    print(f"  [sm_inverse/{device}] passed ✓")


def run_rank1_update_vs_naive(device: torch.device):
    print(f"\n── Rank-1 Update vs Naive [{device}] ──────────────────────────────")

    batch, dim = 2, 16
    W_t = torch.randn(batch, dim, dim, device=device)
    x_t = F.layer_norm(torch.randn(batch, dim, device=device), [dim])
    grad_l_in = torch.randn(batch, dim, device=device)
    eta = torch.tensor([[0.5]], device=device).expand(batch, 1)

    lambda_sq = torch.tensor([float(dim)], device=device)
    W_sm = sherman_morrison_step(W_t, x_t, grad_l_in, eta, lambda_sq)

    W_naive = torch.zeros_like(W_t)
    for b in range(batch):
        xxt = torch.outer(x_t[b], x_t[b])
        inv_mat = torch.linalg.inv(xxt + eta[b, 0] * torch.eye(dim, device=device))
        grad_outer = torch.outer(grad_l_in[b], x_t[b])
        W_naive[b] = (grad_outer + eta[b, 0] * W_t[b]) @ inv_mat

    diff = (W_sm - W_naive).abs().max().item()
    assert diff < 1e-3, f"Rank-1 vs naive mismatch: max_diff={diff}"
    print(f"  max_abs_diff={diff:.2e} ✓")
    print(f"  [rank1_vs_naive/{device}] passed ✓")


def run_zero_gradient_identity(device: torch.device):
    print(f"\n── Zero Gradient Identity [{device}] ──────────────────────────────")

    batch, dim = 4, 16
    W_t = torch.randn(batch, dim, dim, device=device)
    x_t = F.layer_norm(torch.randn(batch, dim, device=device), [dim])

    y_hat = torch.bmm(W_t, x_t.unsqueeze(-1)).squeeze(-1)
    grad_l_in = y_hat

    eta = torch.ones(batch, 1, device=device)
    lambda_sq = torch.tensor([float(dim)], device=device)
    W_next = sherman_morrison_step(W_t, x_t, grad_l_in, eta, lambda_sq)

    diff = (W_next - W_t).abs().max().item()
    assert diff < 1e-5, f"identity failed: max_diff={diff}"
    print(f"  max_abs_diff={diff:.2e} (should be ~0) ✓")
    print(f"  [zero_grad_identity/{device}] passed ✓")


def run_self_referential_gating(device: torch.device):
    print(f"\n── Self-Referential Gating [{device}] ─────────────────────────────")

    dim = 32
    module = ExpressiveUpdateRule(dim_in=dim).to(device)

    x1 = torch.randn(1, dim, device=device) * 5.0
    x2 = torch.randn(1, dim, device=device) * 0.1

    eta1 = F.softplus(module.eta_proj(x1))
    eta2 = F.softplus(module.eta_proj(x2))

    assert not torch.allclose(eta1, eta2, atol=1e-6), (
        f"eta should differ for different inputs: η1={eta1.item():.6f}, η2={eta2.item():.6f}"
    )
    print(f"  η1={eta1.item():.6f}, η2={eta2.item():.6f} (different ✓)")
    print(f"  [self_referential/{device}] passed ✓")


def run_gradient_flow_check(device: torch.device):
    print(f"\n── Gradient Flow [{device}] ────────────────────────────────────────")

    batch, dim = 2, 16
    W_t = torch.randn(batch, dim, dim, device=device, requires_grad=True)
    x_t = F.layer_norm(torch.randn(batch, dim, device=device), [dim])
    grad_l_in = torch.randn(batch, dim, device=device)

    module = ExpressiveUpdateRule(dim_in=dim).to(device)
    W_next = module(W_t, x_t, grad_l_in)
    W_next.sum().backward()

    assert W_t.grad is not None, "no gradient on W_t"
    assert torch.isfinite(W_t.grad).all(), "non-finite W_t grad"
    print(f"  W_t grad_norm={W_t.grad.norm():.4f} ✓")

    eta_grad = module.eta_proj.weight.grad
    assert eta_grad is not None, "no gradient on eta_proj.weight"
    assert torch.isfinite(eta_grad).all(), "non-finite eta_proj grad"
    print(f"  eta_proj.weight grad_norm={eta_grad.norm():.6f} ✓")

    bias_grad = module.eta_proj.bias.grad
    assert bias_grad is not None, "no gradient on eta_proj.bias"
    print(f"  eta_proj.bias grad_norm={bias_grad.norm():.6f} ✓")

    print(f"  [gradient_flow/{device}] passed ✓")


def run_sequential_memory_update(device: torch.device):
    print(f"\n── Sequential Memory Update [{device}] ────────────────────────────")

    batch, dim, seq_len = 2, 16, 8
    module = ExpressiveUpdateRule(dim_in=dim).to(device)

    W = torch.zeros(batch, dim, dim, device=device)
    sequence = F.layer_norm(torch.randn(batch, seq_len, dim, device=device), [dim])
    targets = torch.randn(batch, seq_len, dim, device=device)

    norms = []
    for t in range(seq_len):
        x_t = sequence[:, t]
        target_t = targets[:, t]
        y_hat = torch.bmm(W, x_t.unsqueeze(-1)).squeeze(-1)
        grad_l_in = 2.0 * (y_hat - target_t)
        W = module(W, x_t, grad_l_in)
        norms.append(W.norm().item())

    for i in range(1, len(norms)):
        assert norms[i] != norms[i - 1], f"W did not change at step {i}"

    print(f"  W norms: {' → '.join(f'{n:.2f}' for n in norms)}")
    print(f"  [sequential/{device}] W evolves at each step ✓")


def run_batch_consistency(device: torch.device):
    print(f"\n── Batch Consistency [{device}] ────────────────────────────────────")

    dim = 16
    module = ExpressiveUpdateRule(dim_in=dim).to(device)

    W_single = torch.randn(1, dim, dim, device=device)
    x_single = F.layer_norm(torch.randn(1, dim, device=device), [dim])
    grad_single = torch.randn(1, dim, device=device)

    W_batch = W_single.expand(4, -1, -1).contiguous()
    x_batch = x_single.expand(4, -1).contiguous()
    grad_batch = grad_single.expand(4, -1).contiguous()

    out_single = module(W_single, x_single, grad_single)
    out_batch = module(W_batch, x_batch, grad_batch)

    for i in range(4):
        diff = (out_batch[i] - out_single[0]).abs().max().item()
        assert diff < 1e-5, f"batch element {i} mismatch: {diff}"

    print(f"  all 4 batch elements match single ✓")
    print(f"  [batch_consistency/{device}] passed ✓")


def run_numerical_stability(device: torch.device):
    print(f"\n── Numerical Stability [{device}] ──────────────────────────────────")

    batch, dim = 2, 16
    module = ExpressiveUpdateRule(dim_in=dim).to(device)

    W_large = torch.randn(batch, dim, dim, device=device) * 100.0
    x_large = F.layer_norm(torch.randn(batch, dim, device=device), [dim])
    grad_large = torch.randn(batch, dim, device=device) * 100.0
    out = module(W_large, x_large, grad_large)
    assert torch.isfinite(out).all(), "non-finite with large inputs"
    print(f"  large inputs: finite ✓")

    W_small = torch.randn(batch, dim, dim, device=device) * 1e-6
    x_small = F.layer_norm(torch.randn(batch, dim, device=device), [dim])
    grad_small = torch.randn(batch, dim, device=device) * 1e-6
    out = module(W_small, x_small, grad_small)
    assert torch.isfinite(out).all(), "non-finite with small inputs"
    print(f"  small inputs: finite ✓")

    print(f"  [numerical_stability/{device}] passed ✓")


def run_nested_train_step(device: torch.device):
    print(f"\n── Nested Train Step [{device}] ────────────────────────────────────")

    from open_titans.models.titans_mac import TitansMACConfig, TitansMACModel

    # 1. Instantiate the native MAC architecture
    config = TitansMACConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        segment_len=4,
        neural_memory_layers=[1],
    )
    
    # 2. Inject the Sherman-Morrison update rule into NeuralMemory via kwargs
    # The Nested Learning architecture requires the inner memory to be a single Linear matrix
    update_rule = ExpressiveUpdateRule(dim_in=16).to(device)
    memory_model = nn.Linear(16, 16, bias=False).to(device)
    model = TitansMACModel(config, update_rule=update_rule, neural_memory_model=memory_model).to(device)

    # 3. The Outer Optimizer updates the meta-parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs = torch.randint(0, 100, (2, 8), device=device)
    targets = torch.randint(0, 100, (2, 8), device=device)

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        # 4. The Inner Optimizer (Sherman-Morrison) runs natively inside the forward pass
        output = model(inputs, return_loss=True, labels=targets)
        loss = output.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  losses: {' → '.join(f'{l:.4f}' for l in losses)}")
    print(f"  [nested_train_step/{device}] outer loop executed ✓")


def run_benchmark(device: torch.device):
    print(f"\n── Benchmark [{device}] ────────────────────────────────────────────")

    batch, dim = 8, 128
    W_t = torch.randn(batch, dim, dim, device=device)
    x_t = F.layer_norm(torch.randn(batch, dim, device=device), [dim])
    grad_l_in = torch.randn(batch, dim, device=device)

    module = ExpressiveUpdateRule(dim_in=dim).to(device)
    warmup, runs = 3, 10

    for _ in range(warmup):
        module(W_t, x_t, grad_l_in)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _reset_memory(device)

    start = time.perf_counter()
    for _ in range(runs):
        module(W_t, x_t, grad_l_in)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / runs * 1000

    peak_mb = _peak_memory_mb(device)
    mem_str = f"{peak_mb:.1f} MB" if peak_mb is not None else "N/A"
    print(f"  [expressive] latency={elapsed:.3f} ms  peak_mem={mem_str}")


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    print("\n" + "=" * 70)
    print(" OpenTitans — Nested Learning Test Suite")
    print("=" * 70)

    for device in devices:
        run_correctness_check(device)
        run_sherman_morrison_inverse_check(device)
        run_rank1_update_vs_naive(device)
        run_zero_gradient_identity(device)
        run_self_referential_gating(device)
        run_gradient_flow_check(device)
        run_sequential_memory_update(device)
        run_batch_consistency(device)
        run_numerical_stability(device)
        run_nested_train_step(device)
        run_benchmark(device)

    print("\n" + "=" * 70)
    print(" All tests passed ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
