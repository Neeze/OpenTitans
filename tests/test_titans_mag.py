from __future__ import annotations
import gc
import time
import torch
import torch.nn as nn
from open_titans.models.titans_mag.configuration_mag import TitansMAGConfig
from open_titans.models.titans_mag.modeling_mag import TitansMAGModel

def measure_performance(model, x, device_name="cpu"):
    print(f"--- Measuring performance on {device_name} ---")
    model.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x)
        
        if "cuda" in device_name:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.reset_peak_memory_stats()
            
            start_event.record()
            for _ in range(10):
                _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            
            latency = start_event.elapsed_time(end_event) / 10.0 # ms
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Latency: {latency:.2f} ms")
            print(f"Peak Memory: {peak_memory:.2f} MB")
        else:
            start_time = time.perf_counter()
            for _ in range(10):
                _ = model(x)
            end_time = time.perf_counter()
            
            latency = ((end_time - start_time) / 10.0) * 1000 # ms
            print(f"Latency: {latency:.2f} ms")
            print(f"Peak Memory: Not tracked for CPU directly via PyTorch in this script")

def test_titans_mag():
    config = TitansMAGConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        window_size=32,
        num_attention_heads=4,
        dim_head=64,
        intermediate_size=512,
        num_residual_streams=2,
        neural_memory_layers=[2],
        num_persist_mem_tokens=8,
    )
    
    neural_memory_model = nn.Sequential(
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256)
    )

    print("Instantiating MAG model...")
    model = TitansMAGModel(config, neural_memory_model=neural_memory_model)
    
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test CPU
    measure_performance(model, x, device_name="cpu")
    
    # Test CUDA
    try:
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            measure_performance(model, x, device_name="cuda")
        else:
            print("CUDA is not available, skipping CUDA test.")
    except Exception as e:
        print(f"CUDA test failed with error: {e}")

if __name__ == "__main__":
    test_titans_mag()
