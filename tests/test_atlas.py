import torch
from open_titans.modules.memory.retrospective import RetrospectiveMemoryBuffer
from open_titans.optim.muon import newton_schulz5, Muon
from open_titans.models.atlas.registry import create_atlas_model


def test_retrospective_memory_buffer():
    b, s, d = 2, 5, 16
    c = 3
    buf = RetrospectiveMemoryBuffer(c, d)
    curr = torch.randn(b, s, d)
    
    ctx, next_buf = buf(curr)
    assert ctx.shape == (b, s, d)
    assert next_buf.shape == (b, c, d)
    
    curr2 = torch.randn(b, 2, d)
    ctx2, next_buf2 = buf(curr2, next_buf)
    assert ctx2.shape == (b, c + 2, d)
    assert next_buf2.shape == (b, c, d)
    
    mask = buf.get_causal_mask(2, 3)
    assert mask.shape == (2, 5)


def test_newton_schulz5():
    g = torch.randn(2, 4, 4)
    out = newton_schulz5(g)
    assert out.shape == g.shape


def test_muon_optimizer():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Muon([p], lr=0.01)
    
    loss = p.sum()
    loss.backward()
    opt.step()
    
    assert p.grad is not None


def test_atlas_variants():
    vocab_size = 100
    for variant in ["deep_transformers", "mag", "mal"]:
        model = create_atlas_model(
            variant, 
            vocab_size=vocab_size,
            hidden_size=32, 
            num_hidden_layers=2,
            num_attention_heads=2, 
            intermediate_size=64,
            chunk_size=4,
            retrospective_window=8
        )
        input_ids = torch.randint(0, vocab_size, (2, 10))
        labels = torch.randint(0, vocab_size, (2, 10))
        
        out = model(input_ids, labels=labels)
        
        assert out.logits.shape == (2, 10, vocab_size)
        assert out.loss is not None
        assert len(out.past_key_values) == 2
        
        out.loss.backward()

if __name__ == "__main__":
    test_retrospective_memory_buffer()
    test_newton_schulz5()
    test_muon_optimizer()
    test_atlas_variants()
    print("All tests passed.")
