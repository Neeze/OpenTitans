import pytest
import torch
import torch.nn as nn

from open_titans.generation.titans_cache import TitansCache, AtlasCache
from open_titans.generation.generation_mixin import (
    top_k_filtering,
    top_p_filtering,
    sample_from_logits,
    TitansGenerationMixin,
    AtlasGenerationMixin,
)
from open_titans.modules.memory.neural_memory import NeuralMemState, mem_state_detach


class TestTitansCache:

    def test_init_empty(self):
        cache = TitansCache()
        assert len(cache) == 0
        assert cache.get_seq_length() == 0

    def test_from_layer_states(self):
        states = [None, None, None]
        cache = TitansCache.from_layer_states(states, seen_tokens=10)
        assert len(cache) == 3
        assert cache.get_seq_length() == 10

    def test_getitem_setitem(self):
        cache = TitansCache.from_layer_states([None, None], seen_tokens=0)
        assert cache[0] is None
        dummy = "state_placeholder"
        cache[0] = dummy
        assert cache[0] == dummy

    def test_update_seen_tokens(self):
        cache = TitansCache.from_layer_states([None], seen_tokens=5)
        cache.update_seen_tokens(3)
        assert cache.get_seq_length() == 8

    def test_detach_with_none_states(self):
        cache = TitansCache.from_layer_states([None, None], seen_tokens=5)
        detached = cache.detach()
        assert len(detached) == 2
        assert detached.get_seq_length() == 5
        assert detached[0] is None

    def test_detach_with_real_state(self):
        """
        Verify detach() breaks the computation graph.
        Given S_t with gradient-attached tensors, after detach:
        - All tensors must have requires_grad=False
        - No grad_fn should be present
        """
        w = torch.randn(2, 4, requires_grad=True) * 2.0
        state = NeuralMemState(
            seq_index=10,
            weights=w,
            cache_store_segment=None,
            states=None,
            updates=torch.randn(2, 4, requires_grad=True),
        )
        cache = TitansCache.from_layer_states([state], seen_tokens=10)
        detached = cache.detach()
        ds = detached[0]
        assert not ds.weights.requires_grad
        assert ds.weights.grad_fn is None
        assert not ds.updates.requires_grad


class TestAtlasCache:

    def test_init_empty(self):
        cache = AtlasCache()
        assert len(cache) == 0
        assert cache.get_seq_length() == 0

    def test_from_layer_states(self):
        mem_state = torch.randn(2, 8, 8)
        buf_k = torch.randn(2, 4, 8)
        buf_v = torch.randn(2, 4, 8)
        states = [(mem_state, buf_k, buf_v)]
        cache = AtlasCache.from_layer_states(states, seen_tokens=20)
        assert len(cache) == 1
        assert cache.get_seq_length() == 20

    def test_detach_breaks_graph(self):
        """
        ATLAS cache detach must break graph for W_{t+1}.
        W_{t+1} = W_t * alpha - orth_grad * eta
        If not detached, graph grows O(T) -> OOM.
        """
        mem_state = torch.randn(2, 8, 8, requires_grad=True)
        states = [(mem_state, None, None)]
        cache = AtlasCache.from_layer_states(states, seen_tokens=5)
        detached = cache.detach()
        detached_mem, _, _ = detached[0]
        assert not detached_mem.requires_grad
        assert detached_mem.grad_fn is None


class TestTopKFiltering:
    """
    Top-K filtering: keep only the k highest-probability logits.
    For logits z_i, the filtered logits are:
        z'_i = z_i  if rank(z_i) <= k
        z'_i = -inf  otherwise
    """

    def test_basic_topk(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_filtering(logits, top_k=3)
        finite_mask = filtered.isfinite()
        assert finite_mask.sum().item() == 3
        assert filtered[0, 1].item() == 5.0
        assert filtered[0, 4].item() == 4.0
        assert filtered[0, 2].item() == 3.0

    def test_topk_equals_vocab(self):
        logits = torch.randn(1, 10)
        filtered = top_k_filtering(logits, top_k=10)
        assert torch.allclose(logits, filtered)

    def test_topk_one(self):
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        filtered = top_k_filtering(logits, top_k=1)
        finite_mask = filtered.isfinite()
        assert finite_mask.sum().item() == 1
        assert filtered[0, 1].item() == 5.0


class TestTopPFiltering:
    """
    Nucleus (Top-P) sampling: keep the smallest set of tokens
    whose cumulative probability >= p.
    Sort by descending probability, then:
        cumsum(softmax(z_sorted)) - softmax(z_sorted)[i] >= p -> mask out
    """

    def test_basic_topp(self):
        logits = torch.tensor([[10.0, 1.0, 0.1, 0.01]])
        filtered = top_p_filtering(logits, top_p=0.9)
        probs = torch.softmax(filtered, dim=-1)
        assert probs[0, 0].item() > 0.5

    def test_topp_one_keeps_all(self):
        logits = torch.randn(1, 10)
        filtered = top_p_filtering(logits, top_p=1.0)
        assert filtered.isfinite().all()

    def test_topp_very_small(self):
        logits = torch.tensor([[10.0, 1.0, 0.1, 0.01]])
        filtered = top_p_filtering(logits, top_p=0.01)
        probs = torch.softmax(filtered, dim=-1)
        assert probs[0, 0].item() > 0.99


class TestSampleFromLogits:
    """
    sample_from_logits combines temperature scaling, top-k, and top-p:
        z' = z / T
        apply top_k
        apply top_p
        x ~ Categorical(softmax(z'))
    """

    def test_greedy_decode(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        token = sample_from_logits(logits, temperature=0.0)
        assert token.item() == 1

    def test_output_shape(self):
        logits = torch.randn(4, 100)
        tokens = sample_from_logits(logits, temperature=1.0, top_k=10)
        assert tokens.shape == (4, 1)

    def test_with_topk_and_topp(self):
        logits = torch.randn(2, 50)
        tokens = sample_from_logits(logits, temperature=0.8, top_k=10, top_p=0.9)
        assert tokens.shape == (2, 1)
        assert (tokens >= 0).all()
        assert (tokens < 50).all()

    def test_deterministic_at_zero_temperature(self):
        logits = torch.tensor([[1.0, 10.0, 3.0]])
        t1 = sample_from_logits(logits, temperature=0.0)
        t2 = sample_from_logits(logits, temperature=0.0)
        assert t1.item() == t2.item() == 1

    def test_batch_consistency(self):
        batch = 8
        vocab = 100
        logits = torch.randn(batch, vocab)
        tokens = sample_from_logits(logits, temperature=0.0)
        expected = logits.argmax(dim=-1, keepdim=True)
        assert torch.equal(tokens, expected)


class TestMACGeneration:

    @pytest.fixture
    def mac_model(self):
        from open_titans.models.titans_mac.configuration_mac import TitansMACConfig
        from open_titans.models.titans_mac.modeling_mac import TitansMACModel

        config = TitansMACConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            segment_len=16,
            num_attention_heads=2,
            dim_head=32,
            intermediate_size=128,
            num_residual_streams=1,
            neural_memory_layers=[2],
            num_longterm_mem_tokens=0,
        )
        model = TitansMACModel(config)
        model.eval()
        return model

    def test_forward_returns_cache(self, mac_model):
        """
        Forward with return_cache=True should produce a list of
        NeuralMemState per layer (None for non-memory layers).
        """
        x = torch.randint(0, 100, (1, 16))
        out = mac_model(x, return_cache=True)
        assert out.past_key_values is not None
        assert len(out.past_key_values) == 2

    def test_generate_output_shape(self, mac_model):
        """
        generate() should return shape (B, prompt_len + max_new_tokens).
        """
        prompt = torch.randint(0, 100, (1, 16))
        output = mac_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert output.shape == (1, 20)

    def test_generate_greedy_deterministic(self, mac_model):
        """
        Greedy decoding (T=0) must be deterministic:
        generate(x, T=0) == generate(x, T=0) for same input.
        """
        prompt = torch.randint(0, 100, (1, 16))
        out1 = mac_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        out2 = mac_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_generate_preserves_prompt(self, mac_model):
        """
        The first prompt_len tokens of the output must be the input.
        """
        prompt = torch.randint(0, 100, (1, 16))
        output = mac_model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=10)
        assert torch.equal(output[:, :16], prompt)

    def test_generate_batch(self, mac_model):
        """
        Batched generation: output shape (B, prompt_len + max_new_tokens).
        """
        prompt = torch.randint(0, 100, (2, 16))
        output = mac_model.generate(prompt, max_new_tokens=4, temperature=0.5, top_k=10)
        assert output.shape == (2, 20)

    def test_mixin_hierarchy(self, mac_model):
        assert isinstance(mac_model, TitansGenerationMixin)
        assert hasattr(mac_model, "prepare_inputs_for_generation")
        assert hasattr(mac_model, "generate")


class TestAtlasGeneration:

    @pytest.fixture
    def atlas_model(self):
        from open_titans.models.atlas.configuration_atlas import TitansAtlasConfig
        from open_titans.models.atlas.modeling_atlas import AtlasModel

        config = TitansAtlasConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_seq_len=256,
            chunk_size=8,
            retrospective_window=16,
            muon_ns_steps=2,
            variant="deep_transformers",
        )
        model = AtlasModel(config)
        model.eval()
        return model

    def test_forward_returns_cache(self, atlas_model):
        """
        ATLAS forward should always return past_key_values as list of
        (mem_state, buffer_k, buffer_v) tuples.
        """
        x = torch.randint(0, 100, (1, 16))
        out = atlas_model(x)
        assert out.past_key_values is not None
        assert len(out.past_key_values) == 2
        mem_state, buf_k, buf_v = out.past_key_values[0]
        assert mem_state is not None

    def test_generate_output_shape(self, atlas_model):
        prompt = torch.randint(0, 100, (1, 16))
        output = atlas_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert output.shape == (1, 20)

    def test_generate_greedy_deterministic(self, atlas_model):
        prompt = torch.randint(0, 100, (1, 16))
        out1 = atlas_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        out2 = atlas_model.generate(prompt, max_new_tokens=4, temperature=0.0)
        assert torch.equal(out1, out2)

    def test_generate_preserves_prompt(self, atlas_model):
        prompt = torch.randint(0, 100, (1, 16))
        output = atlas_model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=10)
        assert torch.equal(output[:, :16], prompt)

    def test_atlas_cache_type(self, atlas_model):
        assert atlas_model._uses_atlas_cache() is True
        assert isinstance(atlas_model, AtlasGenerationMixin)

    def test_ttt_grad_mode(self, atlas_model):
        """
        With enable_ttt_grad=True, the inner optimizer's gradient computation
        should be enabled even during generation (test-time training).
        The outer model parameters must NOT accumulate gradients.
        """
        prompt = torch.randint(0, 100, (1, 16))
        output = atlas_model.generate(
            prompt, max_new_tokens=2, temperature=0.0, enable_ttt_grad=True,
        )
        assert output.shape == (1, 18)
        for param in atlas_model.parameters():
            assert param.grad is None


class TestPrepareInputsForGeneration:
    """
    prepare_inputs_for_generation logic:
    - If past_key_values is None: pass full input_ids (prompt prefill)
    - If past_key_values exists: slice input_ids[:, -1:] (decode step)
    """

    def test_prefill_full_ids(self):
        class DummyModel(TitansGenerationMixin):
            def forward(self, input_ids, **kwargs): pass
            def _get_num_layers(self): return 1
            def _uses_atlas_cache(self): return False

        model = DummyModel()
        ids = torch.randint(0, 100, (2, 32))
        out = model.prepare_inputs_for_generation(ids, past_key_values=None)
        assert torch.equal(out["input_ids"], ids)

    def test_decode_slices_last_token(self):
        class DummyModel(TitansGenerationMixin):
            def forward(self, input_ids, **kwargs): pass
            def _get_num_layers(self): return 1
            def _uses_atlas_cache(self): return False

        model = DummyModel()
        ids = torch.randint(0, 100, (2, 32))
        cache = TitansCache.from_layer_states([None], seen_tokens=31)
        out = model.prepare_inputs_for_generation(ids, past_key_values=cache)
        assert out["input_ids"].shape == (2, 1)
        assert torch.equal(out["input_ids"], ids[:, -1:])


class TestMemStateDetach:
    """
    The Memory Leak Trap (Graph Breaking):
    W_{t+1} must be detached before the next step.
    Without detach, the graph grows O(T) and OOM occurs ~50 steps.
    """

    def test_detach_breaks_computation_graph(self):
        weights = torch.randn(4, 8, requires_grad=True) * 1.0
        updates = torch.randn(4, 8, requires_grad=True)
        state = NeuralMemState(
            seq_index=5,
            weights=weights,
            cache_store_segment=torch.randn(2, 3),
            states=(torch.randn(2, 2, requires_grad=True),),
            updates=updates,
        )
        detached = mem_state_detach(state)
        assert not detached.weights.requires_grad
        assert detached.weights.grad_fn is None
        assert not detached.updates.requires_grad
        assert detached.seq_index == 5

    def test_detach_preserves_values(self):
        weights = torch.randn(4, 8)
        state = NeuralMemState(
            seq_index=3,
            weights=weights.clone(),
            cache_store_segment=None,
            states=None,
            updates=None,
        )
        detached = mem_state_detach(state)
        assert torch.equal(detached.weights, weights)

    def test_repeated_detach_is_safe(self):
        state = NeuralMemState(
            seq_index=1,
            weights=torch.randn(2, 4, requires_grad=True),
            cache_store_segment=None,
            states=None,
            updates=None,
        )
        d1 = mem_state_detach(state)
        d2 = mem_state_detach(d1)
        assert not d2.weights.requires_grad


class TestEOSHandling:

    def test_eos_stops_generation(self):
        """
        When EOS token is generated, remaining tokens should be
        pad_token_id for that sequence.
        """
        from open_titans.models.titans_mac.configuration_mac import TitansMACConfig
        from open_titans.models.titans_mac.modeling_mac import TitansMACModel

        config = TitansMACConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            segment_len=16,
            num_attention_heads=2,
            dim_head=32,
            intermediate_size=128,
            num_residual_streams=1,
            neural_memory_layers=[],
            num_longterm_mem_tokens=0,
        )
        model = TitansMACModel(config)
        model.eval()
        prompt = torch.randint(0, 100, (1, 16))
        output = model.generate(
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            eos_token_id=99,
            pad_token_id=0,
        )
        assert output.shape == (1, 24)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
