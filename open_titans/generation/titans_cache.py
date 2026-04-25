from __future__ import annotations

from typing import Optional, List
from dataclasses import dataclass, field

import torch
from torch.utils._pytree import tree_map

from ..modules.memory.neural_memory import NeuralMemState, mem_state_detach


@dataclass
class TitansCache:
    layer_states: List[Optional[NeuralMemState]] = field(default_factory=list)
    _seen_tokens: int = 0

    @classmethod
    def from_layer_states(cls, states: List[Optional[NeuralMemState]], seen_tokens: int = 0) -> TitansCache:
        cache = cls()
        cache.layer_states = list(states)
        cache._seen_tokens = seen_tokens
        return cache

    def __len__(self) -> int:
        return len(self.layer_states)

    def __getitem__(self, idx: int) -> Optional[NeuralMemState]:
        return self.layer_states[idx]

    def __setitem__(self, idx: int, state: Optional[NeuralMemState]):
        self.layer_states[idx] = state

    def get_seq_length(self) -> int:
        return self._seen_tokens

    def detach(self) -> TitansCache:
        detached = []
        for state in self.layer_states:
            if state is None:
                detached.append(None)
            else:
                detached.append(mem_state_detach(state))
        return TitansCache.from_layer_states(detached, self._seen_tokens)

    def update_seen_tokens(self, num_new_tokens: int):
        self._seen_tokens += num_new_tokens


@dataclass
class AtlasCache:
    layer_states: List[Optional[tuple]] = field(default_factory=list)
    _seen_tokens: int = 0

    @classmethod
    def from_layer_states(cls, states: List[Optional[tuple]], seen_tokens: int = 0) -> AtlasCache:
        cache = cls()
        cache.layer_states = list(states)
        cache._seen_tokens = seen_tokens
        return cache

    def __len__(self) -> int:
        return len(self.layer_states)

    def __getitem__(self, idx: int) -> Optional[tuple]:
        return self.layer_states[idx]

    def __setitem__(self, idx: int, state: Optional[tuple]):
        self.layer_states[idx] = state

    def get_seq_length(self) -> int:
        return self._seen_tokens

    def detach(self) -> AtlasCache:
        detached = []
        for state in self.layer_states:
            if state is None:
                detached.append(None)
            else:
                detached_state = tree_map(
                    lambda t: t.detach() if torch.is_tensor(t) else t,
                    state,
                )
                detached.append(detached_state)
        return AtlasCache.from_layer_states(detached, self._seen_tokens)

    def update_seen_tokens(self, num_new_tokens: int):
        self._seen_tokens += num_new_tokens
