import torch
import torch.nn as nn
from typing import Tuple, Optional


class RetrospectiveMemoryBuffer(nn.Module):
    """
    Retrospective Memory Buffer for the ATLAS Architecture.
    Maintains a sliding window of the past $c$ tokens (context_size) to 
    allow optimization of the memory over local contexts (the Omega Rule),
    resolving conflicting token updates before permanent commitment.
    """
    def __init__(self, context_size: int, hidden_size: int):
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        
        # Buffer stores the past context_size states.
        # Shape will typically be (batch_size, context_size, hidden_size)
        # We don't use register_buffer for states as they are sequence-dependent.

    def forward(
        self, 
        current_states: torch.Tensor, 
        past_buffer: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_states: Tensor of shape (batch_size, seq_len, hidden_size)
            past_buffer: Optional Tensor of shape (batch_size, buffer_len, hidden_size)
            
        Returns:
            Tuple of:
            - context_window: The combined window (past_buffer + current_states)
              ready for Omega Rule memory update mapping. 
              Shape: (batch_size, past_len + seq_len, hidden_size)
            - next_buffer: The updated buffer preserving only the most recent `context_size` tokens.
              Shape: (batch_size, min(past_len + seq_len, context_size), hidden_size)
        """
        if past_buffer is not None:
            # Concatenate past buffer and current states along seq_len dimension
            context_window = torch.cat([past_buffer, current_states], dim=1)
        else:
            context_window = current_states
            
        # Truncate to the most recent `context_size` tokens to form the new buffer
        if context_window.size(1) > self.context_size:
            next_buffer = context_window[:, -self.context_size:, :]
        else:
            next_buffer = context_window
            
        return context_window, next_buffer

    def get_causal_mask(self, seq_len: int, buffer_len: int = 0) -> torch.Tensor:
        """
        Creates a localized lower-triangular causal mask for the active chunk
        and retrospective buffer, enforcing strict autoregressive causality 
        without needing N x N exhaustion.
        
        Args:
            seq_len: Current chunk sequence length.
            buffer_len: The length of the retrospective buffer appended before.
            
        Returns:
            Boolean causal mask tensor of shape (seq_len, buffer_len + seq_len).
        """
        total_len = buffer_len + seq_len
        mask = torch.ones(seq_len, total_len, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=buffer_len)
        return mask
