"""
Sequential User Model (SASRec-style)
=====================================
Transformer-based sequential user modeling for recommendation.

Replaces pointwise user embeddings with sequence-aware modeling.
Takes ordered user interaction sequences and produces user embeddings
that capture temporal dynamics and preference evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Single transformer block (self-attention + FFN)."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - True for valid positions, False for padding
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Self-attention with residual
        # key_padding_mask: True = ignore (padding), False = attend
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True = ignore (padding)
        
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SequentialUserModel(nn.Module):
    """
    SASRec-style sequential user model.
    
    Takes ordered item sequences and produces user embeddings that capture
    temporal dynamics and preference evolution.
    """
    
    def __init__(
        self,
        n_items: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        """
        Args:
            n_items: Number of items (for item embedding lookup)
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(SequentialUserModel, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Item embeddings (learned)
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)  # +1 for padding
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        if self.item_embedding.padding_idx is not None:
            self.item_embedding.weight.data[self.item_embedding.padding_idx].fill_(0)
    
    def forward(
        self,
        item_sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            item_sequences: [batch_size, seq_len] - Item indices (0 = padding)
            mask: [batch_size, seq_len] - True for valid positions, False for padding
        
        Returns:
            user_embeddings: [batch_size, d_model] - User embeddings from last position
        """
        batch_size, seq_len = item_sequences.shape
        
        # Embed items
        x = self.item_embedding(item_sequences)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Get user embedding from last valid position
        if mask is not None:
            # Find last valid position for each user
            # mask: True = valid, False = padding
            # We want the last True position
            seq_lengths = mask.sum(dim=1)  # [batch_size]
            seq_lengths = (seq_lengths - 1).clamp(min=0)  # Last valid index
            
            # Extract last valid position
            user_embeddings = x[torch.arange(batch_size), seq_lengths]  # [batch_size, d_model]
        else:
            # No mask: use last position
            user_embeddings = x[:, -1, :]  # [batch_size, d_model]
        
        return user_embeddings
    
    def get_sequence_embeddings(
        self,
        item_sequences: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get embeddings for all positions in sequence (for training).
        
        Args:
            item_sequences: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = item_sequences.shape
        
        # Embed items
        x = self.item_embedding(item_sequences)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        return x


def create_sequence_batch(
    user_sequences: List[List[str]],
    item_to_idx: dict,
    max_seq_len: int,
    pad_token: int = 0
) -> tuple:
    """
    Create batched sequences with padding and masking.
    
    Args:
        user_sequences: List of item sequences (item_ids as strings)
        item_to_idx: Mapping from item_id to index
        max_seq_len: Maximum sequence length
        pad_token: Padding token index (default: 0)
    
    Returns:
        sequences: [batch_size, max_seq_len] - Padded item indices
        mask: [batch_size, max_seq_len] - True for valid, False for padding
    """
    batch_size = len(user_sequences)
    sequences = []
    masks = []
    
    for seq in user_sequences:
        # Convert item_ids to indices
        seq_indices = [item_to_idx.get(item_id, pad_token) for item_id in seq]
        
        # Truncate if too long
        if len(seq_indices) > max_seq_len:
            seq_indices = seq_indices[-max_seq_len:]  # Keep last max_seq_len items
        
        # Pad if too short
        seq_len = len(seq_indices)
        if seq_len < max_seq_len:
            padding = [pad_token] * (max_seq_len - seq_len)
            seq_indices = padding + seq_indices  # Pad at beginning
        
        sequences.append(seq_indices)
        
        # Create mask: True for valid positions, False for padding
        mask = [False] * (max_seq_len - seq_len) + [True] * seq_len
        masks.append(mask)
    
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.bool)
    
    return sequences_tensor, mask_tensor

