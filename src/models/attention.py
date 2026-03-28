#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/30/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Attention Mechanisms for Energy-Information Co-Optimization

Implements:
1. CausalAttention: Respects causal ordering in control DAG
2. CrossDomainAttention: Energy-to-Communication attention with M_physics mask
3. PhysicsMask: Impedance-weighted attention bias

Key formulas:
    α_ij = softmax(q_i^T · k_j / √d + M_causal[i,j] + M_physics[i,j])

    M_causal[i,j] = -∞ if j is not a causal ancestor of i
    M_physics[i,j] = -γ · Z_ij / Z_max (impedance-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PhysicsMask(nn.Module):
    """
    Physics-informed attention mask based on electrical impedance (V2).

    M_physics[i,j] = -γ · Z_ij / Z_max

    V2 improvements (addresses Q3.3 — attention entropy = 0):
    - Adaptive gamma: learnable parameter that scales with attention logits
    - When adaptive=True, gamma adjusts so mask doesn't overwhelm QK scores
    - This prevents one-hot attention distributions (entropy = 0)
    """

    def __init__(self, gamma: float = 1.0, adaptive: bool = False):
        """
        Args:
            gamma: Initial strength of physics mask
            adaptive: If True, gamma is a learnable parameter that scales
                      relative to attention logit magnitude
        """
        super().__init__()
        self.adaptive = adaptive

        if adaptive:
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma)))
        else:
            self.gamma = gamma

    def get_gamma(self) -> torch.Tensor:
        """Get current gamma value."""
        if self.adaptive:
            return torch.exp(self.log_gamma)
        return torch.tensor(self.gamma)

    def forward(
        self,
        impedance_matrix: torch.Tensor,
        attn_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute physics mask from impedance matrix.

        Args:
            impedance_matrix: Impedance between all bus pairs [N, N]
            attn_logits: Optional QK attention logits for adaptive scaling.
                         When provided and adaptive=True, mask is scaled to be
                         the same order of magnitude as attention scores.

        Returns:
            M_physics: Attention mask [N, N]
        """
        gamma = torch.exp(self.log_gamma) if self.adaptive else self.gamma

        # Normalize by max impedance
        Z_max = impedance_matrix.max() + 1e-8
        Z_normalized = impedance_matrix / Z_max

        # Base mask
        M_physics = -gamma * Z_normalized

        # Adaptive scaling: match mask magnitude to attention logit scale
        if self.adaptive and attn_logits is not None:
            attn_scale = attn_logits.abs().mean().detach().clamp(min=1e-4)
            mask_scale = M_physics.abs().mean().detach().clamp(min=1e-4)
            # Scale mask so its magnitude is proportional to attention logits
            M_physics = M_physics * (attn_scale / mask_scale)

        return M_physics

    def from_edge_index(
        self,
        edge_index: torch.Tensor,
        edge_impedance: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Build impedance matrix from sparse edge representation.

        Args:
            edge_index: [2, E]
            edge_impedance: [E]
            num_nodes: N

        Returns:
            M_physics: [N, N]
        """
        # Initialize with large impedance (no direct connection)
        Z = torch.ones(num_nodes, num_nodes, device=edge_index.device) * 1e6

        # Fill in actual impedances
        row, col = edge_index
        Z[row, col] = edge_impedance
        Z[col, row] = edge_impedance  # Symmetric

        # Self-connections have zero impedance
        Z.diagonal().fill_(0)

        return self.forward(Z)


class CausalMask(nn.Module):
    """
    Causal attention mask respecting DAG structure.

    M_causal[i,j] = 0 if j ∈ Ancestors(i) in the causal DAG
    M_causal[i,j] = -∞ otherwise

    This prevents information leakage from effects to causes.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        ancestor_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal mask from ancestor matrix.

        Args:
            ancestor_matrix: Boolean [N, N] where [i,j] = True if j is ancestor of i

        Returns:
            M_causal: Attention mask [N, N]
        """
        # Where not ancestor, mask with -inf
        M_causal = torch.where(
            ancestor_matrix,
            torch.zeros_like(ancestor_matrix, dtype=torch.float),
            torch.full_like(ancestor_matrix, float('-inf'), dtype=torch.float)
        )

        return M_causal

    def from_dag(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Build ancestor matrix from DAG edges using transitive closure.

        Args:
            edge_index: DAG edges [2, E] where edge (u, v) means u -> v
            num_nodes: N

        Returns:
            M_causal: [N, N]
        """
        device = edge_index.device

        # Build adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        row, col = edge_index
        adj[row, col] = True

        # Transitive closure via matrix powers
        # ancestor[i,j] = True if there's a path from j to i
        ancestor = adj.clone()
        power = adj.clone()

        for _ in range(num_nodes - 1):
            power = power @ adj
            ancestor = ancestor | power
            if ancestor.all():
                break

        # Include self as ancestor
        ancestor.diagonal().fill_(True)

        return self.forward(ancestor)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional physics and causal masks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: [batch, seq_q, embed_dim]
            key: [batch, seq_k, embed_dim]
            value: [batch, seq_k, embed_dim]
            attn_mask: [seq_q, seq_k] or [batch, seq_q, seq_k]
            key_padding_mask: [batch, seq_k]

        Returns:
            output: [batch, seq_q, embed_dim]
            attn_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project
        Q = self.q_proj(query)  # [B, Q, D]
        K = self.k_proj(key)    # [B, K, D]
        V = self.v_proj(value)  # [B, K, D]

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [B, H, seq, head_dim]

        # For large sequences (>=1000 nodes), use chunked attention to avoid OOM
        CHUNK_THRESHOLD = 1000
        if seq_q > CHUNK_THRESHOLD or seq_k > CHUNK_THRESHOLD:
            return self._chunked_attention(
                Q, K, V, attn_mask, key_padding_mask,
                batch_size, seq_q, seq_k,
            )

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Q, K]

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, Q, K]
            attn = attn + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, V)  # [B, H, Q, head_dim]

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights

    def _chunked_attention(
        self, Q, K, V, attn_mask, key_padding_mask,
        batch_size, seq_q, seq_k,
    ):
        """Memory-efficient chunked attention for large sequences (>=1000 nodes)."""
        # Adaptive chunk size: for very large sequences, use smaller chunks
        # Memory per chunk ≈ batch_size * num_heads * chunk_size * seq_k * 4 bytes
        if seq_k > 5000:
            chunk_size = 64
        elif seq_k > 2000:
            chunk_size = 128
        else:
            chunk_size = 512
        output_chunks = []

        for q_start in range(0, seq_q, chunk_size):
            q_end = min(q_start + chunk_size, seq_q)
            Q_chunk = Q[:, :, q_start:q_end, :]  # [B, H, chunk, D]

            # Compute attention for this query chunk against all keys
            attn_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) * self.scale

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    mask_chunk = attn_mask[q_start:q_end, :].unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    mask_chunk = attn_mask[:, q_start:q_end, :].unsqueeze(1)
                else:
                    mask_chunk = attn_mask[:, :, q_start:q_end, :]
                attn_chunk = attn_chunk + mask_chunk

            if key_padding_mask is not None:
                attn_chunk = attn_chunk.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
                )

            attn_weights_chunk = F.softmax(attn_chunk, dim=-1)
            attn_weights_chunk = self.dropout(attn_weights_chunk)
            out_chunk = torch.matmul(attn_weights_chunk, V)
            output_chunks.append(out_chunk)

        output = torch.cat(output_chunks, dim=2)  # [B, H, seq_q, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.embed_dim)
        output = self.out_proj(output)
        return output, None  # No full attn_weights for large sequences


class CausalAttention(nn.Module):
    """
    Attention that respects causal ordering in the control DAG.

    α_ij = softmax(q_i^T · k_j / √d + M_causal[i,j])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.causal_mask_builder = CausalMask()

    def forward(
        self,
        x: torch.Tensor,
        dag_edge_index: Optional[torch.Tensor] = None,
        ancestor_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal masking.

        Args:
            x: Input features [batch, seq, embed_dim]
            dag_edge_index: DAG edges [2, E]
            ancestor_matrix: Precomputed ancestor matrix [seq, seq]

        Returns:
            output: [batch, seq, embed_dim]
            attn_weights: [batch, num_heads, seq, seq]
        """
        seq_len = x.shape[1]

        # Build causal mask
        if ancestor_matrix is not None:
            M_causal = self.causal_mask_builder(ancestor_matrix)
        elif dag_edge_index is not None:
            M_causal = self.causal_mask_builder.from_dag(dag_edge_index, seq_len)
        else:
            # No causal constraint
            M_causal = None

        return self.mha(x, x, x, attn_mask=M_causal)


class CrossDomainAttention(nn.Module):
    """
    Cross-domain attention from Energy to Communication.

    Energy queries attend to Communication keys with physics mask:
    α_ij = softmax(q_E,i^T · k_I,j / √d + M_physics[i,j])

    V2: Supports adaptive gamma to prevent zero-entropy attention (Q3.3).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gamma: float = 1.0,
        adaptive_gamma: bool = False,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.physics_mask = PhysicsMask(gamma, adaptive=adaptive_gamma)
        self.adaptive_gamma = adaptive_gamma

    def forward(
        self,
        h_E: torch.Tensor,
        h_I: torch.Tensor,
        impedance_matrix: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_impedance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Cross-domain attention from energy to communication.

        Args:
            h_E: Energy embeddings [batch, N, embed_dim]
            h_I: Communication embeddings [batch, N, embed_dim]
            impedance_matrix: Full impedance matrix [N, N]
            edge_index: Sparse edge representation [2, E]
            edge_impedance: Edge impedances [E]

        Returns:
            output: [batch, N, embed_dim]
            attn_info: Dict with attn_weights and entropy
        """
        num_nodes = h_E.shape[1]

        # For adaptive mask, estimate attention scale from input norms
        attn_logits_estimate = None
        if self.adaptive_gamma:
            # Approximate QK scale: ||h_E|| * ||h_I|| / sqrt(d)
            e_norm = h_E.norm(dim=-1).mean()
            i_norm = h_I.norm(dim=-1).mean()
            d_k = self.mha.head_dim
            attn_logits_estimate = (e_norm * i_norm / math.sqrt(d_k)).unsqueeze(0).unsqueeze(0)

        # Build physics mask
        if impedance_matrix is not None:
            M_physics = self.physics_mask(impedance_matrix, attn_logits=attn_logits_estimate)
        elif edge_index is not None and edge_impedance is not None:
            M_physics = self.physics_mask.from_edge_index(
                edge_index, edge_impedance, num_nodes
            )
        else:
            M_physics = None

        # Energy queries, Communication keys/values
        output, attn_weights = self.mha(h_E, h_I, h_I, attn_mask=M_physics)

        # Compute attention entropy for monitoring
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean()

        attn_info = {
            "attn_weights": attn_weights,
            "entropy": entropy.item(),
            "gamma": self.physics_mask.get_gamma().item(),
        }

        return output, attn_info


class HierarchicalAttention(nn.Module):
    """
    Full attention module combining causal and cross-domain attention.

    1. Self-attention within energy domain (with causal mask)
    2. Cross-attention from energy to communication (with physics mask)
    3. Fusion of both attention outputs
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gamma: float = 1.0,
        adaptive_gamma: bool = False,
    ):
        super().__init__()

        # Causal self-attention for energy
        self.causal_attn = CausalAttention(embed_dim, num_heads, dropout)

        # Cross-domain attention (with adaptive gamma support for V2)
        self.cross_attn = CrossDomainAttention(
            embed_dim, num_heads, dropout, gamma, adaptive_gamma=adaptive_gamma
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h_E: torch.Tensor,
        h_I: torch.Tensor,
        dag_edge_index: Optional[torch.Tensor] = None,
        impedance_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            h_E: Energy embeddings [batch, N, embed_dim]
            h_I: Communication embeddings [batch, N, embed_dim]
            dag_edge_index: Causal DAG edges
            impedance_matrix: Impedance for physics mask

        Returns:
            h_fused: Fused embeddings [batch, N, embed_dim]
            attn_info: Dict with attention weights
        """
        # Causal self-attention on energy
        h_causal, attn_causal = self.causal_attn(h_E, dag_edge_index=dag_edge_index)
        h_E = self.norm1(h_E + h_causal)

        # Cross-domain attention (returns dict with attn_weights + entropy)
        h_cross, cross_attn_info = self.cross_attn(
            h_E, h_I, impedance_matrix=impedance_matrix
        )

        # Fusion
        h_fused = self.fusion(torch.cat([h_E, h_cross], dim=-1))
        h_fused = self.norm2(h_fused + self.ffn(h_fused))

        attn_info = {
            "causal_attn": attn_causal,
            "cross_attn": cross_attn_info.get("attn_weights"),
            "cross_attn_entropy": cross_attn_info.get("entropy", 0.0),
            "physics_gamma": cross_attn_info.get("gamma", 0.0),
        }

        return h_fused, attn_info
