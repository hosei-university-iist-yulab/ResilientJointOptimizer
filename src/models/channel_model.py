#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/06/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Channel State Encoder

MLP that encodes discrete Markov channel states into continuous embeddings.
Used by ResilientJointOptimizer to incorporate channel state information
into the communication GNN embeddings.

States: 0=GOOD, 1=DEGRADED, 2=FAILED
"""

import torch
import torch.nn as nn


class ChannelStateEncoder(nn.Module):
    """
    Encode discrete Markov channel state into a continuous embedding.

    Maps one-hot(state) -> hidden -> embed_dim via a small MLP.
    The embedding is added to communication node features before GNN processing.
    """

    def __init__(
        self,
        n_states: int = 3,
        embed_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_states = n_states
        self.encoder = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode channel states.

        Args:
            state_indices: Channel state indices [N] with values in {0, ..., n_states-1}

        Returns:
            embeddings: Channel state embeddings [N, embed_dim]
        """
        one_hot = torch.nn.functional.one_hot(
            state_indices.long(), num_classes=self.n_states,
        ).float()  # [N, n_states]
        return self.encoder(one_hot)


class ChannelStatePredictor(nn.Module):
    """
    Predict next Markov channel state from current embeddings.

    Used for L_channel loss: cross-entropy on state prediction.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_states: int = 3,
    ):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, n_states),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict channel state logits.

        Args:
            h: Node embeddings [N, embed_dim] or [batch, N, embed_dim]

        Returns:
            logits: Channel state logits [N, n_states] or [batch, N, n_states]
        """
        return self.predictor(h)
