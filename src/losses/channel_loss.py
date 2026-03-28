#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 03/09/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Channel State Loss (L_channel)

Cross-entropy loss for Markov channel state prediction.
The model predicts the current channel state from communication embeddings;
this auxiliary task helps the model learn channel-aware representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ChannelStateLoss(nn.Module):
    """
    Cross-entropy loss for channel state prediction.

    L_channel = CE(predicted_state_logits, true_state)

    Averaged over all generator nodes (non-generator nodes have state=0
    and are masked out).
    """

    def __init__(self, n_states: int = 3):
        super().__init__()
        self.n_states = n_states
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        channel_pred: torch.Tensor,
        channel_state: torch.Tensor,
        gen_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute channel state prediction loss.

        Args:
            channel_pred: Predicted logits [batch, N, n_states] from model
            channel_state: True state indices [batch, n_gen] (0/1/2)
            gen_mask: Boolean mask [N] indicating generator buses (optional).
                      If None, uses all nodes (less accurate but simpler).

        Returns:
            loss: Scalar loss
            components: Dict with loss value and accuracy
        """
        batch_size = channel_pred.shape[0]

        if gen_mask is not None:
            # Only compute loss on generator nodes
            pred_gen = channel_pred[:, gen_mask, :]  # [batch, n_gen, n_states]
        else:
            # Use first n_gen nodes as approximation
            n_gen = channel_state.shape[-1]
            pred_gen = channel_pred[:, :n_gen, :]  # [batch, n_gen, n_states]

        # Reshape for CE: [batch*n_gen, n_states] vs [batch*n_gen]
        pred_flat = pred_gen.reshape(-1, self.n_states)
        target_flat = channel_state.reshape(-1).long()

        # Cross-entropy
        per_sample_loss = self.ce_loss(pred_flat, target_flat)
        loss = per_sample_loss.mean()

        # Accuracy
        with torch.no_grad():
            predicted_classes = pred_flat.argmax(dim=-1)
            accuracy = (predicted_classes == target_flat).float().mean().item()

        components = {
            'L_channel': loss.item(),
            'channel_accuracy': accuracy,
        }

        return loss, components
