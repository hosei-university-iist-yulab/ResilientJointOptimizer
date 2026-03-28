#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/27/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Contrastive Loss: L_align

Physics-aware contrastive learning to align energy and communication embeddings.

L_align = -log(exp(sim(h_E, h_I⁺)/T) / Σ exp(sim(h_E, h_I)/T))

where:
- h_E: Energy domain embedding
- h_I⁺: Positive communication embedding (same node)
- h_I: All communication embeddings (including negatives)
- T: Temperature parameter
- sim(): Cosine similarity

Key innovation: Negatives are weighted by electrical distance, making
the model learn physically meaningful alignments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss.

    Standard contrastive loss used in CLIP, SimCLR, etc.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Softmax temperature
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        h_query: torch.Tensor,
        h_key: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            h_query: Query embeddings [batch, dim]
            h_key: Key embeddings [batch, dim]
            labels: Positive pair labels [batch] (optional, default: diagonal)

        Returns:
            loss: InfoNCE loss (scalar)
        """
        # Normalize embeddings
        h_query = F.normalize(h_query, dim=-1)
        h_key = F.normalize(h_key, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(h_query, h_key.T) / self.temperature

        # Default labels: diagonal (same index = positive pair)
        if labels is None:
            labels = torch.arange(h_query.shape[0], device=h_query.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class PhysicsAwareContrastiveLoss(nn.Module):
    """
    Physics-aware contrastive loss.

    Weights negative samples by electrical distance:
    w_ij = exp(-γ · d_ij / d_max)

    where d_ij is the electrical distance between nodes i and j.
    This makes electrically close nodes harder negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        gamma: float = 1.0,
        hard_negative_weight: float = 1.0,
    ):
        """
        Args:
            temperature: Softmax temperature
            gamma: Electrical distance scaling
            hard_negative_weight: Weight for hard negatives
        """
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        h_E: torch.Tensor,
        h_I: torch.Tensor,
        impedance_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-aware contrastive loss.

        Args:
            h_E: Energy embeddings [batch, N, dim] or [N, dim]
            h_I: Communication embeddings [batch, N, dim] or [N, dim]
            impedance_matrix: Impedance between nodes [N, N]

        Returns:
            loss: Contrastive loss (scalar)
            info: Dict with additional metrics
        """
        # Handle different input shapes
        if h_E.dim() == 3:
            batch_size, N, dim = h_E.shape
            # For large grids, compute per-sample to avoid OOM on [batch*N, batch*N] matmul
            if batch_size * N > 10000:
                return self._forward_chunked(h_E, h_I, impedance_matrix)
            # [batch, N, dim] -> [batch*N, dim]
            h_E = h_E.reshape(-1, dim)
            h_I = h_I.reshape(-1, dim)
        else:
            N = h_E.shape[0]
            batch_size = 1

        # Normalize embeddings
        h_E = F.normalize(h_E, dim=-1)
        h_I = F.normalize(h_I, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(h_E, h_I.T) / self.temperature

        # Positive pairs: diagonal (same node across domains)
        labels = torch.arange(h_E.shape[0], device=h_E.device)

        # Apply physics-based negative weighting
        if impedance_matrix is not None:
            # Compute electrical distance weights
            Z_max = impedance_matrix.max() + 1e-8
            Z_normalized = impedance_matrix / Z_max

            # Higher weight for electrically close (low impedance) nodes
            weights = torch.exp(-self.gamma * Z_normalized)

            # Expand weights for batched input
            if batch_size > 1:
                weights = weights.repeat(batch_size, batch_size)

            # Apply weights to logits (amplify hard negatives)
            # Don't weight positive pairs (diagonal)
            mask = 1 - torch.eye(h_E.shape[0], device=h_E.device)
            weighted_logits = logits + self.hard_negative_weight * torch.log(weights + 1e-8) * mask
        else:
            weighted_logits = logits

        # Cross-entropy loss
        loss = F.cross_entropy(weighted_logits, labels)

        # Compute accuracy (for monitoring)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()

            # Average positive/negative similarity
            pos_sim = logits.diagonal().mean().item()
            neg_sim = (logits.sum() - logits.diagonal().sum()) / (logits.numel() - logits.shape[0])
            neg_sim = neg_sim.item()

        info = {
            'L_align': loss.item(),
            'contrastive_accuracy': accuracy,
            'pos_similarity': pos_sim * self.temperature,
            'neg_similarity': neg_sim * self.temperature,
        }

        return loss, info

    def _forward_chunked(
        self,
        h_E: torch.Tensor,
        h_I: torch.Tensor,
        impedance_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Per-sample contrastive loss for large grids to avoid OOM."""
        batch_size, N, dim = h_E.shape
        losses = []
        accs = []
        for b in range(batch_size):
            e = F.normalize(h_E[b], dim=-1)
            i = F.normalize(h_I[b], dim=-1)
            logits = torch.matmul(e, i.T) / self.temperature
            labels = torch.arange(N, device=h_E.device)
            if impedance_matrix is not None:
                Z_max = impedance_matrix.max() + 1e-8
                weights = torch.exp(-self.gamma * impedance_matrix / Z_max)
                mask = 1 - torch.eye(N, device=h_E.device)
                logits = logits + self.hard_negative_weight * torch.log(weights + 1e-8) * mask
            losses.append(F.cross_entropy(logits, labels))
            with torch.no_grad():
                accs.append((logits.argmax(dim=-1) == labels).float().mean().item())
        loss = torch.stack(losses).mean()
        info = {
            'L_align': loss.item(),
            'contrastive_accuracy': sum(accs) / len(accs),
            'pos_similarity': 0.0,
            'neg_similarity': 0.0,
        }
        return loss, info


class DomainAlignmentLoss(nn.Module):
    """
    Domain alignment loss using both directions.

    L = 0.5 * (L_E→I + L_I→E)

    Bidirectional contrastive loss for better alignment.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.loss_E_to_I = PhysicsAwareContrastiveLoss(temperature, gamma)
        self.loss_I_to_E = PhysicsAwareContrastiveLoss(temperature, gamma)

    def forward(
        self,
        h_E: torch.Tensor,
        h_I: torch.Tensor,
        impedance_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute bidirectional domain alignment loss.

        Args:
            h_E: Energy embeddings
            h_I: Communication embeddings
            impedance_matrix: Impedance between nodes

        Returns:
            loss: Total alignment loss
            info: Dict with metrics
        """
        # Energy queries, Communication keys
        loss_E_I, info_E_I = self.loss_E_to_I(h_E, h_I, impedance_matrix)

        # Communication queries, Energy keys
        loss_I_E, info_I_E = self.loss_I_to_E(h_I, h_E, impedance_matrix)

        # Average both directions
        loss = 0.5 * (loss_E_I + loss_I_E)

        info = {
            'L_align': loss.item(),
            'L_E_to_I': loss_E_I.item(),
            'L_I_to_E': loss_I_E.item(),
            'contrastive_accuracy': 0.5 * (
                info_E_I['contrastive_accuracy'] + info_I_E['contrastive_accuracy']
            ),
        }

        return loss, info
