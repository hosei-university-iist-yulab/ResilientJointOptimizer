#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 02/03/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Baseline Models for Comparison

Required baselines from REQUIREMENTS.md:
- B1: Sequential OPF + QoS (decoupled optimization)
- B2: MLP Joint (simple feedforward)
- B3: GNN-only (no attention)
- B4: LSTM Joint (recurrent)
- B5: CNN Joint (convolutional)
- B6: Vanilla Transformer (standard attention)
- B7: Transformer (no L_coupling loss)

V2 additions:
- B8: Heterogeneous GNN (Q3.1)
- B9: DeepOPF (Q5.2)
"""

from .sequential_opf import SequentialOPFQoS
from .mlp_joint import MLPJoint
from .gnn_only import GNNOnly
from .lstm_joint import LSTMJoint
from .cnn_joint import CNNJoint
from .vanilla_transformer import VanillaTransformer
from .transformer_no_coupling import TransformerNoCoupling
from .heterogeneous_gnn import HeterogeneousGNN
from .deepopf import DeepOPF

# Topic 2: New baselines (B10-B12, B-ROPF, B-SOPF, B-Hinf)
from .delay_only_joint import DelayOnlyJointOptimizer
from .naive_multi_impairment import NaiveMultiImpairment
from .tcp_retransmit import TCPRetransmitModel
from .robust_opf import RobustOPF
from .stochastic_opf import StochasticOPF
from .hinf_controller import HInfController

__all__ = [
    # Base paper baselines (B1-B9, untouched)
    "SequentialOPFQoS",
    "MLPJoint",
    "GNNOnly",
    "LSTMJoint",
    "CNNJoint",
    "VanillaTransformer",
    "TransformerNoCoupling",
    "HeterogeneousGNN",
    "DeepOPF",
    # Topic 2 baselines (NEW)
    "DelayOnlyJointOptimizer",   # B10
    "NaiveMultiImpairment",      # B11
    "TCPRetransmitModel",        # B12
    "RobustOPF",                 # B-ROPF
    "StochasticOPF",             # B-SOPF
    "HInfController",            # B-Hinf
]
