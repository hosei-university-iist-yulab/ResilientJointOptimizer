#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Created on 01/26/2026🚀

Author: Franck Aboya
Email: franckjunioraboya.messou@ieee.org
Github: https://github.com/mesabo
Univ: Hosei University, PhD
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""Loss functions for Energy-Information Co-Optimization."""

from .coupling_loss import CouplingLoss, LogBarrierStabilityLoss
from .energy_loss import EnergyLoss
from .communication_loss import CommunicationLoss
from .contrastive import PhysicsAwareContrastiveLoss
from .combined import JointLoss

# Topic 2: Multi-impairment losses (NEW — base classes untouched)
from .channel_loss import ChannelStateLoss
from .resilient_loss import ResilientJointLoss

__all__ = [
    # Base paper losses (untouched)
    "CouplingLoss",
    "LogBarrierStabilityLoss",
    "EnergyLoss",
    "CommunicationLoss",
    "PhysicsAwareContrastiveLoss",
    "JointLoss",
    # Topic 2 losses (NEW)
    "ChannelStateLoss",
    "ResilientJointLoss",
]
