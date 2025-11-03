from torch import nn
import torch
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import pytest

class TBModel(nn.Module):
    """Trajectory Balance GFlowNet Model - mirroring HyperGrid implementation"""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()

        # Forward policy: current state -> next action probabilities
        self.forward_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Backward policy: current state -> previous action probabilities
        self.backward_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Log partition function
        self.logZ = nn.Parameter(torch.tensor(5.0))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns both forward and backward policy logits"""
        P_F_logits = self.forward_policy(state)
        P_B_logits = self.backward_policy(state)
        return P_F_logits, P_B_logits


import torch
import torch.nn as nn
from typing import Tuple

import torch
import torch.nn as nn
from typing import Tuple


class ConvTBModel(nn.Module):
    """Convolutional Trajectory Balance GFlowNet Model for heightmap and action history"""

    def __init__(
            self,
            heightmap_channels: int = 1,
            heightmap_size: int = 16,
            action_dim: int = 32,  # Total action space dimensions
            hidden_size: int = 256,
            action_history_size: int = 10  # Max number of previous actions to consider
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_history_size = action_history_size

        # Convolutional encoder for heightmap
        self.heightmap_encoder = nn.Sequential(
            nn.Conv2d(heightmap_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # heightmap_size/2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # heightmap_size/4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling -> [batch, 128, 1, 1]
            nn.Flatten()  # -> [batch, 128]
        )

        self.heightmap_encoded_dim = 128

        # Action history encoder
        # Takes concatenated action history and encodes it
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * action_history_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.action_encoded_dim = 64

        # Combined state dimension
        self.combined_dim = self.heightmap_encoded_dim + self.action_encoded_dim

        # Forward policy: combined state -> next action probabilities
        self.forward_policy = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Backward policy: combined state -> previous action probabilities
        self.backward_policy = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        # Log partition function
        self.logZ = nn.Parameter(torch.tensor(5.0))

    def forward(
            self,
            heightmap: torch.Tensor,
            action_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns both forward and backward policy logits

        Args:
            heightmap: [batch, channels, height, width] or [batch, height, width]
            action_history: [batch, history_len, action_dim] - history of action tensors

        Returns:
            P_F_logits: Forward policy logits [batch, action_dim]
            P_B_logits: Backward policy logits [batch, action_dim]
        """
        # Handle [batch, H, W] input by adding channel dimension
        if heightmap.dim() == 3:
            heightmap = heightmap.unsqueeze(1)  # [batch, 1, H, W]

        # Encode heightmap to latent representation
        heightmap_encoded = self.heightmap_encoder(heightmap)  # [batch, 128]

        # Process action history
        batch_size = heightmap.shape[0]

        # Flatten action history: [batch, history_len, action_dim] -> [batch, history_len * action_dim]
        action_history_flat = action_history.view(batch_size, -1)

        # Encode action history
        action_encoded = self.action_encoder(action_history_flat)  # [batch, 64]

        # Combine encodings
        combined_state = torch.cat([heightmap_encoded, action_encoded], dim=1)  # [batch, 192]

        # Get policy logits
        P_F_logits = self.forward_policy(combined_state)
        P_B_logits = self.backward_policy(combined_state)

        return P_F_logits, P_B_logits
