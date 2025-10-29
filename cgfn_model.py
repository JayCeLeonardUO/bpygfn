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


def test_conv_tb_model_with_trajectory():
    """Test the model with real trajectory data"""
    from single_color_ramp import load_blend_single_color_ramp, sample_random_trajectory
    from blender_setup_utils import BlenderTensorUtility
    import bpy

    print("\n" + "=" * 70)
    print("TESTING ConvTBModel WITH TRAJECTORY DATA")
    print("=" * 70)

    # Sample a trajectory
    max_colors = 5
    trajectory_len = 6
    trajectory = sample_random_trajectory(trajectory_len, max_colors)

    # Get dimensions from trajectory
    action_dim = registry.total_actions
    heightmap_size = trajectory['heightmaps'][0].shape[0]

    print(f"\nTrajectory info:")
    print(f"  Actions taken: {trajectory['num_actions']}")
    print(f"  Action space dimensions: {action_dim}")
    print(f"  Heightmap size: {heightmap_size}x{heightmap_size}")

    # Create model
    action_history_size = 10
    model = ConvTBModel(
        heightmap_channels=1,
        heightmap_size=heightmap_size,
        action_dim=action_dim,
        hidden_size=256,
        action_history_size=action_history_size
    )

    print(f"\nModel created:")
    print(f"  Heightmap encoder output: {model.heightmap_encoded_dim}")
    print(f"  Action encoder output: {model.action_encoded_dim}")
    print(f"  Combined state: {model.combined_dim}")
    print(f"  Action history size: {action_history_size}")

    # Test with trajectory data
    num_actions = trajectory['num_actions']

    for step in range(num_actions):
        print(f"\n--- Testing Step {step + 1} ---")

        # Get current heightmap
        heightmap = trajectory['heightmaps'][step]  # [H, W]
        heightmap_batch = heightmap.unsqueeze(0)  # [1, H, W]

        # Build action history (all previous actions)
        if step == 0:
            # First step: no history, use zeros
            action_history = torch.zeros(1, action_history_size, action_dim)
        else:
            # Get all previous action tensors
            prev_actions = trajectory['action_tensors'][:step]  # [step, action_dim]

            # Pad or truncate to action_history_size
            if step < action_history_size:
                # Pad with zeros
                padding = torch.zeros(action_history_size - step, action_dim)
                action_history = torch.cat([padding, prev_actions], dim=0)
            else:
                # Take last action_history_size actions
                action_history = prev_actions[-action_history_size:]

            action_history = action_history.unsqueeze(0)  # [1, history_len, action_dim]

        print(f"  Heightmap shape: {heightmap_batch.shape}")
        print(f"  Action history shape: {action_history.shape}")
        print(f"  Action history non-zero: {(action_history != 0).any(dim=2).sum().item()}/{action_history_size}")

        # Forward pass
        P_F_logits, P_B_logits = model(heightmap_batch, action_history)

        print(f"  Forward policy logits shape: {P_F_logits.shape}")
        print(f"  Backward policy logits shape: {P_B_logits.shape}")

        # Get action mask for this step
        mask = trajectory['action_masks'][step]

        # Apply mask to logits
        masked_P_F_logits = P_F_logits.clone()
        masked_P_F_logits[0, ~mask] = float('-inf')

        # Sample action
        P_F_probs = torch.softmax(masked_P_F_logits, dim=-1)
        valid_probs = P_F_probs[0, mask]

        print(f"  Valid action probabilities sum: {valid_probs.sum().item():.4f}")
        print(f"  Max valid probability: {valid_probs.max().item():.4f}")

        assert P_F_logits.shape == (1, action_dim)
        assert P_B_logits.shape == (1, action_dim)

    print("\n✓ All trajectory steps tested successfully!")


def test_model_with_batched_data():
    """Test model with batched trajectory data"""
    from single_color_ramp import sample_random_trajectory

    print("\n" + "=" * 70)
    print("TESTING BATCHED MODEL INPUT")
    print("=" * 70)

    # Sample multiple trajectories
    max_colors = 5
    num_trajectories = 4

    trajectories = []
    for i in range(num_trajectories):
        print(f"\nSampling trajectory {i + 1}/{num_trajectories}...")
        traj = sample_random_trajectory(trajectory_len=5, max_colors=max_colors)
        trajectories.append(traj)

    # Get dimensions
    action_dim = registry.total_actions
    heightmap_size = trajectories[0]['heightmaps'][0].shape[0]
    action_history_size = 10

    # Create model
    model = ConvTBModel(
        heightmap_channels=1,
        heightmap_size=heightmap_size,
        action_dim=action_dim,
        hidden_size=256,
        action_history_size=action_history_size
    )

    print(f"\nPreparing batch of {num_trajectories} trajectory steps...")

    # Get one step from each trajectory (e.g., step 2)
    step_idx = 2

    batch_heightmaps = []
    batch_action_histories = []

    for traj in trajectories:
        if step_idx < traj['num_actions']:
            # Heightmap
            heightmap = traj['heightmaps'][step_idx]
            batch_heightmaps.append(heightmap)

            # Action history
            if step_idx == 0:
                action_history = torch.zeros(action_history_size, action_dim)
            else:
                prev_actions = traj['action_tensors'][:step_idx]
                if step_idx < action_history_size:
                    padding = torch.zeros(action_history_size - step_idx, action_dim)
                    action_history = torch.cat([padding, prev_actions], dim=0)
                else:
                    action_history = prev_actions[-action_history_size:]

            batch_action_histories.append(action_history)

    # Stack into batches
    heightmap_batch = torch.stack(batch_heightmaps)  # [batch, H, W]
    action_history_batch = torch.stack(batch_action_histories)  # [batch, history_len, action_dim]

    print(f"  Heightmap batch shape: {heightmap_batch.shape}")
    print(f"  Action history batch shape: {action_history_batch.shape}")

    # Forward pass
    P_F_logits, P_B_logits = model(heightmap_batch, action_history_batch)

    print(f"\n  Forward policy logits shape: {P_F_logits.shape}")
    print(f"  Backward policy logits shape: {P_B_logits.shape}")

    assert P_F_logits.shape == (len(batch_heightmaps), action_dim)
    assert P_B_logits.shape == (len(batch_heightmaps), action_dim)

    print("\n✓ Batched input test passed!")


def test_model_masking_integration():
    """Test that model outputs work correctly with action masking"""
    from gfn_environments.single_color_ramp import sample_random_trajectory

    from gfn_environments.single_color_ramp import registry
    print("\n" + "=" * 70)
    print("TESTING MODEL WITH ACTION MASKING")
    print("=" * 70)

    # Sample trajectory
    trajectory = sample_random_trajectory(trajectory_len=5, max_colors=5)

    # Create model
    action_dim = registry.total_actions
    heightmap_size = trajectory['heightmaps'][0].shape[0]

    model = ConvTBModel(
        heightmap_channels=1,
        heightmap_size=heightmap_size,
        action_dim=action_dim,
        hidden_size=128,
        action_history_size=5
    )

    print(f"\nTesting masking at each step...")

    for step in range(trajectory['num_actions']):
        # Prepare inputs
        heightmap = trajectory['heightmaps'][step].unsqueeze(0)

        if step == 0:
            action_history = torch.zeros(1, 5, action_dim)
        else:
            prev_actions = trajectory['action_tensors'][:step]
            if step < 5:
                padding = torch.zeros(5 - step, action_dim)
                action_history = torch.cat([padding, prev_actions], dim=0).unsqueeze(0)
            else:
                action_history = prev_actions[-5:].unsqueeze(0)

        # Get mask
        mask = trajectory['action_masks'][step]

        # Forward pass
        P_F_logits, _ = model(heightmap, action_history)

        # Apply mask
        masked_logits = P_F_logits.clone()
        masked_logits[0, ~mask] = float('-inf')

        # Check that we can sample valid actions
        probs = torch.softmax(masked_logits, dim=-1)

        # Verify only valid actions have non-zero probability
        valid_probs = probs[0, mask]
        invalid_probs = probs[0, ~mask]

        print(f"  Step {step + 1}:")
        print(f"    Valid actions: {mask.sum().item()}")
        print(f"    Valid prob sum: {valid_probs.sum().item():.4f}")
        print(f"    Invalid prob sum: {invalid_probs.sum().item():.10f}")

        assert valid_probs.sum().item() > 0.99  # Should sum to ~1
        assert invalid_probs.sum().item() < 0.01  # Should be ~0

    print("\n✓ Masking integration test passed!")


if __name__ == "__main__":
    test_conv_tb_model_with_trajectory()
    test_model_with_batched_data()
    test_model_masking_integration()