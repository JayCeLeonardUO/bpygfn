import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# ====================================================
# HyperGrid Environment
# ====================================================
class HyperGrid:
    """8x8 HyperGrid environment for GFlowNet - All methods use encoded states"""

    def __init__(self, size: int = 8, reward_region_size: int = 2) -> None:
        self.size = size
        self.reward_region_size = reward_region_size

        # Define goal region bounds (computed, not stored)
        self.goal_min_x = size - reward_region_size
        self.goal_min_y = size - reward_region_size

        # Start state as encoded
        self.start_state = tuple(self.encode_raw_state((0, 0)))

        print(f"Grid size: {size}x{size}")
        print(f"Goal region: x >= {self.goal_min_x}, y >= {self.goal_min_y}")
        print(f"State encoding dimension: {self.get_state_dim()}")

    # ====================================================
    # State encoding utilities
    # ====================================================

    def encode_raw_state(self, raw_state: Tuple[int, int]) -> List[float]:
        """Convert raw (x,y) state to encoded one-hot vector"""
        x, y = raw_state
        idx = y * self.size + x
        encoding = [0.0] * (self.size * self.size)
        encoding[idx] = 1.0
        return encoding

    def decode_state_to_raw(
        self, encoded_state: Union[List[float], Tuple[float, ...]]
    ) -> Tuple[int, int]:
        """Convert encoded state back to raw (x,y) tuple"""
        if isinstance(encoded_state, tuple):
            encoded_state = list(encoded_state)

        idx = encoded_state.index(1.0)
        x = idx % self.size
        y = idx // self.size
        return (x, y)

    def get_state_dim(self) -> int:
        """Get the dimension of encoded states"""
        return self.size * self.size

    # ====================================================
    # Actions (work with encoded states)
    # ====================================================

    def get_valid_actions(
        self, encoded_state: Union[List[float], Tuple[float, ...]]
    ) -> List[str]:
        """Get valid actions from an encoded state"""
        # Convert to raw to check boundaries
        x, y = self.decode_state_to_raw(encoded_state)
        actions = []

        if x + 1 < self.size:
            actions.append("right")
        if y + 1 < self.size:
            actions.append("up")
        if x - 1 >= 0:
            actions.append("left")
        if y - 1 >= 0:
            actions.append("down")

        return actions

    def take_action(
        self, encoded_state: Union[List[float], Tuple[float, ...]], action: str
    ) -> Tuple[float, ...]:
        """Take action from encoded state to get next encoded state"""
        # Convert to raw, take action with bounds checking, convert back
        x, y = self.decode_state_to_raw(encoded_state)

        if action == "right" and x + 1 < self.size:
            new_raw = (x + 1, y)
        elif action == "up" and y + 1 < self.size:
            new_raw = (x, y + 1)
        elif action == "left" and x - 1 >= 0:
            new_raw = (x - 1, y)
        elif action == "down" and y - 1 >= 0:
            new_raw = (x, y - 1)
        else:
            # Invalid action - stay in same state
            new_raw = (x, y)

        return tuple(self.encode_raw_state(new_raw))

    # ====================================================
    # Action encoding
    # ====================================================

    def get_action_list(self) -> List[str]:
        """Get list of all possible actions"""
        return ["right", "up", "left", "down"]

    def action_to_index(self, action: str) -> int:
        """Convert action string to index"""
        action_map = {"right": 0, "up": 1, "left": 2, "down": 3}
        return action_map.get(action, -1)

    def index_to_action(self, index: int) -> str:
        """Convert action index to string"""
        actions = ["right", "up", "left", "down"]
        return actions[index] if 0 <= index < len(actions) else "right"

    def get_action_dim(self) -> int:
        """Get number of possible actions"""
        return 4

    # ====================================================
    # Action masking (work with encoded states)
    # ====================================================

    def get_valid_action_mask(
        self, encoded_state: Union[List[float], Tuple[float, ...]]
    ) -> List[float]:
        """Get binary mask for valid actions"""
        valid_actions = self.get_valid_actions(encoded_state)
        mask = [0.0] * 4
        for action in valid_actions:
            idx = self.action_to_index(action)
            if idx >= 0:
                mask[idx] = 1.0
        return mask

    # ====================================================
    # REWARD (work with encoded states)
    # ====================================================

    def get_reward(self, encoded_state: Union[List[float], Tuple[float, ...]]) -> float:
        """Get reward for an encoded state using bounds checking"""
        # Convert to raw coordinates for bounds checking
        x, y = self.decode_state_to_raw(encoded_state)

        # Check if in goal region using bounds
        if x >= self.goal_min_x and y >= self.goal_min_y:
            return 1.0
        return 0.0

    def is_terminal(self, encoded_state: Union[List[float], Tuple[float, ...]]) -> bool:
        """Check if encoded state is terminal"""
        return self.get_reward(encoded_state) > 0.0


# ====================================================
# Trajectory Balance Model
# ====================================================
class TBModel(nn.Module):
    """Trajectory Balance GFlowNet Model"""

    def __init__(self, state_dim: int, num_hid: int = 128):
        super().__init__()

        # ====================================================
        # Forward policy: current state -> next action probabilities
        # ====================================================

        self.forward_policy = nn.Sequential(
            nn.Linear(state_dim, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 4),  # 4 actions: right, up, left, down
        )

        # ====================================================
        # Backward policy: current state -> previous action probabilities
        # ====================================================

        self.backward_policy = nn.Sequential(
            nn.Linear(state_dim, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, 4),  # 4 actions
        )

        # ====================================================
        # Log partition function
        # ====================================================

        self.logZ = nn.Parameter(torch.tensor(5.0))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        P_F_logits = self.forward_policy(state)
        P_B_logits = self.backward_policy(state)
        return P_F_logits, P_B_logits


# ====================================================
# Trajectory Balance Loss Function
# ====================================================
def trajectory_balance_loss(
    model: TBModel, trajectories: List[List[Tuple[float, ...]]], env: HyperGrid
) -> torch.Tensor:
    """
    Optimized Trajectory Balance Loss
    """

    # Early check: Filter for rewarded trajectories upfront
    rewarded_trajectories = []
    for traj in trajectories:
        if len(traj) >= 2:  # Valid length
            final_state = traj[-1]
            reward = env.get_reward(final_state)
            if reward > 0:  # Has reward
                rewarded_trajectories.append((traj, reward))

    # ====================================================
    # Early exit if no valid trajectories
    # ====================================================
    if len(rewarded_trajectories) == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Process only rewarded trajectories
    total_loss = torch.tensor(0.0, requires_grad=True)

    for traj, reward in rewarded_trajectories:
        # ====================================================
        # FORWARD path: Z_θ * ∏P_F(s_t|s_{t-1})
        # ====================================================

        log_forward = model.logZ

        for step in range(len(traj) - 1):
            current_state = traj[step]
            next_state = traj[step + 1]

            # Encode current state for neural network
            current_tensor = torch.tensor(list(current_state), dtype=torch.float)

            # Get forward policy logits
            P_F_logits, _ = model(current_tensor)

            # Mask invalid actions using vectorized operation
            action_mask = torch.tensor(env.get_valid_action_mask(current_state))
            masked_logits = P_F_logits.where(action_mask.bool(), torch.tensor(-100.0))

            # Get probabilities
            probs = F.softmax(masked_logits, dim=0)

            # Find which action was taken
            for action_str in env.get_valid_actions(current_state):
                if env.take_action(current_state, action_str) == next_state:
                    action_idx = env.action_to_index(action_str)
                    log_forward = log_forward + torch.log(
                        probs[action_idx].clamp(min=1e-8)
                    )
                    break

        # ====================================================
        # BACKWARD path: R(x) * ∏P_B(s_{t-1}|s_t)
        # ====================================================

        log_backward = torch.log(
            torch.tensor(reward, dtype=torch.float, requires_grad=True)
        )

        for step in range(len(traj) - 1, 0, -1):
            current_state = traj[step]
            prev_state = traj[step - 1]

            # Encode current state
            current_tensor = torch.tensor(list(current_state), dtype=torch.float)

            # Get backward policy logits
            _, P_B_logits = model(current_tensor)

            # Find valid previous actions (safely with bounds checking)
            valid_prev_actions = []
            for action_str in env.get_action_list():
                # Check if taking this action from prev_state leads to current_state
                test_next_state = env.take_action(prev_state, action_str)
                if test_next_state == current_state:
                    action_idx = env.action_to_index(action_str)
                    valid_prev_actions.append(action_idx)

            # Create mask and apply vectorized masking
            prev_action_mask = torch.zeros(4)
            for action_idx in valid_prev_actions:
                prev_action_mask[action_idx] = 1.0

            masked_logits = P_B_logits.where(
                prev_action_mask.bool(), torch.tensor(-100.0)
            )
            probs = F.softmax(masked_logits, dim=0)

            # Find which action led to current state
            for action_str in env.get_action_list():
                test_next_state = env.take_action(prev_state, action_str)
                if test_next_state == current_state:
                    action_idx = env.action_to_index(action_str)
                    log_backward = log_backward + torch.log(
                        probs[action_idx].clamp(min=1e-8)
                    )
                    break

        # ====================================================
        # Apply trajectory balance equation
        # ====================================================

        trajectory_loss = (log_forward - log_backward) ** 2
        total_loss = total_loss + trajectory_loss

    return total_loss / len(rewarded_trajectories)


# ====================================================
# Visualization Functions
# ====================================================
def visualize_trajectory_grid(  # {{{
    trajectories: List[List[Tuple]],
    env,
    title: str = "Trajectories",
    max_trajectories: int = 10,
    save_path: str = None,
):
    """Visualize trajectories on the grid with success/failure distinction"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create base grid
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)

    # Add grid lines
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color="lightgray", linewidth=0.5)
        ax.axvline(i - 0.5, color="lightgray", linewidth=0.5)

    # Mark start state
    start_raw = env.decode_state_to_raw(env.start_state)
    ax.scatter(
        start_raw[0],
        start_raw[1],
        c="green",
        s=400,
        marker="s",
        label="Start",
        zorder=5,
        edgecolor="black",
        linewidth=2,
    )

    # Mark goal region
    for x in range(max(0, env.goal_min_x), env.size):
        for y in range(max(0, env.goal_min_y), env.size):
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.4, y - 0.4), 0.8, 0.8, facecolor="red", alpha=0.3, zorder=1
                )
            )
    ax.scatter([], [], c="red", s=200, marker="s", alpha=0.3, label="Goal Region")

    # Separate successful and failed trajectories
    successful_trajs = []
    failed_trajs = []

    for traj in trajectories[:max_trajectories]:
        if len(traj) < 2:
            continue
        raw_traj = [env.decode_state_to_raw(state) for state in traj]
        if env.is_terminal(traj[-1]):
            successful_trajs.append(raw_traj)
        else:
            failed_trajs.append(raw_traj)

    # Plot failed trajectories first
    for i, raw_traj in enumerate(failed_trajs):
        xs, ys = zip(*raw_traj)
        ax.plot(
            xs,
            ys,
            color="red",
            linewidth=2,
            alpha=0.6,
            linestyle="--",
            label="Failed" if i == 0 else "",
        )
        ax.scatter(
            xs[-1], ys[-1], color="red", s=100, marker="x", linewidth=3, zorder=4
        )

    # Plot successful trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, len(successful_trajs)))
    for i, (raw_traj, color) in enumerate(zip(successful_trajs, colors)):
        xs, ys = zip(*raw_traj)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=3,
            alpha=0.8,
            label=f"Success {i + 1} (len={len(raw_traj)})",
        )

        ax.scatter(xs[1:-1], ys[1:-1], color=color, s=30, alpha=0.7, zorder=3)
        ax.scatter(
            xs[-1],
            ys[-1],
            color=color,
            s=150,
            marker="o",
            edgecolor="black",
            linewidth=2,
            zorder=4,
        )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(
        f"{title}\nSuccessful: {len(successful_trajs)}, Failed: {len(failed_trajs)}"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_aspect("equal")
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def analyze_trajectory_patterns(
    trajectories: List[List[Tuple]], env, save_path: str = None
):
    """Analyze and visualize trajectory patterns and behaviors"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Convert all trajectories to raw coordinates
    raw_trajectories = []
    successful_indices = []
    for i, traj in enumerate(trajectories):
        raw_traj = [env.decode_state_to_raw(state) for state in traj]
        raw_trajectories.append(raw_traj)
        if env.is_terminal(traj[-1]):
            successful_indices.append(i)

    # 1. Trajectory Length Distribution
    lengths = [len(traj) for traj in raw_trajectories]
    successful_lengths = [len(raw_trajectories[i]) for i in successful_indices]
    failed_lengths = [
        len(traj)
        for i, traj in enumerate(raw_trajectories)
        if i not in successful_indices
    ]

    ax1.hist(
        failed_lengths,
        bins=20,
        alpha=0.6,
        label=f"Failed ({len(failed_lengths)})",
        color="red",
    )
    ax1.hist(
        successful_lengths,
        bins=20,
        alpha=0.6,
        label=f"Successful ({len(successful_lengths)})",
        color="green",
    )
    ax1.set_xlabel("Trajectory Length")
    ax1.set_ylabel("Count")
    ax1.set_title("Trajectory Length Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. State Visit Heatmap
    visit_counts = np.zeros((env.size, env.size))
    for raw_traj in raw_trajectories:
        for x, y in raw_traj:
            visit_counts[y, x] += 1

    im = ax2.imshow(visit_counts, cmap="YlOrRd", origin="lower")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.set_title("State Visitation Heatmap")

    for i in range(env.size):
        for j in range(env.size):
            text = ax2.text(
                j,
                i,
                int(visit_counts[i, j]),
                ha="center",
                va="center",
                color="black"
                if visit_counts[i, j] < visit_counts.max() / 2
                else "white",
            )

    plt.colorbar(im, ax=ax2, label="Visit Count")

    # 3. Final State Distribution
    final_states = [traj[-1] for traj in raw_trajectories]
    final_state_counts = Counter(final_states)

    final_grid = np.zeros((env.size, env.size))
    for (x, y), count in final_state_counts.items():
        final_grid[y, x] = count

    im3 = ax3.imshow(final_grid, cmap="Blues", origin="lower")
    ax3.set_xlabel("X Position")
    ax3.set_ylabel("Y Position")
    ax3.set_title("Final State Distribution")

    # Mark goal region
    for x in range(max(0, env.goal_min_x), env.size):
        for y in range(max(0, env.goal_min_y), env.size):
            rect = plt.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3
            )
            ax3.add_patch(rect)

    for i in range(env.size):
        for j in range(env.size):
            if final_grid[i, j] > 0:
                text = ax3.text(
                    j,
                    i,
                    int(final_grid[i, j]),
                    ha="center",
                    va="center",
                    color="black"
                    if final_grid[i, j] < final_grid.max() / 2
                    else "white",
                )

    plt.colorbar(im3, ax=ax3, label="Final State Count")

    # 4. Action Frequency Analysis
    action_counts = defaultdict(int)

    for traj in trajectories:
        for i in range(len(traj) - 1):
            current_state = traj[i]
            next_state = traj[i + 1]

            for action_str in env.get_valid_actions(current_state):
                if env.take_action(current_state, action_str) == next_state:
                    action_counts[action_str] += 1
                    break

    actions = list(action_counts.keys())
    counts = list(action_counts.values())

    bars = ax4.bar(
        actions, counts, color=["skyblue", "lightgreen", "lightcoral", "lightsalmon"]
    )
    ax4.set_xlabel("Action")
    ax4.set_ylabel("Count")
    ax4.set_title("Action Frequency")
    ax4.grid(True, alpha=0.3)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def visualize_policy_heatmaps(model, env, save_path: str = None):
    """Visualize learned forward and backward policies as heatmaps"""
    model.eval()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    action_names = ["Right", "Up", "Left", "Down"]
    forward_policies = []
    backward_policies = []

    with torch.no_grad():
        for x in range(env.size):
            for y in range(env.size):
                raw_state = (x, y)
                encoded_state = env.encode_raw_state(raw_state)
                state_tensor = torch.tensor(encoded_state, dtype=torch.float)

                P_F_logits, P_B_logits = model(state_tensor)

                action_mask = torch.tensor(
                    env.get_valid_action_mask(tuple(encoded_state))
                )
                masked_F_logits = P_F_logits.where(
                    action_mask.bool(), torch.tensor(-100.0)
                )
                masked_B_logits = P_B_logits.where(
                    action_mask.bool(), torch.tensor(-100.0)
                )

                F_probs = F.softmax(masked_F_logits, dim=0).numpy()
                B_probs = F.softmax(masked_B_logits, dim=0).numpy()

                forward_policies.append(F_probs)
                backward_policies.append(B_probs)

    forward_policies = np.array(forward_policies).reshape(env.size, env.size, 4)
    backward_policies = np.array(backward_policies).reshape(env.size, env.size, 4)

    # Forward policy dominant action
    dominant_forward_action = np.argmax(forward_policies, axis=2)
    im1 = ax1.imshow(dominant_forward_action, cmap="viridis", origin="lower")
    ax1.set_title("Forward Policy - Dominant Action")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")

    for i in range(env.size):
        for j in range(env.size):
            action_idx = dominant_forward_action[i, j]
            ax1.text(
                j,
                i,
                action_names[action_idx][:1],
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_ticks([0, 1, 2, 3])
    cbar1.set_ticklabels(action_names)

    # Backward policy dominant action
    dominant_backward_action = np.argmax(backward_policies, axis=2)
    im2 = ax2.imshow(dominant_backward_action, cmap="plasma", origin="lower")
    ax2.set_title("Backward Policy - Dominant Action")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")

    for i in range(env.size):
        for j in range(env.size):
            action_idx = dominant_backward_action[i, j]
            ax2.text(
                j,
                i,
                action_names[action_idx][:1],
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_ticks([0, 1, 2, 3])
    cbar2.set_ticklabels(action_names)

    # Policy uncertainty (entropy)
    forward_entropy = -np.sum(
        forward_policies * np.log(forward_policies + 1e-8), axis=2
    )
    backward_entropy = -np.sum(
        backward_policies * np.log(backward_policies + 1e-8), axis=2
    )

    im3 = ax3.imshow(forward_entropy, cmap="RdYlBu_r", origin="lower")
    ax3.set_title("Forward Policy Uncertainty (Entropy)")
    ax3.set_xlabel("X Position")
    ax3.set_ylabel("Y Position")
    plt.colorbar(im3, ax=ax3, label="Entropy (High = Uncertain)")

    im4 = ax4.imshow(backward_entropy, cmap="RdYlBu_r", origin="lower")
    ax4.set_title("Backward Policy Uncertainty (Entropy)")
    ax4.set_xlabel("X Position")
    ax4.set_ylabel("Y Position")
    plt.colorbar(im4, ax=ax4, label="Entropy (High = Uncertain)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


# }}}


class PolicyTrajectorySampler:
    """Sample trajectories using learned policy with encoded states"""

    def __init__(self, env: HyperGrid, model: TBModel = None, epsilon: float = 0.2):
        self.env = env
        self.model = model
        self.epsilon = epsilon

    def sample_trajectory(self, max_steps: int = 20) -> List[Tuple[float, ...]]:
        """Sample trajectory using policy with epsilon-greedy exploration"""
        trajectory = [self.env.start_state]
        state = self.env.start_state

        for _ in range(max_steps):
            if self.env.is_terminal(state):
                break

            valid_actions = self.env.get_valid_actions(state)
            if not valid_actions:
                break

            if self.model is None or random.random() < self.epsilon:
                # Random exploration
                action = random.choice(valid_actions)
            else:
                # Use learned policy
                state_tensor = torch.tensor(list(state), dtype=torch.float)
                P_F_logits, _ = self.model(state_tensor)

                # Mask invalid actions
                action_mask = torch.tensor(self.env.get_valid_action_mask(state))
                masked_logits = P_F_logits.where(
                    action_mask.bool(), torch.tensor(-100.0)
                )

                # Sample from policy
                probs = F.softmax(masked_logits, dim=0)
                action_idx = torch.multinomial(probs, 1).item()
                action = self.env.index_to_action(action_idx)

            next_state = self.env.take_action(state, action)
            trajectory.append(next_state)
            state = next_state

        return trajectory

    def sample_batch(
        self, batch_size: int, max_steps: int = 20
    ) -> List[List[Tuple[float, ...]]]:
        """Sample batch of trajectories"""
        trajectories = []
        for _ in range(batch_size):
            traj = self.sample_trajectory(max_steps)
            trajectories.append(traj)
        return trajectories


# ====================================================
# MLflow Trajectory Balance Experiment
# ====================================================
def trajectory_balance_experiment(
    lr: float = 0.001,
    hidden_dim: int = 128,
    n_steps: int = 1000,
    batch_size: int = 32,
    grid_size: int = 8,
    reward_region_size: int = 2,
    epsilon: float = 0.3,
    max_trajectory_steps: int = 20,
) -> float:
    """
    Complete trajectory balance experiment with MLflow tracking
    """

    with mlflow.start_run():
        # ====================================================
        # Log Hyperparameters
        # ====================================================
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("n_steps", n_steps)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("grid_size", grid_size)
        mlflow.log_param("reward_region_size", reward_region_size)
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("max_trajectory_steps", max_trajectory_steps)
        mlflow.log_param("method", "trajectory_balance")

        # ====================================================
        # Setup Environment and Model
        # ====================================================
        env = HyperGrid(size=grid_size, reward_region_size=reward_region_size)
        state_dim = env.get_state_dim()

        mlflow.log_param("state_dim", state_dim)

        # Create model and optimizer
        model = TBModel(state_dim, hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Create policy-based sampler
        sampler = PolicyTrajectorySampler(env, model, epsilon)

        mlflow.log_param("initial_logZ", model.logZ.item())

        print(f"🚀 Training Trajectory Balance GFlowNet")
        print(f"   Grid: {grid_size}x{grid_size}")
        print(f"   State dim: {state_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Learning rate: {lr}")
        print(f"   Initial logZ: {model.logZ.item():.4f}")

        # ====================================================
        # Training Loop
        # ====================================================
        losses = []
        log_z_values = []
        all_trajectories = []
        successful_trajectories = []

        # Progress bar
        pbar = tqdm(range(n_steps), desc=f"TB Training")

        for step in pbar:
            # Sample trajectories using current policy
            step_trajectories = sampler.sample_batch(batch_size, max_trajectory_steps)
            all_trajectories.extend(step_trajectories)

            # Collect successful trajectories
            for traj in step_trajectories:
                if env.is_terminal(traj[-1]):
                    successful_trajectories.append(traj)

            # Compute trajectory balance loss
            loss = trajectory_balance_loss(model, step_trajectories, env)

            # Update model
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)
            log_z_values.append(model.logZ.item())

            # Update progress bar
            success_count = sum(
                1 for traj in step_trajectories if env.is_terminal(traj[-1])
            )
            success_rate = success_count / len(step_trajectories)
            avg_length = sum(len(traj) for traj in step_trajectories) / len(
                step_trajectories
            )

            pbar.set_postfix(
                {
                    "Loss": f"{loss_value:.4f}",
                    "LogZ": f"{model.logZ.item():.3f}",
                    "Success": f"{success_rate:.1%}",
                    "AvgLen": f"{avg_length:.1f}",
                    "TotalSucc": len(successful_trajectories),
                }
            )

            # Log metrics periodically
            log_interval = max(1, n_steps // 50)
            if step % log_interval == 0:
                mlflow.log_metric("loss", loss_value, step=step)
                mlflow.log_metric("logZ", model.logZ.item(), step=step)
                mlflow.log_metric("success_rate", success_rate, step=step)
                mlflow.log_metric("avg_trajectory_length", avg_length, step=step)

        pbar.close()

        # ====================================================
        # Final Evaluation
        # ====================================================
        print(f"🎯 Final evaluation...")

        # Test final policy (no exploration)
        model.eval()
        test_sampler = PolicyTrajectorySampler(env, model, epsilon=0.0)
        test_trajectories = test_sampler.sample_batch(100, max_trajectory_steps)

        test_successful = sum(
            1 for traj in test_trajectories if env.is_terminal(traj[-1])
        )
        test_success_rate = test_successful / len(test_trajectories)

        # Final metrics
        final_loss = losses[-1]
        final_logZ = log_z_values[-1]
        final_Z = math.exp(final_logZ)

        # Overall training success rate
        total_successful = sum(
            1 for traj in all_trajectories if env.is_terminal(traj[-1])
        )
        overall_success_rate = total_successful / len(all_trajectories)

        # Log final metrics
        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("final_logZ", final_logZ)
        mlflow.log_metric("final_Z", final_Z)
        mlflow.log_metric("training_success_rate", overall_success_rate)
        mlflow.log_metric("test_success_rate", test_success_rate)
        mlflow.log_metric("total_trajectories_sampled", len(all_trajectories))

        # ====================================================
        # Generate Comprehensive Visualizations
        # ====================================================
        print(f"🎨 Generating comprehensive visualizations...")

        # 1. Unique successful trajectories
        unique_successful = []
        seen_paths = set()
        for traj in successful_trajectories:
            raw_path = tuple(env.decode_state_to_raw(state) for state in traj)
            if raw_path not in seen_paths:
                unique_successful.append(traj)
                seen_paths.add(raw_path)

        # 2. Generate visualizations
        if unique_successful:
            visualize_trajectory_grid(
                unique_successful[:10],
                env,
                "Unique Successful Trajectories",
                max_trajectories=10,
                save_path="successful_trajectories.png",
            )
            mlflow.log_artifact("successful_trajectories.png")

        # 3. Pattern analysis of recent trajectories
        recent_trajectories = (
            all_trajectories[-500:] if len(all_trajectories) > 500 else all_trajectories
        )
        analyze_trajectory_patterns(
            recent_trajectories, env, save_path="trajectory_patterns.png"
        )
        mlflow.log_artifact("trajectory_patterns.png")

        # 4. Policy heatmaps
        visualize_policy_heatmaps(model, env, save_path="policy_heatmaps.png")
        mlflow.log_artifact("policy_heatmaps.png")

        # 5. Recent trajectory sample (what model is currently generating)
        very_recent = (
            all_trajectories[-50:] if len(all_trajectories) > 50 else all_trajectories
        )
        visualize_trajectory_grid(
            very_recent,
            env,
            "Recent Trajectory Sample (Last 50)",
            max_trajectories=15,
            save_path="recent_trajectories.png",
        )
        mlflow.log_artifact("recent_trajectories.png")

        # 6. Basic training metrics plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curve
        ax1.plot(losses)
        ax1.set_title("Trajectory Balance Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # LogZ evolution
        ax2.plot(log_z_values)
        ax2.set_title("Log Partition Function Evolution")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Log Z")
        ax2.grid(True)

        # Success rate over time
        window_size = max(1, len(all_trajectories) // 20)
        success_rates = []
        for i in range(0, len(all_trajectories), window_size):
            window_trajs = all_trajectories[i : i + window_size]
            window_success = sum(
                1 for traj in window_trajs if env.is_terminal(traj[-1])
            ) / len(window_trajs)
            success_rates.append(window_success)

        ax3.plot(success_rates)
        ax3.set_title("Success Rate Over Training")
        ax3.set_xlabel("Window")
        ax3.set_ylabel("Success Rate")
        ax3.grid(True)

        # Trajectory length distribution
        lengths = [len(traj) for traj in all_trajectories[-200:]]
        ax4.hist(lengths, bins=15, alpha=0.7)
        ax4.set_title("Trajectory Length Distribution (Recent)")
        ax4.set_xlabel("Length")
        ax4.set_ylabel("Count")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig("tb_training_summary.png", dpi=150, bbox_inches="tight")
        mlflow.log_artifact("tb_training_summary.png")
        plt.close()

        # Log additional metrics
        mlflow.log_metric("unique_successful_paths", len(unique_successful))
        recent_success_rate = (
            sum(1 for traj in very_recent if env.is_terminal(traj[-1]))
            / len(very_recent)
            if very_recent
            else 0
        )
        mlflow.log_metric("recent_success_rate", recent_success_rate)

        # ====================================================
        # Log Results Summary
        # ====================================================
        if successful_trajectories:
            unique_successful = []
            seen_paths = set()
            for traj in successful_trajectories:
                # Convert to raw coordinates for deduplication
                raw_path = tuple(env.decode_state_to_raw(state) for state in traj)
                if raw_path not in seen_paths:
                    unique_successful.append(traj)
                    seen_paths.add(raw_path)

            mlflow.log_metric("unique_successful_trajectories", len(unique_successful))

            # Save analysis
            with open("tb_results.txt", "w") as f:
                f.write("Trajectory Balance GFlowNet Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Hyperparameters:\n")
                f.write(f"  Learning Rate: {lr}\n")
                f.write(f"  Hidden Dim: {hidden_dim}\n")
                f.write(f"  Grid Size: {grid_size}x{grid_size}\n")
                f.write(f"  Training Steps: {n_steps}\n")
                f.write(f"  Batch Size: {batch_size}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Final Loss: {final_loss:.6f}\n")
                f.write(f"  Final LogZ: {final_logZ:.4f} (Z = {final_Z:.4f})\n")
                f.write(f"  Training Success Rate: {overall_success_rate:.2%}\n")
                f.write(f"  Test Success Rate: {test_success_rate:.2%}\n")
                f.write(f"  Total Trajectories: {len(all_trajectories)}\n")
                f.write(f"  Successful Trajectories: {len(successful_trajectories)}\n")
                f.write(f"  Unique Successful: {len(unique_successful)}\n\n")

                f.write("Sample successful paths (raw coordinates):\n")
                for i, traj in enumerate(unique_successful[:10]):
                    raw_path = [env.decode_state_to_raw(state) for state in traj]
                    f.write(f"  {i + 1}: {raw_path} (length: {len(traj)})\n")

            mlflow.log_artifact("tb_results.txt")

            print(
                f"✅ Found {len(successful_trajectories)} successful trajectories ({len(unique_successful)} unique)"
            )
        else:
            print("❌ No successful trajectories found!")

        # Save model weights
        torch.save(model.state_dict(), "tb_model_weights.pth")
        mlflow.log_artifact("tb_model_weights.pth")

        print(f"\n📊 Final Results:")
        print(f"   Loss: {final_loss:.6f}")
        print(f"   LogZ: {final_logZ:.4f} (Z = {final_Z:.4f})")
        print(f"   Training Success: {overall_success_rate:.2%}")
        print(f"   Test Success: {test_success_rate:.2%}")
        print(f"   Total Trajectories: {len(all_trajectories)}")

        return final_loss


# ====================================================
# Hyperparameter Sweep
# ====================================================
def run_trajectory_balance_sweep():
    """Run systematic hyperparameter sweep"""

    # Hyperparameter grid
    learning_rates = [0.01, 0.001, 0.0001]
    hidden_dims = [64, 128, 256]
    grid_sizes = [6, 8]

    results = []

    for lr in learning_rates:
        for hidden_dim in hidden_dims:
            for grid_size in grid_sizes:
                print(f"\n{'=' * 60}")
                print(
                    f"🧪 Running: lr={lr}, hidden_dim={hidden_dim}, grid_size={grid_size}"
                )
                print(f"{'=' * 60}")

                final_loss = trajectory_balance_experiment(
                    lr=lr,
                    hidden_dim=hidden_dim,
                    grid_size=grid_size,
                    n_steps=500,  # Shorter for sweep
                    batch_size=32,
                )

                results.append(
                    {
                        "lr": lr,
                        "hidden_dim": hidden_dim,
                        "grid_size": grid_size,
                        "final_loss": final_loss,
                    }
                )

    # Print sweep summary
    print(f"\n{'=' * 80}")
    print("🏆 TRAJECTORY BALANCE HYPERPARAMETER SWEEP RESULTS")
    print(f"{'=' * 80}")

    # Sort by final loss
    results.sort(key=lambda x: x["final_loss"])

    for i, result in enumerate(results):
        print(
            f"{i + 1:2d}. lr={result['lr']:6.4f}, hidden_dim={result['hidden_dim']:3d}, "
            f"grid_size={result['grid_size']:2d} -> loss={result['final_loss']:8.6f}"
        )

    # Best configuration
    best = results[0]
    print(f"\n🥇 Best Configuration:")
    print(f"   Learning Rate: {best['lr']}")
    print(f"   Hidden Dim: {best['hidden_dim']}")
    print(f"   Grid Size: {best['grid_size']}")
    print(f"   Final Loss: {best['final_loss']:.6f}")


# ====================================================
# Main Execution
# ====================================================
if __name__ == "__main__":
    # MLflow setup
    try:
        mlflow.end_run()
    except:
        pass

    mlflow.set_experiment("Trajectory_Balance_GFlowNet")

    # Run single experiment
    trajectory_balance_experiment(lr=0.001, hidden_dim=128, n_steps=1000)

    # Or run hyperparameter sweep
    # run_trajectory_balance_sweep()

    # Clean up
    try:
        mlflow.end_run()
    except:
        pass
