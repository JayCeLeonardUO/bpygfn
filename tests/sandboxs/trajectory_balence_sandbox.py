import torch
import torch.nn as nn
import torch.optim as optim
import random
from gfn_environments.single_color_ramp import *

# ============================================================================
# Model Definition
# ============================================================================

class TerrainGFlowNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Forward head: all actions including "done"
        self.forward_head = nn.Linear(hidden_dim, action_dim)

        # Backward head: only color removal actions (no "done" in backward)
        self.backward_head = nn.Linear(hidden_dim, action_dim - 1)

        self.logZ = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        features = self.backbone(state)
        P_F_logits = self.forward_head(features)
        P_B_logits = self.backward_head(features)

        # Extract colors used and done flag
        num_colors = self.action_dim - 1  # 34 colors
        colors_used = state[:num_colors]  # First 34 elements
        is_done = state[-1]  # Last element

        # Split forward logits into colors and done
        color_logits = P_F_logits[:num_colors]  # First 34 logits
        done_logit = P_F_logits[-1]  # Last logit

        # Mask forward colors: can't use colors already selected
        masked_colors = color_logits * (1 - colors_used) + colors_used * -100

        # Mask done action: can't do if already done
        masked_done = done_logit * (1 - is_done) + is_done * -100

        # Combine into full forward policy
        P_F = torch.cat([masked_colors, masked_done.unsqueeze(0)])

        # Mask backward: can only remove colors that were added
        P_B = P_B_logits * colors_used + (1 - colors_used) * -100

        return P_F, P_B


# ============================================================================
# Random Trajectory Sampler
# ============================================================================

def sample_random_trajectory(blender_api, s_wstep, action_dim, target_length=10):
    """
    Sample a trajectory with random actions, aiming for target_length colors
    """
    blender_api.reset_env()
    state = blender_api.blender_env_to_tensor()

    trajectory_data = []
    num_colors_added = 0

    while state[-1] == 0:  # While not done
        # Get list of valid color actions (not yet used)
        valid_color_actions = []
        for i in range(action_dim - 1):  # All color actions
            if state[i] == 0:  # Not used yet
                valid_color_actions.append(i)

        # Decide whether to continue or stop
        if num_colors_added >= target_length or len(valid_color_actions) == 0:
            # Do "done" action
            action = action_dim - 1
        else:
            # Pick random color from valid actions
            action = random.choice(valid_color_actions)
            num_colors_added += 1

        # Execute action
        s_wstep.execute_idx(blender_api, action)
        next_state = blender_api.blender_env_to_tensor()

        # Store transition
        trajectory_data.append({
            'state': state.clone(),
            'action': action,
            'next_state': next_state.clone(),
        })

        assert isinstance(next_state, object)
        state = next_state

    # Get final heightmap
# ============================================================================
# Generate Random Dataset
# ============================================================================

def generate_random_dataset(blender_api, s_wstep, action_dim,
                            num_trajectories=100, target_length=10):
    """
    Generate a dataset of random trajectories
    """
    dataset = []

    print(f"Generating {num_trajectories} random trajectories...")

    for i in range(num_trajectories):
        trajectory_data, terminal_state, heightmap = sample_random_trajectory(
            blender_api, s_wstep, action_dim, target_length
        )

        dataset.append({
            'trajectory': trajectory_data,
            'terminal_state': terminal_state,
            'heightmap': heightmap
        })

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_trajectories} trajectories")

    print(f"✓ Generated {len(dataset)} trajectories")
    print(f"  Avg trajectory length: {sum(len(d['trajectory']) for d in dataset) / len(dataset):.1f}")

    return dataset


# ============================================================================
# Compute Reward
# ============================================================================

def compute_reward(terminal_state, heightmap, action_dim,
                   target_min_height=0.2, target_max_height=0.8):
    """
    Compute reward based on heightmap and diversity
    """
    # Count how many height values are in target range
    in_range = ((heightmap >= target_min_height) &
                (heightmap <= target_max_height)).float()

    proportion_in_range = in_range.mean()

    # Number of colors used
    num_colors_used = terminal_state[:action_dim - 1].sum()

    # Base reward from height range
    height_reward = proportion_in_range

    # Bonus for using more colors (encourage diversity)
    diversity_bonus = (num_colors_used / (action_dim - 1)) * 0.5

    # Combined reward
    combined_reward = height_reward + diversity_bonus

    # Convert to exponential reward
    reward = torch.exp(combined_reward * 3.0)

    return reward, proportion_in_range, num_colors_used


# ============================================================================
# Train on Pre-collected Dataset
# ============================================================================

def train_on_dataset(model, dataset, action_dim, num_epochs=10, lr=1e-3):
    """
    Train on a pre-collected dataset of trajectories
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        random.shuffle(dataset)
        epoch_loss = 0

        for data in dataset:
            trajectory_data = data['trajectory']
            terminal_state = data['terminal_state']
            heightmap = data['heightmap']

            # -------------------------
            # Compute forward and backward probs with current model
            # -------------------------
            trajectory = []
            for step in trajectory_data:
                state = step['state']
                action = step['action']
                next_state = step['next_state']

                # Get forward prob
                P_F, _ = model(state)
                log_pf = P_F[action]

                # Get backward prob (if applicable)
                log_pb = None
                if len(trajectory) > 0 and action < action_dim - 1:
                    _, P_B = model(next_state)
                    log_pb = P_B[action]

                trajectory.append({
                    'log_pf': log_pf,
                    'log_pb': log_pb,
                    'action': action
                })

            # -------------------------
            # Compute reward
            # -------------------------
            reward, proportion_in_range, num_colors = compute_reward(
                terminal_state, heightmap, action_dim
            )
            log_reward = torch.log(reward + 1e-8)

            # -------------------------
            # Compute loss
            # -------------------------
            log_pf_sum = sum(step['log_pf'] for step in trajectory)

            log_pb_sum = sum(
                step['log_pb'] for step in trajectory
                if step['log_pb'] is not None
            )

            # Detailed balance loss
            log_ratio = model.logZ + log_pf_sum - log_reward - log_pb_sum
            loss = log_ratio ** 2

            # -------------------------
            # Update
            # -------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, LogZ: {model.logZ.item():.4f}")

    return model


# ============================================================================
# Initialize Everything
# ============================================================================

blender_api = BlenderTerrainAPI()
s_wstep = v2StepWEnv()

# Get actual dimensions from environment
blender_api.reset_env()
sample_state = blender_api.blender_env_to_tensor()
state_dim = sample_state.shape[0]
action_dim = s_wstep.n_actions

print(f"State dim: {state_dim}, Action dim: {action_dim}")

model = TerrainGFlowNet(state_dim, action_dim, hidden_dim=128)

target_min_height = 0.2
target_max_height = 0.8

# ============================================================================
# PHASE 1: Generate Random Dataset
# ============================================================================

print("\n" + "=" * 50)
print("PHASE 1: Generating random trajectories")
print("=" * 50)

random_dataset = generate_random_dataset(
    blender_api,
    s_wstep,
    action_dim,
    num_trajectories=200,
    target_length=10
)

# ============================================================================
# PHASE 2: Train on Random Dataset
# ============================================================================

print("\n" + "=" * 50)
print("PHASE 2: Training on random dataset")
print("=" * 50)

model = train_on_dataset(
    model,
    random_dataset,
    action_dim,
    num_epochs=20,
    lr=1e-3
)

# ============================================================================
# PHASE 3: Continue with On-Policy Training
# ============================================================================

print("\n" + "=" * 50)
print("PHASE 3: On-policy fine-tuning")
print("=" * 50)

optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate

num_iterations = 1000

for iteration in range(num_iterations):

    # Sample trajectory using current model
    blender_api.reset_env()
    state = blender_api.blender_env_to_tensor()

    trajectory = []

    while state[-1] == 0:
        P_F, _ = model(state)
        action_probs = torch.softmax(P_F, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        log_pf = P_F[action]

        s_wstep.execute_idx(blender_api, action)
        next_state = blender_api.blender_env_to_tensor()

        trajectory.append({
            'state': state.clone(),
            'action': action,
            'next_state': next_state.clone(),
            'log_pf': log_pf,
            'log_pb': None
        })

        state = next_state

    terminal_state = state
    final_heightmap = blender_api.get_heightmap()

    # Fill backward probabilities
    for i in range(len(trajectory)):
        if i > 0:
            next_state = trajectory[i]['next_state']
            action = trajectory[i]['action']

            _, P_B = model(next_state)

            if action < action_dim - 1:
                trajectory[i]['log_pb'] = P_B[action]
            else:
                trajectory[i]['log_pb'] = torch.tensor(0.0)

    # Compute reward
    reward, proportion_in_range, num_colors = compute_reward(
        terminal_state, final_heightmap, action_dim,
        target_min_height, target_max_height
    )
    log_reward = torch.log(reward + 1e-8)

    # Compute loss
    log_pf_sum = sum(step['log_pf'] for step in trajectory)
    log_pb_sum = sum(
        step['log_pb'] for step in trajectory
        if step['log_pb'] is not None and step['action'] < action_dim - 1
    )

    log_ratio = model.logZ + log_pf_sum - log_reward - log_pb_sum
    loss = log_ratio ** 2

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if iteration % 100 == 0:
        print(f"Iter {iteration}, Loss: {loss.item():.4f}, "
              f"LogZ: {model.logZ.item():.4f}, "
              f"Traj Len: {len(trajectory)}, "
              f"Colors: {int(num_colors.item())}, "
              f"Reward: {reward.item():.4f}, "
              f"In Range: {proportion_in_range.item():.2%}")

# ============================================================================
# After Training: Sample from Trained Model
# ============================================================================

print("\n" + "=" * 50)
print("Sampling from trained model:")

blender_api.reset_env()
state = blender_api.blender_env_to_tensor()

sampled_actions = []

while state[-1] == 0:
    P_F, _ = model(state)
    action_probs = torch.softmax(P_F, dim=-1)
    action = torch.multinomial(action_probs, 1).item()

    sampled_actions.append(action)
    s_wstep.execute_idx(blender_api, action)
    state = blender_api.blender_env_to_tensor()

final_heightmap = blender_api.get_heightmap()

print(f"Sampled actions: {sampled_actions}")
print(f"Colors used: {int(state[:action_dim - 1].sum().item())}")
print(f"Heightmap stats:")
print(f"  Min: {final_heightmap.min().item():.3f}")
print(f"  Max: {final_heightmap.max().item():.3f}")
print(f"  Mean: {final_heightmap.mean().item():.3f}")
print(
    f"  In target range: {((final_heightmap >= target_min_height) & (final_heightmap <= target_max_height)).float().mean().item():.2%}")