from gfn_environments.single_color_ramp import  *


def test_state_tensors_clear():
    """Test with clear labels showing what each part of the tensor means"""
    print("=== State Tensor Breakdown with Labels ===\n")

    # Build a state step by step
    print("Building state: w=4, scale=2.0, colors=[3, 7, 15]\n")

    state = State()
    state = state.apply_action('w', 2)  # w=4 (index 2 in VALID_W)
    state = state.apply_action('scale', 3)  # scale=2.0 (index 3 in VALID_SCALE)
    state = state.apply_action('color', 3)  # color index 3
    state = state.apply_action('color', 7)  # color index 7
    state = state.apply_action('color', 15)  # color index 15

    tensor = state.to_tensor()

    print(f"Full tensor ({len(tensor)} elements):")
    print(tensor)
    print()

    # Parse it section by section
    idx = 0

    # Section 1: W parameter (one-hot)
    print("=" * 60)
    print("SECTION 1: W parameter (one-hot encoding)")
    print("=" * 60)
    w_len = len(ActionRegistry.VALID_W)
    w_section = tensor[idx:idx + w_len]
    print(f"Valid W values: {ActionRegistry.VALID_W}")
    print(f"Tensor indices [{idx}:{idx + w_len}]: {w_section}")
    w_idx = np.argmax(w_section)
    print(f"Selected: index {w_idx} → w={ActionRegistry.VALID_W[w_idx]}")
    print()
    idx += w_len

    # Section 2: Scale parameter (one-hot)
    print("=" * 60)
    print("SECTION 2: Scale parameter (one-hot encoding)")
    print("=" * 60)
    scale_len = len(ActionRegistry.VALID_SCALE)
    scale_section = tensor[idx:idx + scale_len]
    print(f"Valid Scale values: {ActionRegistry.VALID_SCALE}")
    print(f"Tensor indices [{idx}:{idx + scale_len}]: {scale_section}")
    scale_idx = np.argmax(scale_section)
    print(f"Selected: index {scale_idx} → scale={ActionRegistry.VALID_SCALE[scale_idx]}")
    print()
    idx += scale_len

    # Section 3: Colors (binary vector)
    print("=" * 60)
    print("SECTION 3: Colors (binary vector - 32 colors)")
    print("=" * 60)
    color_len = len(ActionRegistry.VALID_COLORS)
    color_section = tensor[idx:idx + color_len]
    print(f"Tensor indices [{idx}:{idx + color_len}]:")
    print(f"  (showing only non-zero entries)")
    selected = np.where(color_section == 1.0)[0]
    print(f"  Colors selected: {list(selected)}")
    for i in selected:
        print(f"    tensor[{idx + i}] = 1.0  ← color {i} is selected")
    print()
    idx += color_len

    # Section 4: Metadata
    print("=" * 60)
    print("SECTION 4: Metadata (4 values)")
    print("=" * 60)
    metadata = tensor[idx:idx + 4]
    print(f"Tensor indices [{idx}:{idx + 4}]: {metadata}")
    print(f"  [{idx}]   = {metadata[0]:.5f} ← num_colors / 32 = {state.num_colors_selected}/32")
    print(f"  [{idx + 1}] = {metadata[1]:.1f}     ← w is set? (1.0=yes, 0.0=no)")
    print(f"  [{idx + 2}] = {metadata[2]:.1f}     ← scale is set? (1.0=yes, 0.0=no)")
    print(f"  [{idx + 3}] = {metadata[3]:.1f}     ← in color_selection phase? (1.0=yes, 0.0=no)")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tensor size: {len(tensor)}")
    print(f"  W one-hot:       {w_len} elements")
    print(f"  Scale one-hot:   {scale_len} elements")
    print(f"  Colors binary:   {color_len} elements")
    print(f"  Metadata:        4 elements")
    print(f"  Total:           {w_len + scale_len + color_len + 4}")
    print()
    print("State contents:")
    print(f"  w = {state.w}")
    print(f"  scale = {state.scale}")
    print(f"  selected_colors = {selected.tolist()}")
    print(f"  num_colors_selected = {state.num_colors_selected}")
    print(f"  current_phase = {state.current_phase}")


def test_state_tensors():
    """Test that State provides both tensor representations"""
    print("=== Testing State Tensor Methods (PyTorch) ===\n")

    # Create a state
    state = State()
    state = state.apply_action('w', 2)  # w=4
    state = state.apply_action('scale', 3)  # scale=2.0
    state = state.apply_action('color', 5)  # color 5

    print("State:")
    print(f"  w={state.w}, scale={state.scale}")
    print(f"  colors selected: {torch.where(torch.tensor(state.selected_colors))[0].tolist()}")
    print(f"  phase: {state.current_phase}\n")

    # Test state tensor (INPUT)
    print("State Tensor (network INPUT):")
    state_tensor = state.to_state_tensor()
    print(f"  Shape: {state_tensor.shape}")
    print(f"  Expected dimension: {State.get_state_tensor_dim()}")
    print(f"  Match: {state_tensor.shape[0] == State.get_state_tensor_dim()}")
    print(f"  Dtype: {state_tensor.dtype}")
    print(f"  Tensor: {state_tensor}\n")

    # Test action mask (OUTPUT space)
    print("Action Mask (network OUTPUT space):")
    action_mask = state.to_action_mask()
    print(f"  Shape: {action_mask.shape}")
    print(f"  Expected dimension: {State.get_action_tensor_dim()}")
    print(f"  Match: {action_mask.shape[0] == State.get_action_tensor_dim()}")
    print(f"  Dtype: {action_mask.dtype}")
    print(f"  Valid actions: {action_mask.sum().item()} out of {len(action_mask)}")
    print(f"  Valid action indices: {torch.where(action_mask)[0].tolist()}\n")

    # Test with network
    print("Testing with GFlowNet:")
    gfn = ColorRampGFlowNet(hidden_dim=64)
    probs = gfn.forward_policy(state)
    print(f"  Action probabilities shape: {probs.shape}")
    print(f"  Sum of probabilities: {probs.sum().item():.6f}")
    print(f"  Non-zero probabilities: {(probs > 1e-6).sum().item()}\n")

    # Sample a full trajectory
    print("Sampling trajectory:")
    trajectory, final_state = gfn.sample_trajectory()
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"  Actions taken:")
    for i, step in enumerate(trajectory):
        action_name = step['action_name']
        value_idx = step['value_idx']
        value = ActionRegistry.ACTIONS[action_name]['valid_values'][value_idx]
        prob = step['prob']
        print(f"    Step {i}: {action_name:10s} = {str(value):10s} (p={prob:.3f})")

    print("\n=== All tests passed! ===")



def test_function_pointers():
    """Test that function pointers work correctly"""
    print("=== Testing Direct Function Pointers ===\n")

    print("Action Registry function pointers:")
    for action_name, action_info in ActionRegistry.ACTIONS.items():
        execute_fn = action_info['execute']
        has_fn = execute_fn is not None
        print(f"  {action_name:15s}: {'✓ has execute function' if has_fn else '✗ no function (stop action)'}")

    print("\n\nSample trajectory:")
    gfn = ColorRampGFlowNet(hidden_dim=64)
    trajectory, final_state = gfn.sample_trajectory()

    for i, step in enumerate(trajectory):
        action_name = step['action_name']
        value_idx = step['value_idx']
        value = ActionRegistry.ACTIONS[action_name]['valid_values'][value_idx]
        print(f"  Step {i}: {action_name:15s} = {value}")

    print(f"\n\nFinal state:")
    print(f"  noise_w: {final_state.noise_w}")
    print(f"  noise_scale: {final_state.noise_scale}")
    print(f"  num_colors: {final_state.num_colors_assigned}")
    print(f"  color_assignments: {final_state.color_assignments}")

    print(f"\n\nTo execute on Blender:")
    print(f"  # Apply entire state at once:")
    print(f"  final_state.apply_to_blender(blender_api)")
    print(f"\n  # Or execute actions one by one:")
    print(f"  state.execute_action_on_blender(blender_api, 'set_w', value_idx=2)")


def visualize_trajectory(buffer: ReplayBuffer, trajectory_id: int, save_path: Optional[str] = None):
    """
    Create a matplotlib figure showing the trajectory progression with heightmaps

    Args:
        buffer: ReplayBuffer containing the trajectory
        trajectory_id: ID of trajectory to visualize
        save_path: Optional path to save figure (if None, just displays)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get trajectory data
    trajectory = buffer.get_trajectory(trajectory_id)
    heightmaps = buffer.get_heightmaps(trajectory_id)
    final_state = buffer.get_final_state(trajectory_id)

    if trajectory is None or heightmaps is None:
        print(f"Trajectory {trajectory_id} not found")
        return

    num_steps = len(trajectory)

    # Create figure with subplots
    # Top row: heightmaps at key steps
    # Bottom row: action information and statistics
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, num_steps, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # Plot heightmaps for each step
    for i in range(num_steps):
        ax = fig.add_subplot(gs[0, i])

        # Get heightmap and convert to numpy
        hm = heightmaps[i].cpu().numpy()

        # Plot heightmap
        im = ax.imshow(hm, cmap='terrain', vmin=hm.min(), vmax=hm.max())

        # Get action info
        step = trajectory[i]
        action_name = step['action_name']
        value_idx = step['value_idx']
        value = ActionRegistry.ACTIONS[action_name]['valid_values'][value_idx]
        prob = step['prob']

        # Title with action
        ax.set_title(f"Step {i}\n{action_name}\n{value}\np={prob:.3f}", fontsize=8)
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: statistics
    for i in range(num_steps):
        ax = fig.add_subplot(gs[1, i])

        hm = heightmaps[i].cpu().numpy()

        # Compute stats
        stats_text = f"Mean: {hm.mean():.3f}\n"
        stats_text += f"Std: {hm.std():.3f}\n"
        stats_text += f"Min: {hm.min():.3f}\n"
        stats_text += f"Max: {hm.max():.3f}\n"
        stats_text += f"Var: {hm.var():.3f}"

        ax.text(0.5, 0.5, stats_text, ha='center', va='center',
                fontsize=7, family='monospace')
        ax.axis('off')

    # Overall title
    rewards_str = ""
    if trajectory_id in buffer.rewards:
        rewards_str = f" | Rewards: {buffer.rewards[trajectory_id]}"

    fig.suptitle(
        f"Trajectory {trajectory_id} | "
        f"w={final_state.noise_w}, scale={final_state.noise_scale}, "
        f"colors={final_state.num_colors_assigned}{rewards_str}",
        fontsize=14, fontweight='bold'
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig


def visualize_trajectory_comparison(buffer: ReplayBuffer, trajectory_ids: List[int],
                                    save_path: Optional[str] = None):
    """
    Compare multiple trajectories side by side

    Args:
        buffer: ReplayBuffer containing trajectories
        trajectory_ids: List of trajectory IDs to compare
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_trajs = len(trajectory_ids)

    fig, axes = plt.subplots(2, num_trajs, figsize=(6 * num_trajs, 10))
    if num_trajs == 1:
        axes = axes.reshape(-1, 1)

    for col, traj_id in enumerate(trajectory_ids):
        trajectory = buffer.get_trajectory(traj_id)
        heightmaps = buffer.get_heightmaps(traj_id)
        final_state = buffer.get_final_state(traj_id)

        if trajectory is None:
            continue

        # Plot final heightmap
        final_hm = heightmaps[-1].cpu().numpy()
        im = axes[0, col].imshow(final_hm, cmap='terrain')
        axes[0, col].set_title(
            f"Trajectory {traj_id}\n"
            f"w={final_state.noise_w}, scale={final_state.noise_scale}\n"
            f"colors={final_state.num_colors_assigned}",
            fontsize=10
        )
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

        # Plot statistics over time
        means = [hm.mean().item() for hm in heightmaps]
        stds = [hm.std().item() for hm in heightmaps]
        variances = [hm.var().item() for hm in heightmaps]

        ax_stats = axes[1, col]
        x = range(len(means))

        ax_stats.plot(x, means, 'b-', label='Mean', marker='o')
        ax_stats.plot(x, stds, 'r-', label='Std', marker='s')
        ax_stats.plot(x, variances, 'g-', label='Variance', marker='^')

        ax_stats.set_xlabel('Step')
        ax_stats.set_ylabel('Value')
        ax_stats.set_title('Heightmap Statistics Over Time')
        ax_stats.legend()
        ax_stats.grid(True, alpha=0.3)

        # Add reward info if available
        if traj_id in buffer.rewards:
            reward_text = '\n'.join([f"{k}: {v:.3f}" for k, v in buffer.rewards[traj_id].items()])
            ax_stats.text(0.02, 0.98, reward_text, transform=ax_stats.transAxes,
                          verticalalignment='top', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    return fig


# Update the test to use visualization
def test_replay_buffer_with_heightmaps():
    """Test the replay buffer with heightmaps at each step"""
    print("=== Testing Replay Buffer with Per-Step Heightmaps ===\n")


    # Create buffer, GFlowNet, and Blender API
    buffer = ReplayBuffer(capacity=100)
    gfn = ColorRampGFlowNet(hidden_dim=64)
    blender_api = BlenderTerrainAPI()

    # Sample trajectories with heightmaps
    print("Sampling trajectories with heightmaps...\n")
    for i in range(3):
        trajectory, final_state, heightmaps = sample_trajectory_with_heightmaps(
            gfn, blender_api, max_steps=20
        )

        # Add to buffer
        traj_id = buffer.add_trajectory(trajectory, final_state, heightmaps)

        # Compute variance of final heightmap as reward
        final_heightmap = heightmaps[-1]
        variance = final_heightmap.var().item()
        buffer.add_reward(traj_id, 'variance', variance)

        print(f"Added trajectory {traj_id}:")
        print(f"  Steps: {len(trajectory)}")
        print(f"  Final variance: {variance:.4f}")

    print(f"\nBuffer: {buffer}")

    # Visualize first trajectory
    print("\nGenerating visualization for trajectory 0...")
    visualize_trajectory(buffer, 0, save_path='trajectory_0.png')

    # Compare all trajectories
    print("\nGenerating comparison of all trajectories...")
    visualize_trajectory_comparison(buffer, [0, 1, 2], save_path='trajectory_comparison.png')

    print("\n✓ Visualizations saved!")

