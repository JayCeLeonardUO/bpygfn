from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
import torch


@dataclass
class ColorRampState:
    """Represents a state in the color ramp construction process"""

    scale: Optional[float] = None
    colors: Dict[int, int] = None  # position_index -> color_id
    step_count: int = 0

    def __post_init__(self):
        if self.colors is None:
            self.colors = {}

    def __hash__(self):
        if self.scale is None:
            scale_tuple = (None,)
        else:
            scale_tuple = (self.scale,)
        colors_tuple = tuple(sorted(self.colors.items()))
        return hash((scale_tuple, colors_tuple, self.step_count))

    def __eq__(self, other):
        if not isinstance(other, ColorRampState):
            return False
        return (
            self.scale == other.scale
            and self.colors == other.colors
            and self.step_count == other.step_count
        )

    def copy(self):
        return ColorRampState(
            scale=self.scale, colors=self.colors.copy(), step_count=self.step_count
        )

    def is_terminal(self, max_colors: int) -> bool:
        return self.scale is not None and len(self.colors) >= max_colors


class BlenderColorRampEnvironment:
    """
    Blender environment for color ramp generation - composition based.
    Similar to your HyperGrid approach.
    """

    def __init__(
        self,
        max_colors: int = 5,
        num_color_choices: int = 32,
        available_scales: List[float] = None,
    ):
        self.max_colors = max_colors
        self.num_color_choices = num_color_choices

        if available_scales is None:
            self.available_scales = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
        else:
            self.available_scales = available_scales

        # Blender connections (set externally)
        self.plane = None
        self.modifier = None
        self.created_nodes = None

        print(f"🎨 Blender ColorRamp Environment created")
        print(f"  Max colors: {max_colors}")
        print(f"  Color choices: {num_color_choices}")
        print(f"  Available scales: {len(self.available_scales)}")

    def get_initial_state(self) -> ColorRampState:
        """
        Get the initial empty state.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=3)
            >>> state = env.get_initial_state()
            >>> print(state.scale)     # None
            >>> print(state.colors)    # {}
        """
        return ColorRampState()

    def get_valid_actions(self, state: ColorRampState) -> List[int]:
        """
        Get all valid actions from current state.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=2, num_color_choices=3)
            >>> state = env.get_initial_state()
            >>> actions = env.get_valid_actions(state)
            >>> print(actions)  # [0, 1, 2, 3, 4, 5, 6] - scale choices
        """
        if state.scale is None:
            return list(range(len(self.available_scales)))

        occupied_positions = set(state.colors.keys())
        available_positions = [
            i for i in range(self.max_colors) if i not in occupied_positions
        ]

        if not available_positions:
            return []

        valid_actions = []
        scale_offset = len(self.available_scales)

        for pos in available_positions:
            for color_id in range(self.num_color_choices):
                action = scale_offset + pos * self.num_color_choices + color_id
                valid_actions.append(action)

        return valid_actions

    def apply_action(self, state: ColorRampState, action: int) -> ColorRampState:
        """
        Apply action to state to get next state.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=2)
            >>> state = env.get_initial_state()
            >>> next_state = env.apply_action(state, action=2)  # Choose scale
            >>> print(next_state.scale)  # 2.0 (or whatever scale index 2 is)
        """
        if action not in self.get_valid_actions(state):
            raise ValueError(f"Invalid action {action} for state {state}")

        next_state = state.copy()
        next_state.step_count += 1

        if state.scale is None:
            if 0 <= action < len(self.available_scales):
                next_state.scale = self.available_scales[action]
            else:
                raise ValueError(f"Invalid scale action {action}")
        else:
            scale_offset = len(self.available_scales)
            color_action = action - scale_offset

            position_idx = color_action // self.num_color_choices
            color_id = color_action % self.num_color_choices

            if position_idx in state.colors:
                raise ValueError(f"Position {position_idx} already occupied")

            next_state.colors[position_idx] = color_id

        return next_state

    def is_terminal(self, state: ColorRampState) -> bool:
        """
        Check if state is terminal.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=2)
            >>> state = ColorRampState(scale=5.0, colors={0: 1, 1: 2})
            >>> print(env.is_terminal(state))  # True
        """
        return state.is_terminal(self.max_colors)

    def get_reward(self, state: ColorRampState) -> float:
        """
        Get reward for terminal state.

        Example:
            >>> env = BlenderColorRampEnvironment()
            >>> env.connect_blender(plane, modifier, nodes)
            >>> terminal_state = ColorRampState(scale=5.0, colors={0: 1, 1: 2})
            >>> reward = env.get_reward(terminal_state)
            >>> print(reward)  # 1.0 if no holes, 0.0 if holes
        """
        if not self.is_terminal(state):
            return 0.0

        try:
            # ====================================================
            # Apply state to Blender
            # ====================================================
            if not self.created_nodes:
                return 0.0

            # Apply noise scale
            if state.scale is not None and "noise" in self.created_nodes:
                noise_node = self.created_nodes["noise"]
                noise_node.inputs["Scale"].default_value = state.scale

            # Apply color ramp
            if "ramp" in self.created_nodes:
                ramp_node = self.created_nodes["ramp"]
                color_ramp = ramp_node.color_ramp

                # Clear existing points
                while len(color_ramp.elements) > 0:
                    color_ramp.elements.remove(color_ramp.elements[0])

                # Add color points
                if not state.colors:
                    # Default black to white
                    elem0 = color_ramp.elements.new(0.0)
                    elem0.color = (0, 0, 0, 1)
                    elem1 = color_ramp.elements.new(1.0)
                    elem1.color = (1, 1, 1, 1)
                else:
                    # Add colors at their positions
                    for position_idx in sorted(state.colors.keys()):
                        color_id = state.colors[position_idx]
                        normalized_pos = (position_idx + 1) / self.max_colors

                        # Convert color ID to RGBA
                        if color_id == 0:
                            rgba = (0.0, 0.0, 0.0, 1.0)  # Black
                        elif color_id == 1:
                            rgba = (1.0, 1.0, 1.0, 1.0)  # White
                        elif color_id < 8:
                            colors = [
                                (1.0, 0.0, 0.0, 1.0),
                                (0.0, 1.0, 0.0, 1.0),
                                (0.0, 0.0, 1.0, 1.0),
                                (1.0, 1.0, 0.0, 1.0),
                                (1.0, 0.0, 1.0, 1.0),
                                (0.0, 1.0, 1.0, 1.0),
                            ]
                            rgba = colors[color_id - 2]
                        else:
                            import colorsys

                            hue = ((color_id - 8) / (32 - 8)) * 360
                            saturation = 0.8 if color_id % 2 == 0 else 1.0
                            value = 0.9 if color_id % 3 == 0 else 0.7
                            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
                            rgba = (r, g, b, 1.0)

                        elem = color_ramp.elements.new(normalized_pos)
                        elem.color = rgba

            # Force Blender update
            bpy.context.view_layer.update()

            # ====================================================
            # Extract noise tensor from Blender
            # ====================================================
            if not self.plane:
                raise ValueError("Blender not connected")

            depsgraph = bpy.context.evaluated_depsgraph_get()
            plane_eval = self.plane.evaluated_get(depsgraph)
            mesh = plane_eval.to_mesh()

            # Extract vertex heights
            verts = np.array([(v.co.x, v.co.y, v.co.z) for v in mesh.vertices])
            grid_size = int(np.sqrt(len(verts)))
            heights = verts[:, 2].reshape(grid_size, grid_size)
            tensor = torch.from_numpy(heights).float()

            # Clean up Blender mesh
            plane_eval.to_mesh_clear()

            # ====================================================
            # Hole detection
            # ====================================================
            threshold = 0.5

            # Convert to numpy if needed
            if torch.is_tensor(tensor):
                noise = tensor.numpy()
            else:
                noise = tensor

            # Create binary mask
            binary_mask = noise > threshold

            # Detect holes using fill method
            from scipy import ndimage

            filled = ndimage.binary_fill_holes(binary_mask)
            has_holes = (filled.sum() - binary_mask.sum()) > 0

            # ====================================================
            # Return reward
            # ====================================================
            return 0.0 if has_holes else 1.0

        except Exception as e:
            print(f"Error getting reward: {e}")
            return 0.0

    def connect_blender(self, plane, modifier, created_nodes):
        """
        Connect to Blender objects.

        Example:
            >>> env = BlenderColorRampEnvironment()
            >>> env.connect_blender(plane, modifier, nodes)
            >>> print("Connected to Blender")
        """
        self.plane = plane
        self.modifier = modifier
        self.created_nodes = created_nodes
        print("🔗 Connected to Blender")


class BlenderSamplerUtility:
    """
    Static utility functions for sampling trajectories with Blender environments.
    No state, just pure functions.
    """

    @staticmethod
    def sample_trajectory(
        env: BlenderColorRampEnvironment,
    ) -> List[Tuple[ColorRampState, int]]:
        """
        Sample a complete trajectory using the environment.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=3)
            >>> trajectory = BlenderSamplerUtility.sample_trajectory(env)
            >>> print(len(trajectory))  # 4 - scale + 3 colors
        """
        trajectory = []
        state = env.get_initial_state()

        while not env.is_terminal(state):
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break

            action = np.random.choice(valid_actions)
            trajectory.append((state.copy(), action))
            state = env.apply_action(state, action)

        return trajectory

    @staticmethod
    def sample_batch(
        env: BlenderColorRampEnvironment, n_trajectories: int
    ) -> List[List[Tuple[ColorRampState, int]]]:
        """
        Sample batch of trajectories using the environment.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=3)
            >>> trajectories = BlenderSamplerUtility.sample_batch(env, 100)
            >>> print(len(trajectories))  # 100
        """
        trajectories = []
        for i in range(n_trajectories):
            trajectory = BlenderSamplerUtility.sample_trajectory(env)
            trajectories.append(trajectory)

            if (i + 1) % 10 == 0:
                print(f"Sampled {i + 1}/{n_trajectories} trajectories")

        return trajectories

    @staticmethod
    def evaluate_trajectory(
        env: BlenderColorRampEnvironment, trajectory: List[Tuple[ColorRampState, int]]
    ) -> Dict:
        """
        Evaluate a trajectory using the environment.

        Example:
            >>> env = BlenderColorRampEnvironment()
            >>> traj = BlenderSamplerUtility.sample_trajectory(env)
            >>> stats = BlenderSamplerUtility.evaluate_trajectory(env, traj)
            >>> print(stats['reward'])  # 1.0 or 0.0
        """
        if not trajectory:
            return {"length": 0, "reward": 0.0}

        final_state = trajectory[-1][0]
        final_action = trajectory[-1][1]
        terminal_state = env.apply_action(final_state, final_action)

        return {
            "length": len(trajectory),
            "reward": env.get_reward(terminal_state),
            "terminal": env.is_terminal(terminal_state),
            "scale": terminal_state.scale,
            "colors_count": len(terminal_state.colors),
            "final_state": terminal_state,
        }

    @staticmethod
    def evaluate_batch(
        env: BlenderColorRampEnvironment,
        trajectories: List[List[Tuple[ColorRampState, int]]],
    ) -> List[Dict]:
        """
        Evaluate a batch of trajectories using the environment.

        Example:
            >>> env = BlenderColorRampEnvironment()
            >>> trajectories = BlenderSamplerUtility.sample_batch(env, 50)
            >>> stats = BlenderSamplerUtility.evaluate_batch(env, trajectories)
            >>> rewards = [s['reward'] for s in stats]
            >>> print(f"Average reward: {np.mean(rewards):.3f}")
        """
        return [
            BlenderSamplerUtility.evaluate_trajectory(env, traj)
            for traj in trajectories
        ]

    @staticmethod
    def sample_and_evaluate(
        env: BlenderColorRampEnvironment, n_trajectories: int
    ) -> Tuple[List[List[Tuple[ColorRampState, int]]], List[Dict]]:
        """
        Sample trajectories and evaluate them in one call.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=3)
            >>> trajectories, stats = BlenderSamplerUtility.sample_and_evaluate(env, 100)
            >>> success_rate = sum(1 for s in stats if s['reward'] > 0) / len(stats)
            >>> print(f"Success rate: {success_rate:.2%}")
        """
        trajectories = BlenderSamplerUtility.sample_batch(env, n_trajectories)
        stats = BlenderSamplerUtility.evaluate_batch(env, trajectories)
        return trajectories, stats


import colorsys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D


class BlenderVisualizationUtility:
    """
    Static utility functions for visualizing Blender color ramp environments and results.
    """

    @staticmethod
    def visualize_color_ramp(
        state, title: str = "Color Ramp", save_path: Optional[str] = None
    ):
        """
        Visualize the color ramp configuration.

        Example:
            >>> state = ColorRampState(scale=5.0, colors={0: 1, 2: 15, 4: 31})
            >>> BlenderVisualizationUtility.visualize_color_ramp(state, "My Color Ramp")
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Color ramp visualization
        if state.colors:
            positions = []
            colors = []

            for pos_idx in sorted(state.colors.keys()):
                color_id = state.colors[pos_idx]
                normalized_pos = (pos_idx + 1) / 5  # Assuming max_colors=5 for now
                rgba = BlenderVisualizationUtility._color_id_to_rgb(color_id)

                positions.append(normalized_pos)
                colors.append(rgba[:3])  # RGB only

            # Create gradient visualization
            gradient = np.linspace(0, 1, 256).reshape(1, -1)

            # Interpolate colors for display
            interp_colors = np.zeros((1, 256, 3))
            for i in range(256):
                x = i / 255.0
                # Find surrounding color positions
                left_idx = 0
                right_idx = len(positions) - 1

                for j in range(len(positions) - 1):
                    if positions[j] <= x <= positions[j + 1]:
                        left_idx = j
                        right_idx = j + 1
                        break

                if len(positions) == 1:
                    interp_colors[0, i] = colors[0]
                else:
                    # Linear interpolation
                    if x <= positions[0]:
                        interp_colors[0, i] = colors[0]
                    elif x >= positions[-1]:
                        interp_colors[0, i] = colors[-1]
                    else:
                        t = (x - positions[left_idx]) / (
                            positions[right_idx] - positions[left_idx]
                        )
                        interp_colors[0, i] = (1 - t) * np.array(
                            colors[left_idx]
                        ) + t * np.array(colors[right_idx])

            ax1.imshow(interp_colors, aspect="auto", extent=[0, 1, 0, 1])

            # Mark color positions
            for pos, color in zip(positions, colors):
                ax1.axvline(x=pos, color="white", linewidth=2, alpha=0.8)
                ax1.axvline(x=pos, color="black", linewidth=1, alpha=0.8)
                ax1.text(
                    pos,
                    0.5,
                    f"{pos:.2f}",
                    rotation=90,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        else:
            # Default black to white
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax1.imshow(gradient, cmap="gray", aspect="auto", extent=[0, 1, 0, 1])

        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Position")
        ax1.set_title("Color Ramp")
        ax1.set_yticks([])

        # Right: Color information
        ax2.axis("off")
        info_text = f"""
        COLOR RAMP INFO:
        
        Scale: {state.scale if state.scale else "Not set"}
        Colors placed: {len(state.colors)}
        
        Color Details:
        """

        if state.colors:
            for pos_idx in sorted(state.colors.keys()):
                color_id = state.colors[pos_idx]
                normalized_pos = (pos_idx + 1) / 5
                rgba = BlenderVisualizationUtility._color_id_to_rgb(color_id)
                info_text += (
                    f"\n  Pos {normalized_pos:.2f}: Color {color_id} {rgba[:3]}"
                )
        else:
            info_text += "\n  No colors placed"

        ax2.text(
            0.1,
            0.9,
            info_text,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"),
        )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def visualize_terrain(
        env, state, title: str = "Generated Terrain", save_path: Optional[str] = None
    ):
        """
        Visualize the 3D terrain generated by the color ramp state.

        Example:
            >>> env = BlenderColorRampEnvironment()
            >>> env.connect_blender(plane, modifier, nodes)
            >>> state = ColorRampState(scale=5.0, colors={0: 1, 2: 15})
            >>> BlenderVisualizationUtility.visualize_terrain(env, state)
        """
        if not env.is_terminal(state):
            print("⚠️ State is not terminal, terrain may be incomplete")

        try:
            # Get terrain data
            reward = env.get_reward(state)  # This applies state to Blender

            # Extract terrain tensor (duplicate the logic from get_reward)
            if not env.plane:
                print("❌ Blender not connected")
                return

            import bpy

            depsgraph = bpy.context.evaluated_depsgraph_get()
            plane_eval = env.plane.evaluated_get(depsgraph)
            mesh = plane_eval.to_mesh()

            verts = np.array([(v.co.x, v.co.y, v.co.z) for v in mesh.vertices])
            grid_size = int(np.sqrt(len(verts)))
            heights = verts[:, 2].reshape(grid_size, grid_size)

            plane_eval.to_mesh_clear()

            fig = plt.figure(figsize=(15, 10))

            # 1. 3D Surface
            ax1 = fig.add_subplot(2, 2, 1, projection="3d")
            x = np.arange(heights.shape[1])
            y = np.arange(heights.shape[0])
            X, Y = np.meshgrid(x, y)

            # Subsample for performance
            step = max(1, heights.shape[0] // 32)
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            heights_sub = heights[::step, ::step]

            surf = ax1.plot_surface(
                X_sub, Y_sub, heights_sub, cmap="terrain", alpha=0.8
            )
            ax1.set_title("3D Terrain Surface")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Height")

            # 2. Height map
            ax2 = fig.add_subplot(2, 2, 2)
            im = ax2.imshow(heights, cmap="terrain", origin="lower")
            ax2.set_title("Height Map")
            plt.colorbar(im, ax=ax2, label="Height")

            # 3. Binary threshold visualization
            ax3 = fig.add_subplot(2, 2, 3)
            threshold = 0.5
            binary_mask = heights > threshold
            ax3.imshow(binary_mask, cmap="RdYlBu_r", origin="lower")
            ax3.set_title(f"Binary Mask (threshold={threshold})")

            # 4. Statistics
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis("off")

            # Hole detection
            from scipy import ndimage

            filled = ndimage.binary_fill_holes(binary_mask)
            has_holes = (filled.sum() - binary_mask.sum()) > 0

            stats_text = f"""
            TERRAIN STATISTICS:
            
            Scale: {state.scale}
            Colors: {len(state.colors)}
            Reward: {reward:.1f}
            
            Terrain Info:
            Shape: {heights.shape}
            Height range: [{heights.min():.3f}, {heights.max():.3f}]
            Mean height: {heights.mean():.3f}
            Std height: {heights.std():.3f}
            
            Threshold Analysis:
            Threshold: {threshold}
            Above threshold: {binary_mask.sum()}/{binary_mask.size}
            Coverage: {100 * binary_mask.sum() / binary_mask.size:.1f}%
            
            Hole Detection:
            Has holes: {"Yes" if has_holes else "No"}
            Status: {"❌ FAIL" if has_holes else "✅ PASS"}
            """

            ax4.text(
                0.1,
                0.9,
                stats_text,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="lightcoral" if has_holes else "lightgreen",
                ),
            )

            plt.suptitle(
                f"{title} (Reward: {reward:.1f})", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"❌ Error visualizing terrain: {e}")

    @staticmethod
    def visualize_trajectory(
        env,
        trajectory: List[Tuple],
        title: str = "Trajectory Evolution",
        save_path: Optional[str] = None,
    ):
        """
        Visualize the evolution of a trajectory.

        Example:
            >>> trajectory = BlenderSamplerUtility.sample_trajectory(env)
            >>> BlenderVisualizationUtility.visualize_trajectory(env, trajectory)
        """
        if not trajectory:
            print("❌ Empty trajectory")
            return

        n_steps = len(trajectory)
        fig, axes = plt.subplots(2, min(4, n_steps), figsize=(4 * min(4, n_steps), 8))

        if n_steps == 1:
            axes = axes.reshape(2, 1)
        elif n_steps < 4:
            # Pad with empty subplots
            for i in range(n_steps, 4):
                if n_steps > 1:
                    axes[0, i].axis("off")
                    axes[1, i].axis("off")

        for i, (state, action) in enumerate(trajectory[:4]):  # Show first 4 steps
            # Color ramp evolution
            ax_ramp = axes[0, i] if n_steps > 1 else axes[0]

            if state.colors:
                positions = []
                colors = []

                for pos_idx in sorted(state.colors.keys()):
                    color_id = state.colors[pos_idx]
                    normalized_pos = (pos_idx + 1) / 5
                    rgba = BlenderVisualizationUtility._color_id_to_rgb(color_id)
                    positions.append(normalized_pos)
                    colors.append(rgba[:3])

                # Simple color bar visualization
                for j, (pos, color) in enumerate(zip(positions, colors)):
                    ax_ramp.barh(0, 0.1, left=pos - 0.05, color=color, height=0.5)
                    ax_ramp.text(pos, 0.7, f"{pos:.2f}", ha="center", fontsize=8)

            ax_ramp.set_xlim(0, 1)
            ax_ramp.set_ylim(0, 1)
            ax_ramp.set_title(f"Step {i + 1}: Action {action}")
            ax_ramp.set_xlabel("Position")
            ax_ramp.set_yticks([])

            # State info
            ax_info = axes[1, i] if n_steps > 1 else axes[1]
            ax_info.axis("off")

            info_text = f"""
            Step {i + 1}:
            
            Action: {action}
            Scale: {state.scale}
            Colors: {len(state.colors)}
            
            State:
            {state.colors}
            """

            ax_info.text(
                0.1,
                0.9,
                info_text,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
            )

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def visualize_batch_results(
        env,
        trajectories: List[List[Tuple]],
        stats: List[Dict],
        title: str = "Batch Results",
        save_path: Optional[str] = None,
    ):
        """
        Visualize batch sampling results.

        Example:
            >>> trajectories, stats = BlenderSamplerUtility.sample_and_evaluate(env, 100)
            >>> BlenderVisualizationUtility.visualize_batch_results(env, trajectories, stats)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Extract data
        rewards = [s["reward"] for s in stats]
        lengths = [s["length"] for s in stats]
        scales = [s["scale"] for s in stats if s["scale"] is not None]
        colors_counts = [s["colors_count"] for s in stats]

        # 1. Reward distribution
        axes[0, 0].hist(rewards, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Reward Distribution")
        axes[0, 0].set_xlabel("Reward")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(True, alpha=0.3)

        success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        axes[0, 0].text(
            0.7,
            0.8,
            f"Success Rate: {success_rate:.1%}",
            transform=axes[0, 0].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"),
        )

        # 2. Trajectory length distribution
        axes[0, 1].hist(
            lengths, bins=20, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        axes[0, 1].set_title("Trajectory Length Distribution")
        axes[0, 1].set_xlabel("Length")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Scale distribution
        if scales:
            axes[0, 2].hist(
                scales, bins=20, alpha=0.7, color="lightcoral", edgecolor="black"
            )
            axes[0, 2].set_title("Scale Distribution")
            axes[0, 2].set_xlabel("Scale")
            axes[0, 2].set_ylabel("Count")
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, "No scales recorded", ha="center", va="center")
            axes[0, 2].set_title("Scale Distribution")

        # 4. Colors vs Reward scatter
        axes[1, 0].scatter(colors_counts, rewards, alpha=0.6, c=rewards, cmap="RdYlGn")
        axes[1, 0].set_title("Colors vs Reward")
        axes[1, 0].set_xlabel("Number of Colors")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Scale vs Reward scatter (if available)
        if scales and len(scales) == len(rewards):
            scatter = axes[1, 1].scatter(
                scales, rewards, alpha=0.6, c=rewards, cmap="RdYlGn"
            )
            axes[1, 1].set_title("Scale vs Reward")
            axes[1, 1].set_xlabel("Scale")
            axes[1, 1].set_ylabel("Reward")
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label="Reward")
        else:
            axes[1, 1].text(
                0.5, 0.5, "Insufficient scale data", ha="center", va="center"
            )
            axes[1, 1].set_title("Scale vs Reward")

        # 6. Summary statistics
        axes[1, 2].axis("off")

        summary_text = f"""
        BATCH SUMMARY:
        
        Total trajectories: {len(trajectories)}
        
        Rewards:
        Success rate: {success_rate:.1%}
        Average reward: {np.mean(rewards):.3f}
        
        Trajectories:
        Average length: {np.mean(lengths):.1f}
        Length range: [{min(lengths)}, {max(lengths)}]
        
        Scales:
        Unique scales: {len(set(scales)) if scales else 0}
        Most common: {max(set(scales), key=scales.count) if scales else "N/A"}
        
        Colors:
        Average colors: {np.mean(colors_counts):.1f}
        Max colors: {max(colors_counts)}
        """

        axes[1, 2].text(
            0.1,
            0.9,
            summary_text,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"),
        )

        plt.suptitle(
            f"{title} ({len(trajectories)} trajectories)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _color_id_to_rgb(color_id: int) -> Tuple[float, float, float, float]:
        """Convert color ID to RGBA (internal helper method)"""
        if color_id == 0:
            return (0.0, 0.0, 0.0, 1.0)
        elif color_id == 1:
            return (1.0, 1.0, 1.0, 1.0)
        elif color_id < 8:
            colors = [
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
                (0.0, 1.0, 1.0, 1.0),
            ]
            return colors[color_id - 2]
        else:
            hue = ((color_id - 8) / (32 - 8)) * 360
            saturation = 0.8 if color_id % 2 == 0 else 1.0
            value = 0.9 if color_id % 3 == 0 else 0.7
            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            return (r, g, b, 1.0)


def create_blender_environment(
    max_colors: int = 5, num_color_choices: int = 32
) -> BlenderColorRampEnvironment:
    """
    Factory function to create Blender environment.

    Example:
        >>> env = create_blender_environment(max_colors=3, num_color_choices=8)
        >>> print(f"Created environment with {env.max_colors} max colors")
    """
    return BlenderColorRampEnvironment(max_colors, num_color_choices)
