import colorsys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import bpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Detect holes using fill method
from scipy import ndimage

import numpy as np
import torch
from scipy import ndimage
from typing import List, Optional, Tuple, Union
import bpy
import colorsys


import numpy as np
import torch
from scipy import ndimage
from typing import List, Optional, Tuple, Union
import bpy
import colorsys


# ====================================================
# Color Logic
# ====================================================


class ColorID(IntEnum):
    """Predefined color IDs for easy reference"""

    BLACK = 0
    WHITE = 1
    RED = 2
    GREEN = 3
    BLUE = 4
    YELLOW = 5
    MAGENTA = 6
    CYAN = 7
    # IDs 8+ are procedurally generated


class RGBA(NamedTuple):
    """Named tuple for RGBA color values (0.0 to 1.0)"""

    red: float
    green: float
    blue: float
    alpha: float = 1.0

    def __str__(self):
        return f"RGBA(r={self.red:.1f}, g={self.green:.1f}, b={self.blue:.1f}, a={self.alpha:.1f})"


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


# ====================================================
# Blender Experiment Environment
# ====================================================


import numpy as np
import torch
from scipy import ndimage
from typing import List, Optional, Tuple, Union, NamedTuple
import bpy
import colorsys
from enum import IntEnum


class RGBA(NamedTuple):
    """Named tuple for RGBA color values (0.0 to 1.0)"""

    red: float
    green: float
    blue: float
    alpha: float = 1.0

    def __str__(self):
        return f"RGBA(r={self.red:.1f}, g={self.green:.1f}, b={self.blue:.1f}, a={self.alpha:.1f})"


class ColorID(IntEnum):
    """Predefined color IDs for easy reference"""

    BLACK = 0
    WHITE = 1
    RED = 2
    GREEN = 3
    BLUE = 4
    YELLOW = 5
    MAGENTA = 6
    CYAN = 7
    # IDs 8+ are procedurally generated


class BlenderColorRampEnvironment:
    """
    Blender environment for color ramp generation - composition based.
    Similar to your HyperGrid approach.
    """

    class ColorUtilities:
        """
        Nested utility class for color operations and position calculations.
        """

        # Predefined color palette using RGBA named tuples
        BASIC_COLORS = {
            ColorID.BLACK: RGBA(0.0, 0.0, 0.0, 1.0),
            ColorID.WHITE: RGBA(1.0, 1.0, 1.0, 1.0),
            ColorID.RED: RGBA(1.0, 0.0, 0.0, 1.0),
            ColorID.GREEN: RGBA(0.0, 1.0, 0.0, 1.0),
            ColorID.BLUE: RGBA(0.0, 0.0, 1.0, 1.0),
            ColorID.YELLOW: RGBA(1.0, 1.0, 0.0, 1.0),
            ColorID.MAGENTA: RGBA(1.0, 0.0, 1.0, 1.0),
            ColorID.CYAN: RGBA(0.0, 1.0, 1.0, 1.0),
        }

        @staticmethod
        def calculate_position(position_idx: int, max_colors: int) -> float:
            """
            Calculate normalized position (0.0 to 1.0) for color ramp.

            Args:
                position_idx: Index of position (0 to max_colors-1)
                max_colors: Maximum number of color positions

            Returns:
                Normalized position between 0.0 and 1.0

            Example:
                >>> pos = ColorUtilities.calculate_position(2, 5)
                >>> print(f"Position 2 of 5 = {pos:.1%}")  # "Position 2 of 5 = 60.0%"
            """
            return (position_idx + 1) / max_colors

        @staticmethod
        def color_id_to_rgba_tuple(color_id: int) -> Tuple[float, float, float, float]:
            """
            Convert color ID to RGBA tuple (for Blender compatibility).

            Args:
                color_id: Integer color identifier

            Returns:
                Tuple of (red, green, blue, alpha) values
            """
            rgba = BlenderColorRampEnvironment.ColorUtilities.color_id_to_rgba(color_id)
            return (rgba.red, rgba.green, rgba.blue, rgba.alpha)

        @staticmethod
        def color_id_to_rgba(color_id: int) -> RGBA:
            """
            Convert color ID to RGBA tuple.

            Args:
                color_id: Integer color identifier

            Returns:
                RGBA named tuple with color values

            Example:
                >>> color = ColorUtilities.color_id_to_rgba(ColorID.RED)
                >>> print(color)  # "RGBA(r=1.0, g=0.0, b=0.0, a=1.0)"
            """
            # Use predefined colors for IDs 0-7
            if color_id in BlenderColorRampEnvironment.ColorUtilities.BASIC_COLORS:
                return BlenderColorRampEnvironment.ColorUtilities.BASIC_COLORS[color_id]

            # Generate procedural colors for higher IDs
            return (
                BlenderColorRampEnvironment.ColorUtilities._generate_procedural_color(
                    color_id
                )
            )

        @staticmethod
        def _generate_procedural_color(color_id: int, max_colors: int = 32) -> RGBA:
            """
            Generate procedural color using HSV color space.

            Args:
                color_id: Color identifier (should be >= 8)
                max_colors: Total number of available colors

            Returns:
                RGBA named tuple with generated color
            """
            # Map color_id to hue (0-360 degrees)
            hue_range = max_colors - 8  # Colors 8+ are procedural
            hue = ((color_id - 8) / hue_range) * 360

            # Vary saturation and value for variety
            saturation = 0.8 if color_id % 2 == 0 else 1.0
            value = 0.9 if color_id % 3 == 0 else 0.7

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)

            return RGBA(r, g, b, 1.0)

        @staticmethod
        def get_color_name(color_id: int) -> str:
            """
            Get human-readable name for color ID.

            Example:
                >>> name = ColorUtilities.get_color_name(ColorID.RED)
                >>> print(name)  # "Red"
            """
            color_names = {
                ColorID.BLACK: "Black",
                ColorID.WHITE: "White",
                ColorID.RED: "Red",
                ColorID.GREEN: "Green",
                ColorID.BLUE: "Blue",
                ColorID.YELLOW: "Yellow",
                ColorID.MAGENTA: "Magenta",
                ColorID.CYAN: "Cyan",
            }

            if color_id in color_names:
                return color_names[color_id]
            else:
                return f"Generated-{color_id}"

        @staticmethod
        def describe_color_ramp(state_colors: dict, max_colors: int) -> str:
            """
            Create human-readable description of color ramp.

            Example:
                >>> description = ColorUtilities.describe_color_ramp({1: 2, 3: 5}, 5)
                >>> print(description)
                # "Color Ramp: 40.0% Red, 80.0% Yellow"
            """
            if not state_colors:
                return "Color Ramp: Default (Black to White)"

            descriptions = []
            for pos_idx in sorted(state_colors.keys()):
                color_id = state_colors[pos_idx]
                position_percent = (
                    BlenderColorRampEnvironment.ColorUtilities.calculate_position(
                        pos_idx, max_colors
                    )
                    * 100
                )
                color_name = BlenderColorRampEnvironment.ColorUtilities.get_color_name(
                    color_id
                )
                descriptions.append(f"{position_percent:.1f}% {color_name}")

            return "Color Ramp: " + ", ".join(descriptions)

    def visualize_state_translation(self, state: "ColorRampState") -> str:
        """
        Visualize what will be translated to Blender for a given state.

        Args:
            state: ColorRampState to visualize

        Returns:
            String description of the translation

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=5)
            >>> state = ColorRampState(scale=10.0, colors={1: 2, 3: 5})
            >>> print(env.visualize_state_translation(state))
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BLENDER STATE TRANSLATION PREVIEW")
        lines.append("=" * 60)

        # Scale information
        if state.scale is not None:
            lines.append(f"Noise Scale: {state.scale}")
        else:
            lines.append("Noise Scale: NOT SET (will be set first)")

        lines.append("")

        # Color ramp information
        lines.append("Color Ramp Configuration:")

        if not state.colors:
            lines.append("  Default Gradient:")
            lines.append("    0.0% (0.000) -> Black (0.0, 0.0, 0.0, 1.0)")
            lines.append("  100.0% (1.000) -> White (1.0, 1.0, 1.0, 1.0)")
        else:
            lines.append("  Custom Colors:")
            for position_idx in sorted(state.colors.keys()):
                color_id = state.colors[position_idx]

                # Calculate position
                normalized_pos = self.ColorUtilities.calculate_position(
                    position_idx, self.max_colors
                )

                # Get color info
                color_name = self.ColorUtilities.get_color_name(color_id)
                rgba_tuple = self.ColorUtilities.color_id_to_rgba_tuple(color_id)

                lines.append(
                    f"    {normalized_pos:.1%} ({normalized_pos:.3f}) -> {color_name} {rgba_tuple}"
                )

        lines.append("")

        # Summary
        description = self.ColorUtilities.describe_color_ramp(
            state.colors, self.max_colors
        )
        lines.append(f"Summary: {description}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def debug_state_translation(self, state: "ColorRampState") -> dict:
        """
        Get detailed debug information about state translation.

        Args:
            state: ColorRampState to debug

        Returns:
            Dictionary with translation details
        """
        debug_info = {
            "state": state,
            "scale": state.scale,
            "num_colors": len(state.colors),
            "color_positions": {},
            "blender_ready": self.created_nodes is not None,
        }

        # Process each color
        for position_idx, color_id in state.colors.items():
            normalized_pos = self.ColorUtilities.calculate_position(
                position_idx, self.max_colors
            )
            color_name = self.ColorUtilities.get_color_name(color_id)
            rgba_tuple = self.ColorUtilities.color_id_to_rgba_tuple(color_id)

            debug_info["color_positions"][position_idx] = {
                "color_id": color_id,
                "color_name": color_name,
                "position_idx": position_idx,
                "normalized_position": normalized_pos,
                "rgba_tuple": rgba_tuple,
            }

        return debug_info
        """
        Nested utility class for Blender operations within the environment.
        """

    class BlenderUtilities:
        """
        Nested utility class for Blender operations within the environment.
        """

        @staticmethod
        def extract_terrain_tensor(plane) -> Optional[torch.Tensor]:
            """
            Extract terrain height data from Blender plane as tensor.

            Args:
                plane: Blender plane object

            Returns:
                2D tensor of height values, or None if extraction fails
            """
            try:
                depsgraph = bpy.context.evaluated_depsgraph_get()
                plane_eval = plane.evaluated_get(depsgraph)
                mesh = plane_eval.to_mesh()

                # Extract vertex heights
                verts = np.array([(v.co.x, v.co.y, v.co.z) for v in mesh.vertices])
                grid_size = int(np.sqrt(len(verts)))
                heights = verts[:, 2].reshape(grid_size, grid_size)
                tensor = torch.from_numpy(heights).float()

                # Clean up Blender mesh
                plane_eval.to_mesh_clear()

                return tensor

            except Exception as e:
                print(f"Error extracting terrain tensor: {e}")
                return None

        @staticmethod
        def translate_state_to_blender(
            state: "ColorRampState", created_nodes: dict, max_colors: int
        ) -> bool:
            """
            Translate ColorRampState to Blender color ramp.

            Args:
                state: ColorRampState to translate
                created_nodes: Dictionary of Blender nodes
                max_colors: Maximum number of colors

            Returns:
                True if successfully translated, False otherwise
            """
            try:
                # Apply noise scale
                if state.scale is not None and "noise" in created_nodes:
                    noise_node = created_nodes["noise"]
                    noise_node.inputs["Scale"].default_value = state.scale
                    print(f"  Applied noise scale: {state.scale}")

                # Apply color ramp
                if "ramp" in created_nodes:
                    ramp_node = created_nodes["ramp"]
                    color_ramp = ramp_node.color_ramp

                    # Clear existing points
                    while len(color_ramp.elements) > 0:
                        color_ramp.elements.remove(color_ramp.elements[0])

                    # Add color points
                    if not state.colors:
                        # Default black to white gradient
                        elem0 = color_ramp.elements.new(0.0)
                        elem0.color = (0.0, 0.0, 0.0, 1.0)  # Black tuple
                        elem1 = color_ramp.elements.new(1.0)
                        elem1.color = (1.0, 1.0, 1.0, 1.0)  # White tuple
                        print("  Applied default black-to-white gradient")
                    else:
                        # Add colors at their calculated positions
                        for position_idx in sorted(state.colors.keys()):
                            color_id = state.colors[position_idx]

                            # Calculate normalized position
                            normalized_pos = (position_idx + 1) / max_colors

                            # Get RGBA color as tuple (not named tuple for Blender compatibility)
                            rgba_tuple = BlenderColorRampEnvironment.ColorUtilities.color_id_to_rgba_tuple(
                                color_id
                            )

                            # Create color ramp element
                            elem = color_ramp.elements.new(normalized_pos)
                            elem.color = rgba_tuple

                            # Debug print
                            color_name = BlenderColorRampEnvironment.ColorUtilities.get_color_name(
                                color_id
                            )
                            print(
                                f"  Added {color_name} at {normalized_pos:.1%} - {rgba_tuple}"
                            )

                # Force Blender update
                bpy.context.view_layer.update()
                return True

            except Exception as e:
                print(f"Error translating state to Blender: {e}")
                return False

    class RewardUtilities:
        """
        Nested utility class for reward calculations within the Blender environment.
        """

        @staticmethod
        def detect_holes(
            tensor: Union[np.ndarray, torch.Tensor], threshold: float = 0.5
        ) -> bool:
            """
            Detect holes in terrain data using binary fill method.

            Args:
                tensor: 2D array/tensor of height values
                threshold: Height threshold for binary mask creation

            Returns:
                True if holes are detected, False otherwise
            """
            # Convert to numpy if needed
            if torch.is_tensor(tensor):
                noise = tensor.numpy()
            else:
                noise = tensor

            # Create binary mask
            binary_mask = noise > threshold
            filled = ndimage.binary_fill_holes(binary_mask)
            has_holes = (filled.sum() - binary_mask.sum()) > 0

            return has_holes

        @staticmethod
        def compute_reward(
            tensor: Union[np.ndarray, torch.Tensor], threshold: float = 0.5
        ) -> float:
            """
            Compute reward based on hole detection.

            Args:
                tensor: 2D array/tensor of height values
                threshold: Height threshold for binary mask creation

            Returns:
                1.0 if no holes detected, 0.0 if holes detected
            """
            has_holes = BlenderColorRampEnvironment.RewardUtilities.detect_holes(
                tensor, threshold
            )
            return 0.0 if has_holes else 1.0

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

    def get_initial_state(self) -> "ColorRampState":
        """
        Get the initial empty state.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=3)
            >>> state = env.get_initial_state()
            >>> print(state.scale)     # None
            >>> print(state.colors)    # {}
        """
        return ColorRampState()

    def get_valid_actions(self, state: "ColorRampState") -> List[int]:
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

    def apply_action(self, state: "ColorRampState", action: int) -> "ColorRampState":
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

    def is_terminal(self, state: "ColorRampState") -> bool:
        """
        Check if state is terminal.

        Example:
            >>> env = BlenderColorRampEnvironment(max_colors=2)
            >>> state = ColorRampState(scale=5.0, colors={0: 1, 1: 2})
            >>> print(env.is_terminal(state))  # True
        """
        return state.is_terminal(self.max_colors)

    def get_reward(self, state: "ColorRampState") -> float:
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
                    # Default black to white gradient
                    elem0 = color_ramp.elements.new(0.0)
                    elem0.color = self.ColorUtilities.BASIC_COLORS[ColorID.BLACK]
                    elem1 = color_ramp.elements.new(1.0)
                    elem1.color = self.ColorUtilities.BASIC_COLORS[ColorID.WHITE]
                else:
                    # Add colors at their calculated positions
                    for position_idx in sorted(state.colors.keys()):
                        color_id = state.colors[position_idx]

                        # Calculate normalized position using utility
                        normalized_pos = self.ColorUtilities.calculate_position(
                            position_idx, self.max_colors
                        )

                        # Get RGBA color using utility
                        rgba_color = self.ColorUtilities.color_id_to_rgba(color_id)

                        # Create color ramp element
                        elem = color_ramp.elements.new(normalized_pos)
                        elem.color = rgba_color  # RGBA named tuple works directly

                        # Debug print
                        color_name = self.ColorUtilities.get_color_name(color_id)
                        print(
                            f"  Added {color_name} at {normalized_pos:.1%} - {rgba_color}"
                        )

            # Force Blender update
            bpy.context.view_layer.update()

            # ====================================================
            # Extract noise tensor from Blender
            # ====================================================
            if not self.plane:
                raise ValueError("Blender not connected")

            tensor = self.BlenderUtilities.extract_terrain_tensor(self.plane)
            if tensor is None:
                return 0.0

            # ====================================================
            # REWARD LOGIC: Use nested RewardUtilities class
            # ====================================================
            threshold = 0.5
            return self.RewardUtilities.compute_reward(tensor, threshold)

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


class BlenderTrajectorySamplerUtilities:
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
            trajectory = BlenderTrajectorySamplerUtilities.sample_trajectory(env)
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
            BlenderTrajectorySamplerUtilities.evaluate_trajectory(env, traj)
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
        trajectories = BlenderTrajectorySamplerUtilities.sample_batch(
            env, n_trajectories
        )
        stats = BlenderTrajectorySamplerUtilities.evaluate_batch(env, trajectories)
        return trajectories, stats


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


def state_to_tensor(env, state):
    """Convert ColorRampState to tensor representation"""
    encoding = []

    # Scale encoding (one-hot)
    if state.scale is None:
        scale_encoding = [0.0] * len(env.available_scales)
    else:
        scale_encoding = [0.0] * len(env.available_scales)
        if state.scale in env.available_scales:
            scale_idx = env.available_scales.index(state.scale)
            scale_encoding[scale_idx] = 1.0
    encoding.extend(scale_encoding)

    # Color encoding (one-hot for each position)
    for pos in range(env.max_colors):
        if pos in state.colors:
            color_id = state.colors[pos]
            color_encoding = [0.0] * env.num_color_choices
            if 0 <= color_id < env.num_color_choices:
                color_encoding[color_id] = 1.0
        else:
            color_encoding = [0.0] * env.num_color_choices
        encoding.extend(color_encoding)

    # Step count (normalized)
    max_steps = env.max_colors + 1  # scale + colors
    step_encoding = [state.step_count / max_steps]
    encoding.extend(step_encoding)

    return torch.tensor(encoding, dtype=torch.float32)


class PolicyTrajectorySampler:
    """Sample trajectories using learned policy - mirroring HyperGrid implementation"""

    def __init__(self, env, model: TBModel = None, epsilon: float = 0.2):
        self.env = env
        self.model = model
        self.epsilon = epsilon

    def sample_trajectory(self, max_steps: int = 20):
        """Sample trajectory using policy with epsilon-greedy exploration"""
        trajectory = []
        state = self.env.get_initial_state()

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
                state_tensor = state_to_tensor(self.env, state)
                P_F_logits, _ = self.model(state_tensor)

                # Mask invalid actions
                action_mask = torch.tensor(
                    [
                        1.0 if a in valid_actions else 0.0
                        for a in range(P_F_logits.shape[0])
                    ]
                )
                masked_logits = P_F_logits.where(
                    action_mask.bool(), torch.tensor(-100.0)
                )

                # Sample from policy
                probs = F.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()

            trajectory.append(state.copy())
            state = self.env.apply_action(state, action)

        return trajectory

    def sample_batch(self, batch_size: int, max_steps: int = 20):
        """Sample batch of trajectories"""
        trajectories = []
        for _ in range(batch_size):
            traj = self.sample_trajectory(max_steps)
            trajectories.append(traj)
        return trajectories


def trajectory_balance_loss(model: TBModel, trajectories, env):
    """
    Trajectory Balance Loss - mirroring HyperGrid implementation
    """
    if len(trajectories) == 0:
        return torch.tensor(0.0, requires_grad=True)

    total_loss = torch.tensor(0.0, requires_grad=True)

    for traj in trajectories:
        if len(traj) < 2:  # Skip invalid trajectories
            continue

        # ====================================================
        # FORWARD path: Z_θ * ∏P_F(s_t|s_{t-1})
        # ====================================================
        log_forward = model.logZ

        for step in range(len(traj) - 1):
            current_state = traj[step]
            next_state = traj[step + 1]

            # Encode current state for neural network
            current_tensor = state_to_tensor(env, current_state)

            # Get forward policy logits
            P_F_logits, _ = model(current_tensor)

            # Find which action was taken
            valid_actions = env.get_valid_actions(current_state)
            action_taken = None

            for action in valid_actions:
                if env.apply_action(current_state, action) == next_state:
                    action_taken = action
                    break

            if action_taken is not None:
                # Mask invalid actions
                action_mask = torch.tensor(
                    [
                        1.0 if a in valid_actions else 0.0
                        for a in range(P_F_logits.shape[0])
                    ]
                )
                masked_logits = P_F_logits.where(
                    action_mask.bool(), torch.tensor(-100.0)
                )

                # Get probabilities
                probs = F.softmax(masked_logits, dim=0)
                log_forward = log_forward + torch.log(
                    probs[action_taken].clamp(min=1e-8)
                )

        # ====================================================
        # BACKWARD path: R(x) * ∏P_B(s_{t-1}|s_t)
        # ====================================================
        reward = env.get_reward(traj[-1])

        # Handle zero rewards by using a small epsilon instead of log(0)
        if reward <= 0:
            reward = 1e-8

        log_backward = torch.log(torch.tensor(reward, dtype=torch.float))

        for step in range(len(traj) - 1, 0, -1):
            current_state = traj[step]
            prev_state = traj[step - 1]

            # Encode current state
            current_tensor = state_to_tensor(env, current_state)

            # Get backward policy logits
            _, P_B_logits = model(current_tensor)

            # Find valid previous actions
            valid_prev_actions = []
            action_taken = None

            for action in range(P_B_logits.shape[0]):
                try:
                    # Check if taking this action from prev_state leads to current_state
                    if action in env.get_valid_actions(prev_state):
                        test_next_state = env.apply_action(prev_state, action)
                        if test_next_state == current_state:
                            valid_prev_actions.append(action)
                            action_taken = action
                except:
                    continue

            if valid_prev_actions and action_taken is not None:
                # Create mask and apply
                prev_action_mask = torch.tensor(
                    [
                        1.0 if a in valid_prev_actions else 0.0
                        for a in range(P_B_logits.shape[0])
                    ]
                )
                masked_logits = P_B_logits.where(
                    prev_action_mask.bool(), torch.tensor(-100.0)
                )
                probs = F.softmax(masked_logits, dim=0)

                log_backward = log_backward + torch.log(
                    probs[action_taken].clamp(min=1e-8)
                )

        # ====================================================
        # Apply trajectory balance equation
        # ====================================================
        trajectory_loss = (log_forward - log_backward) ** 2
        total_loss = total_loss + trajectory_loss

    return total_loss / len(trajectories)


def trajectory_balance_experiment(
    env: BlenderColorRampEnvironment,
    lr: float = 0.001,
    hidden_dim: int = 128,
    n_steps: int = 1000,
    batch_size: int = 32,
    epsilon: float = 0.3,
    max_trajectory_steps: int = 20,
):
    """
    Complete trajectory balance experiment - mirroring HyperGrid implementation
    """

    print(f"🚀 Training Trajectory Balance GFlowNet on Blender Environment")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Learning rate: {lr}")
    print(f"   Training steps: {n_steps}")
    print(f"   Batch size: {batch_size}")

    # ====================================================
    # Setup Environment and Model
    # ====================================================
    dummy_state = env.get_initial_state()
    state_dim = state_to_tensor(env, dummy_state).shape[0]
    action_dim = len(env.available_scales) + env.max_colors * env.num_color_choices

    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")

    # ====================================================
    # Create model and optimizer
    # ====================================================
    model = TBModel(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ====================================================
    # Create policy-based sampler
    # ====================================================
    sampler = PolicyTrajectorySampler(env, model, epsilon)

    print(f"   Initial logZ: {model.logZ.item():.4f}")

    # ====================================================
    # Training Loop
    # ====================================================
    losses = []
    log_z_values = []
    all_trajectories = []
    successful_trajectories = []

    for step in range(n_steps):
        # Sample trajectories using current policy
        step_trajectories = sampler.sample_batch(batch_size, max_trajectory_steps)
        all_trajectories.extend(step_trajectories)

        # Collect successful trajectories
        for traj in step_trajectories:
            if len(traj) > 0 and env.get_reward(traj[-1]) > 0:
                successful_trajectories.append(traj)

        # ====================================================
        # Update model
        # ====================================================
        loss = trajectory_balance_loss(model, step_trajectories, env)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        log_z_values.append(model.logZ.item())

        # Print progress
        if step % 50 == 0:
            success_count = sum(
                1
                for traj in step_trajectories
                if len(traj) > 0 and env.get_reward(traj[-1]) > 0
            )
            success_rate = success_count / len(step_trajectories)
            avg_length = sum(len(traj) for traj in step_trajectories) / len(
                step_trajectories
            )

            print(
                f"Step {step:4d}: Loss={loss_value:.4f}, LogZ={model.logZ.item():.3f}, "
                f"Success={success_rate:.1%}, AvgLen={avg_length:.1f}, "
                f"TotalSucc={len(successful_trajectories)}"
            )

    # ====================================================
    # Final Evaluation
    # ====================================================
    print(f"🎯 Final evaluation...")

    # Test final policy (no exploration)
    test_sampler = PolicyTrajectorySampler(env, model, epsilon=0.0)
    test_trajectories = test_sampler.sample_batch(100, max_trajectory_steps)

    test_successful = sum(
        1
        for traj in test_trajectories
        if len(traj) > 0 and env.get_reward(traj[-1]) > 0
    )
    test_success_rate = test_successful / len(test_trajectories)

    # Final metrics
    final_loss = losses[-1]
    final_logZ = log_z_values[-1]

    # Overall training success rate
    total_successful = sum(
        1 for traj in all_trajectories if len(traj) > 0 and env.get_reward(traj[-1]) > 0
    )
    overall_success_rate = total_successful / len(all_trajectories)

    print(f"\n📊 Final Results:")
    print(f"   Loss: {final_loss:.6f}")
    print(f"   LogZ: {final_logZ:.4f}")
    print(f"   Training Success: {overall_success_rate:.2%}")
    print(f"   Test Success: {test_success_rate:.2%}")
    print(f"   Total Trajectories: {len(all_trajectories)}")
    print(f"   Successful Trajectories: {len(successful_trajectories)}")

    return {
        "model": model,
        "losses": losses,
        "log_z_values": log_z_values,
        "successful_trajectories": successful_trajectories,
        "test_success_rate": test_success_rate,
        "overall_success_rate": overall_success_rate,
        "final_loss": final_loss,
    }


def run_blender_experiment(env):
    """
    Run a simple Blender GFlowNet experiment

    Example:
        >>> env = create_blender_environment(max_colors=5)
        >>> env.connect_blender(plane, modifier, nodes)
        >>> results = run_blender_experiment(env)
    """
    print("🧪 Running Blender GFlowNet Experiment")

    # Run experiment
    results = trajectory_balance_experiment(
        env, lr=0.001, hidden_dim=128, n_steps=1000, batch_size=32
    )

    # Basic visualization if successful trajectories found
    if results["successful_trajectories"]:
        print(
            f"✅ Found {len(results['successful_trajectories'])} successful trajectories!"
        )

        # Show a few successful examples
        for i, traj in enumerate(results["successful_trajectories"][:3]):
            final_state = traj[-1]
            print(
                f"Success {i + 1}: Scale={final_state.scale}, Colors={final_state.colors}"
            )
    else:
        print("❌ No successful trajectories found")

    return results


if __name__ == "__main__":
    print("🧪 Blender GFlowNet Training Ready")
    print("Usage:")
    print("  results = run_blender_experiment(env)")
    print("  results = trajectory_balance_experiment(env, lr=0.001, n_steps=500)")
