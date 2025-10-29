import colorsys
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import bpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D

# Detect holes using fill method
from scipy import ndimage

# ====================================================
# Color Funcitons
# ====================================================


class RGBA(NamedTuple):
    """Named tuple for RGBA color values (0.0 to 1.0)"""

    red: float
    green: float
    blue: float
    alpha: float = 1.0

    def __str__(self):
        return f"RGBA(r={self.red:.1f}, g={self.green:.1f}, b={self.blue:.1f}, a={self.alpha:.1f})"


class ColorUtilities:
    """
    Nested utility class for color operations and position calculations.
    Uses a curated 32-color palette instead of procedural generation.
    """

    # Popular 32-color palette (hex converted to RGBA tuples)
    PALETTE_32_COLORS = [
        # Basic colors (0-7)
        (0.0, 0.0, 0.0, 1.0),  # 0:  Black
        (1.0, 1.0, 1.0, 1.0),  # 1:  White
        (1.0, 0.0, 0.0, 1.0),  # 2:  Red
        (0.0, 1.0, 0.0, 1.0),  # 3:  Green
        (0.0, 0.0, 1.0, 1.0),  # 4:  Blue
        (1.0, 1.0, 0.0, 1.0),  # 5:  Yellow
        (1.0, 0.0, 1.0, 1.0),  # 6:  Magenta
        (0.0, 1.0, 1.0, 1.0),  # 7:  Cyan
        # Earth tones (8-15)
        (0.545, 0.271, 0.075, 1.0),  # 8:  Saddle Brown
        (0.824, 0.412, 0.118, 1.0),  # 9:  Chocolate
        (0.957, 0.643, 0.376, 1.0),  # 10: Sandy Brown
        (0.871, 0.722, 0.529, 1.0),  # 11: Burlywood
        (0.824, 0.706, 0.549, 1.0),  # 12: Tan
        (0.737, 0.561, 0.561, 1.0),  # 13: Rosy Brown
        (0.804, 0.522, 0.247, 1.0),  # 14: Peru
        (0.627, 0.322, 0.176, 1.0),  # 15: Sienna
        # Pastels (16-23)
        (1.0, 0.714, 0.757, 1.0),  # 16: Light Pink
        (1.0, 0.627, 0.478, 1.0),  # 17: Light Salmon
        (0.596, 0.984, 0.596, 1.0),  # 18: Pale Green
        (0.529, 0.808, 0.922, 1.0),  # 19: Sky Blue
        (0.867, 0.627, 0.867, 1.0),  # 20: Plum
        (0.941, 0.902, 0.549, 1.0),  # 21: Khaki
        (0.902, 0.902, 0.980, 1.0),  # 22: Lavender
        (1.0, 0.937, 0.835, 1.0),  # 23: Papaya Whip
        # Vibrant colors (24-31)
        (1.0, 0.271, 0.0, 1.0),  # 24: Orange Red
        (1.0, 0.078, 0.576, 1.0),  # 25: Deep Pink
        (0.0, 0.808, 0.820, 1.0),  # 26: Dark Turquoise
        (0.196, 0.804, 0.196, 1.0),  # 27: Lime Green
        (1.0, 0.388, 0.278, 1.0),  # 28: Tomato
        (0.255, 0.412, 0.882, 1.0),  # 29: Royal Blue
        (0.541, 0.169, 0.886, 1.0),  # 30: Blue Violet
        (1.0, 0.549, 0.0, 1.0),  # 31: Dark Orange
    ]

    # Human-readable color names
    COLOR_NAMES = [
        # Basic colors (0-7)
        "Black",
        "White",
        "Red",
        "Green",
        "Blue",
        "Yellow",
        "Magenta",
        "Cyan",
        # Earth tones (8-15)
        "Saddle Brown",
        "Chocolate",
        "Sandy Brown",
        "Burlywood",
        "Tan",
        "Rosy Brown",
        "Peru",
        "Sienna",
        # Pastels (16-23)
        "Light Pink",
        "Light Salmon",
        "Pale Green",
        "Sky Blue",
        "Plum",
        "Khaki",
        "Lavender",
        "Papaya Whip",
        # Vibrant colors (24-31)
        "Orange Red",
        "Deep Pink",
        "Dark Turquoise",
        "Lime Green",
        "Tomato",
        "Royal Blue",
        "Blue Violet",
        "Dark Orange",
    ]

    @staticmethod
    def calculate_position(position_idx: int, max_colors: int) -> float:
        """
        Calculate normalized position (0.0 to 1.0) for color ramp.
        """
        return (position_idx + 1) / max_colors

    @staticmethod
    def color_id_to_rgba_tuple(color_id: int) -> Tuple[float, float, float, float]:
        """
        Convert color ID to RGBA tuple (for Blender compatibility).

        FIXED: No self-references, uses class name directly.
        """
        if 0 <= color_id < len(ColorUtilities.PALETTE_32_COLORS):
            return ColorUtilities.PALETTE_32_COLORS[color_id]
        else:
            # Fallback to white for invalid IDs
            return (1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def color_id_to_rgba(color_id: int):
        """
        Convert color ID to RGBA named tuple.

        FIXED: Creates RGBA inline to avoid import issues.
        """
        rgba_tuple = ColorUtilities.color_id_to_rgba_tuple(color_id)

        # Create RGBA inline to avoid circular imports
        try:
            from collections import namedtuple

            RGBA = namedtuple("RGBA", ["red", "green", "blue", "alpha"])
            return RGBA(rgba_tuple[0], rgba_tuple[1], rgba_tuple[2], rgba_tuple[3])
        except:
            # Fallback to tuple if namedtuple fails
            return rgba_tuple

    @staticmethod
    def get_color_name(color_id: int) -> str:
        """
        Get human-readable name for color ID.

        FIXED: No self-references.
        """
        if 0 <= color_id < len(ColorUtilities.COLOR_NAMES):
            return ColorUtilities.COLOR_NAMES[color_id]
        else:
            return f"Unknown-{color_id}"

    @staticmethod
    def describe_color_ramp(state_colors: dict, max_colors: int) -> str:
        """
        Create human-readable description of color ramp.

        FIXED: Uses direct method calls, no self-references.
        """
        if not state_colors:
            return "Color Ramp: Default (Black to White)"

        descriptions = []
        for pos_idx in sorted(state_colors.keys()):
            color_id = state_colors[pos_idx]
            position_percent = (
                ColorUtilities.calculate_position(pos_idx, max_colors) * 100
            )
            color_name = ColorUtilities.get_color_name(color_id)
            descriptions.append(f"{position_percent:.1f}% {color_name}")

        return "Color Ramp: " + ", ".join(descriptions)

    @staticmethod
    def get_all_colors():
        """Get all 32 colors in the palette."""
        return ColorUtilities.PALETTE_32_COLORS.copy()

    @staticmethod
    def get_all_color_names():
        """Get all color names in the palette."""
        return ColorUtilities.COLOR_NAMES.copy()

    @staticmethod
    def display_palette():
        """Display the complete 32-color palette."""
        print("32-COLOR PALETTE")
        print("=" * 60)

        for color_id in range(len(ColorUtilities.PALETTE_32_COLORS)):
            rgba = ColorUtilities.color_id_to_rgba_tuple(color_id)
            name = ColorUtilities.get_color_name(color_id)

            # Convert to 0-255 for readability
            r255 = int(rgba[0] * 255)
            g255 = int(rgba[1] * 255)
            b255 = int(rgba[2] * 255)

            print(
                f"ID {color_id:2d}: {name:<15} RGB({r255:3d},{g255:3d},{b255:3d}) "
                f"RGBA({rgba[0]:.3f},{rgba[1]:.3f},{rgba[2]:.3f},{rgba[3]:.1f})"
            )

        print(f"\nTotal colors: {len(ColorUtilities.PALETTE_32_COLORS)}")

    @staticmethod
    def validate_color_id(color_id: int) -> bool:
        """Check if a color ID is valid."""
        return 0 <= color_id < len(ColorUtilities.PALETTE_32_COLORS)

    @staticmethod
    def get_random_color_id() -> int:
        """Get a random valid color ID."""
        import random

        return random.randint(0, len(ColorUtilities.PALETTE_32_COLORS) - 1)


@dataclass
class ColorRampConfig:
    """Configuration for ColorRamp environment"""

    available_scales: List[float] = None
    max_colors: int = 5
    num_color_choices: int = 32

    def __post_init__(self):
        if self.available_scales is None:
            self.available_scales = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]


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



@dataclass
class ColorRampEnvironmentConfig:
    """Configuration for ColorRamp environment"""

    available_scales: List[float] = None
    max_colors: int = 5
    num_color_choices: int = 32

    def __post_init__(self):
        if self.available_scales is None:
            self.available_scales = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]

    # ====================================================
    # ACTION ENCODING LOGIC
    # ====================================================

    def get_total_actions(self) -> int:
        """Get total number of possible actions"""
        # Scale actions + Color placement actions
        return len(self.available_scales) + (self.max_colors * self.num_color_choices)

    def encode_scale_action(self, scale_idx: int) -> int:
        """
        Encode scale selection action

        Args:
            scale_idx: Index in available_scales
        Returns:
            Action integer
        """
        if 0 <= scale_idx < len(self.available_scales):
            return scale_idx
        raise ValueError(f"Invalid scale index {scale_idx}")

    def encode_color_action(self, position: int, color_id: int) -> int:
        """
        Encode color placement action

        Args:
            position: Position index (0 to max_colors-1)
            color_id: Color ID (0 to num_color_choices-1)
        Returns:
            Action integer
        """
        if not (0 <= position < self.max_colors):
            raise ValueError(f"Invalid position {position}")
        if not (0 <= color_id < self.num_color_choices):
            raise ValueError(f"Invalid color_id {color_id}")

        scale_offset = len(self.available_scales)
        return scale_offset + position * self.num_color_choices + color_id

    def decode_action(self, action: int) -> dict:
        """
        Decode action integer back to action type and parameters

        Args:
            action: Action integer
        Returns:
            Dictionary with action info
        """
        if action < 0 or action >= self.get_total_actions():
            raise ValueError(f"Invalid action {action}")

        scale_actions = len(self.available_scales)

        if action < scale_actions:
            # Scale action
            return {
                "type": "scale",
                "scale_idx": action,
                "scale_value": self.available_scales[action],
            }
        else:
            # Color action
            color_action = action - scale_actions
            position = color_action // self.num_color_choices
            color_id = color_action % self.num_color_choices

            return {"type": "color", "position": position, "color_id": color_id}

    def get_valid_actions(self, state) -> List[int]:
        """
        Get valid actions for a given state

        Args:
            state: ColorRampState object
        Returns:
            List of valid action integers
        """
        valid_actions = []

        if state.scale is None:
            # Must choose scale first
            for i in range(len(self.available_scales)):
                valid_actions.append(self.encode_scale_action(i))
        else:
            # Can place colors at unoccupied positions
            occupied_positions = set(state.colors.keys())
            available_positions = [
                i for i in range(self.max_colors) if i not in occupied_positions
            ]

            for pos in available_positions:
                for color_id in range(self.num_color_choices):
                    valid_actions.append(self.encode_color_action(pos, color_id))

        return valid_actions

    def action_to_string(self, action: int) -> str:
        """Convert action to human-readable string"""
        action_info = self.decode_action(action)

        if action_info["type"] == "scale":
            return f"Scale[{action_info['scale_idx']}] = {action_info['scale_value']}"
        else:
            return f"Color[pos={action_info['position']}, id={action_info['color_id']}]"

    # ====================================================
    # FACTORY CONFIGS
    # ====================================================

    class DefaultConfigs:
        """Factory for common ColorRamp configurations"""

        @staticmethod
        def default():
            """Default configuration"""
            return ColorRampEnvironmentConfig()

        @staticmethod
        def small():
            """Small configuration for testing"""
            return ColorRampEnvironmentConfig(
                available_scales=[0.5, 1.0, 2.0], max_colors=3, num_color_choices=8
            )

        @staticmethod
        def large():
            """Large configuration for complex experiments"""
            return ColorRampEnvironmentConfig(
                available_scales=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
                max_colors=8,
                num_color_choices=64,
            )

        @staticmethod
        def notebook_test():
            """Configuration for notebook testing"""
            return ColorRampEnvironmentConfig(
                available_scales=[0.5, 1.0, 2.0, 5.0, 10.0],
                max_colors=5,
                num_color_choices=32,
            )


class ColorRampStateManager:
    """
    Static methods for managing ColorRampState environment logic
    """

    @staticmethod
    def state_to_tensor(state, config):
        """
        Convert ColorRampState to tensor representation

        Args:
            state: ColorRampState object
            config: ColorRampConfig object

        Returns:
            torch.Tensor: Encoded state tensor
        """
        encoding = []

        # Scale encoding (one-hot based on available scales)
        if state.scale is None:
            scale_encoding = [0.0] * len(config.available_scales)
        else:
            scale_encoding = [0.0] * len(config.available_scales)
            if state.scale in config.available_scales:
                scale_idx = config.available_scales.index(state.scale)
                scale_encoding[scale_idx] = 1.0
        encoding.extend(scale_encoding)

        # Colors encoding (one-hot for each position)
        for pos in range(config.max_colors):
            if pos in state.colors:
                color_id = state.colors[pos]
                color_encoding = [0.0] * config.num_color_choices
                if 0 <= color_id < config.num_color_choices:
                    color_encoding[color_id] = 1.0
            else:
                color_encoding = [0.0] * config.num_color_choices
            encoding.extend(color_encoding)

        # Step count (normalized)
        max_steps = config.max_colors + 1  # scale + colors
        step_encoding = [state.step_count / max_steps]
        encoding.extend(step_encoding)

        return torch.tensor(encoding, dtype=torch.float32)

    @staticmethod
    def get_valid_actions(state, config):
        """Get valid actions from current state"""
        if state.scale is None:
            # Must choose scale first - return scale action indices
            return list(range(len(config.available_scales)))

        # Can add colors to unoccupied positions
        occupied_positions = set(state.colors.keys())
        available_positions = [
            i for i in range(config.max_colors) if i not in occupied_positions
        ]

        if not available_positions:
            return []  # Terminal state

        valid_actions = []
        scale_offset = len(config.available_scales)

        for pos in available_positions:
            for color_id in range(config.num_color_choices):
                action = scale_offset + pos * config.num_color_choices + color_id
                valid_actions.append(action)

        return valid_actions

    @staticmethod
    def apply_action(state, action, config):
        """Apply action to state to get next state"""
        if action not in ColorRampStateManager.get_valid_actions(state, config):
            raise ValueError(f"Invalid action {action} for state {state}")

        next_state = state.copy()
        next_state.step_count += 1

        if state.scale is None:
            # Choose scale
            if 0 <= action < len(config.available_scales):
                next_state.scale = config.available_scales[action]
            else:
                raise ValueError(f"Invalid scale action {action}")
        else:
            # Add color
            scale_offset = len(config.available_scales)
            color_action = action - scale_offset

            position_idx = color_action // config.num_color_choices
            color_id = color_action % config.num_color_choices

            if position_idx in state.colors:
                raise ValueError(f"Position {position_idx} already occupied")

            next_state.colors[position_idx] = color_id

        return next_state

    @staticmethod
    def is_terminal(state, config):
        """Check if state is terminal"""
        return state.is_terminal(config.max_colors)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    BLENDER STUFF
# ╚══════════════════════════════════════════════════════════════════════╝


@dataclass
class BlenderProcedureConfig:
    """Configuration for Blender geometry node procedures"""

    # Node definitions: (name, node_type)
    nodes: List[Tuple[str, str]]

    # Connections: (from_node, from_output, to_node, to_input)
    connections: List[Tuple[str, str, str, str]]

    # Node group settings
    node_group_name: str = "ProcedureGroup"
    node_group_type: str = "GeometryNodeTree"

    # Interface sockets: (name, in_out, socket_type)
    input_sockets: List[Tuple[str, str]] = None
    output_sockets: List[Tuple[str, str]] = None

    def __post_init__(self):
        if self.input_sockets is None:
            self.input_sockets = [("Geometry", "NodeSocketGeometry")]
        if self.output_sockets is None:
            self.output_sockets = [("Geometry", "NodeSocketGeometry")]


class BlenderProcedureFactory:
    """Factory for creating common Blender geometry node procedures"""

    @staticmethod
    def color_ramp_terrain():
        """
        Create config for color ramp terrain generation procedure
        Based on your test_color_ramp configuration
        """
        return BlenderProcedureConfig(
            nodes=[
                ("noise", "ShaderNodeTexNoise"),
                ("combine", "ShaderNodeCombineXYZ"),
                ("ramp", "ShaderNodeValToRGB"),
                ("set_pos", "GeometryNodeSetPosition"),
            ],
            connections=[
                ("noise", "Fac", "combine", "X"),
                ("noise", "Fac", "combine", "Y"),
                ("noise", "Fac", "combine", "Z"),
                ("combine", "Vector", "ramp", "Fac"),
                ("ramp", "Color", "set_pos", "Offset"),
            ],
            node_group_name="ColorRampTerrainGroup",
        )

    @staticmethod
    def simple_noise():
        """Simple noise displacement procedure"""
        return BlenderProcedureConfig(
            nodes=[
                ("noise", "ShaderNodeTexNoise"),
                ("set_pos", "GeometryNodeSetPosition"),
            ],
            connections=[("noise", "Fac", "set_pos", "Offset")],
            node_group_name="SimpleNoiseGroup",
        )

    @staticmethod
    def displacement_with_scale():
        """Noise displacement with scale control"""
        return BlenderProcedureConfig(
            nodes=[
                ("noise", "ShaderNodeTexNoise"),
                ("multiply", "ShaderNodeMath"),
                ("set_pos", "GeometryNodeSetPosition"),
            ],
            connections=[
                ("noise", "Fac", "multiply", "Value"),
                ("multiply", "Value", "set_pos", "Offset"),
            ],
            node_group_name="ScaledDisplacementGroup",
        )


# ====================================================
# Helper functions for create_blender_procedure
# ====================================================


class BlenderProcedureBuilder:
    """Builder class to create Blender procedures from config"""

    @staticmethod
    def create_procedure(config: BlenderProcedureConfig, clear_existing: bool = True):
        """
        Create a Blender geometry node procedure from config

        Args:
            config: BlenderProcedureConfig defining the procedure
            clear_existing: Whether to clear existing objects and node groups

        Returns:
            Tuple of (plane_object, created_nodes_dict)
        """

        if clear_existing:
            BlenderProcedureBuilder._clear_scene()

        # Create node group
        node_group = BlenderProcedureBuilder._create_node_group(config)

        # Create and connect nodes
        created_nodes = BlenderProcedureBuilder._create_nodes(node_group, config)

        # Set up group interface and connections
        BlenderProcedureBuilder._setup_group_interface(
            node_group, config, created_nodes
        )

        # Create demonstration object (plane with subdivisions)
        plane = BlenderProcedureBuilder._create_demo_plane(node_group)

        return plane, created_nodes

    @staticmethod
    def _clear_scene():
        """Clear existing objects and node groups"""
        print("Clearing existing mesh objects...")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        print(f"Clearing {len(bpy.data.node_groups)} existing node groups...")
        for node_group in bpy.data.node_groups:
            bpy.data.node_groups.remove(node_group)

    @staticmethod
    def _create_node_group(config: BlenderProcedureConfig):
        """Create the geometry node group"""
        print(f"Creating geometry node group: {config.node_group_name}")
        return bpy.data.node_groups.new(config.node_group_name, config.node_group_type)

    @staticmethod
    def _create_nodes(node_group, config: BlenderProcedureConfig) -> Dict[str, Any]:
        """Create nodes and connections according to config"""
        nodes = node_group.nodes
        links = node_group.links
        created_nodes = {}

        print(f"Creating {len(config.nodes)} nodes...")
        for name, node_type in config.nodes:
            created_nodes[name] = nodes.new(node_type)
            print(f"  - Created {node_type} node named '{name}'")

        print(f"Creating {len(config.connections)} connections...")
        for from_node, from_out, to_node, to_in in config.connections:
            links.new(
                created_nodes[from_node].outputs[from_out],
                created_nodes[to_node].inputs[to_in],
            )
            print(f"  - Connected {from_node}.{from_out} → {to_node}.{to_in}")

        return created_nodes

    @staticmethod
    def _setup_group_interface(
        node_group, config: BlenderProcedureConfig, created_nodes
    ):
        """Set up group input/output interface"""
        print("Setting up node group interface...")

        # Add group input and output nodes
        group_input = node_group.nodes.new("NodeGroupInput")
        group_output = node_group.nodes.new("NodeGroupOutput")

        # Set up interface sockets (Blender 4.4+)
        try:
            if hasattr(node_group, "interface"):
                # Add input sockets
                for name, socket_type in config.input_sockets:
                    node_group.interface.new_socket(
                        name=name, in_out="INPUT", socket_type=socket_type
                    )
                    print(f"  - Added input socket: {name}")

                # Add output sockets
                for name, socket_type in config.output_sockets:
                    node_group.interface.new_socket(
                        name=name, in_out="OUTPUT", socket_type=socket_type
                    )
                    print(f"  - Added output socket: {name}")
        except Exception as e:
            print(f"  - Error setting up interface: {e}")

        # Connect to first and last nodes (assumes set_pos is the final node)
        if "set_pos" in created_nodes:
            node_group.links.new(
                group_input.outputs[0], created_nodes["set_pos"].inputs["Geometry"]
            )
            node_group.links.new(
                created_nodes["set_pos"].outputs["Geometry"], group_output.inputs[0]
            )
            print("  - Connected Group Input → Set Position → Group Output")

    @staticmethod
    def _create_demo_plane(node_group):
        """Create a plane with subdivisions and apply the node group"""
        print("Creating demonstration plane...")

        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
        plane = bpy.context.active_object

        # Add subdivisions
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=15)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Add geometry nodes modifier
        geo_mod = plane.modifiers.new("GeometryNodes", "NODES")

        try:
            geo_mod.node_group = node_group
            print(f"  - Applied node group: {node_group.name}")
        except Exception as e:
            print(f"  - Could not assign node group: {e}")

        return plane


# ====================================================
# IMPORTANT FUNCTION: this is the function that sets up Blender
# ====================================================


def create_blender_procedure(config: BlenderProcedureConfig):
    clear_existing: bool = True
    BlenderProcedureBuilder.create_procedure(config, clear_existing)


# ====================================================
# Blender Environment to Tensor
# ====================================================


class BlenderTensorUtility:
    """Utility class for extracting tensor data from Blender objects"""

    @staticmethod
    def extract_terrain_tensor(plane: bpy.types.Object) -> Optional[torch.Tensor]:
        """
        Extract terrain height data from Blender plane

        Args:
            plane: Blender mesh object (plane with geometry nodes applied)

        Returns:
            2D tensor of height values, or None if extraction fails
        """
        try:
            # Force scene update
            bpy.context.view_layer.update()
            depsgraph: bpy.types.Depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()

            # Get evaluated mesh
            plane_eval: bpy.types.Object = plane.evaluated_get(depsgraph)
            mesh: bpy.types.Mesh = plane_eval.to_mesh(
                preserve_all_data_layers=True, depsgraph=depsgraph
            )

            if len(mesh.vertices) == 0:
                plane_eval.to_mesh_clear()
                return None

            # Extract vertices
            verts: list[tuple[float, float, float]] = [
                (v.co.x, v.co.y, v.co.z) for v in mesh.vertices
            ]
            grid_size: int = int(len(verts) ** 0.5)
            heights: list[float] = [v[2] for v in verts]

            # Clean up
            plane_eval.to_mesh_clear()

            # Reshape to grid
            heights_array: np.ndarray = np.array(heights).reshape(grid_size, grid_size)
            terrain_tensor: torch.Tensor = torch.from_numpy(heights_array).float()

            return terrain_tensor

        except Exception as e:
            print(f"❌ Error extracting terrain: {e}")
            return None


# ====================================================
# Convenience function with type hints
# ====================================================


def extract_terrain_tensor(plane: bpy.types.Object) -> Optional[torch.Tensor]:
    """
    Extract terrain height data from Blender plane

    Args:
        plane: Blender mesh object (plane with geometry nodes applied)

    Returns:
        2D tensor of height values, or None if extraction fails
    """
    return BlenderTensorUtility.extract_terrain_tensor(plane)


# ====================================================
# File Utilities
# ====================================================


class BlenderFileSaverUtility:
    """Super simple Blender file saver"""

    @staticmethod
    def save_now(name="blender_save"):
        import os

        """Save current Blender file with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.blend"

        # Create saves directory if it doesn't exist
        save_dir = "./saves"
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)

        try:
            bpy.ops.wm.save_as_mainfile(filepath=filepath)
            print(f"✅ Saved: {filename}")
            return filepath
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return None


# ====================================================
# Save File Funtion
# ====================================================


def save_blend(name="experiment"):
    """Quick save function"""
    return BlenderFileSaverUtility.save_now(name)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    GFLOW STUFF
# ╚══════════════════════════════════════════════════════════════════════╝

# ====================================================
# This is the Arcitecture of a Standard GFN with trajectory Balance
# ====================================================


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


# ====================================================
# Trajectory Sample Logic
# ====================================================


class ColorRampTrajectorySampler:
    """
    Trajectory sampler specifically for ColorRamp environment
    Uses ColorRampEnvironmentConfig methods
    """

    def __init__(self, env_config: ColorRampEnvironmentConfig):
        self.env_config = env_config

    # ====================================================
    # Random Sampleing Logic
    # ====================================================

    def sample_random_trajectory(
        self, max_steps: int = 10
    ) -> List[Tuple[ColorRampState, int]]:
        """
        Sample random trajectory using env_config methods

        Returns:
            List of (state, action) pairs
        """
        trajectory = []
        current_state = ColorRampState()  # Start empty

        for step in range(max_steps):
            # Check if terminal
            if ColorRampStateManager.is_terminal(current_state, self.env_config):
                break

            # Get valid actions using env_config
            valid_actions = self.env_config.get_valid_actions(current_state)
            if not valid_actions:
                break

            # Sample random action
            action = random.choice(valid_actions)

            # Store state-action pair
            trajectory.append((current_state.copy(), action))

            # Apply action to get next state
            try:
                current_state = ColorRampStateManager.apply_action(
                    current_state, action, self.env_config
                )
            except ValueError as e:
                print(f"Invalid action {action}: {e}")
                break

        return trajectory

    # ====================================================
    # Using state of the Model to sample Trajectorys
    # (How we run mass infernect on the models)
    # ====================================================

    def sample_policy_trajectory(
        self, model: TBModel, epsilon: float = 0.1, max_steps: int = 10
    ) -> List[Tuple[ColorRampState, int]]:
        """
        Sample trajectory using models policy with epsilon-greedy

        Args:
            model: TBModel for policy
            epsilon: Exploration probability
            max_steps: Maximum steps

        Returns:
            List of (state, action) pairs
        """
        trajectory = []
        current_state = ColorRampState()  # Start empty

        for step in range(max_steps):
            # Check if terminal
            if ColorRampStateManager.is_terminal(current_state, self.env_config):
                break

            # Get valid actions using env_config
            valid_actions = self.env_config.get_valid_actions(current_state)
            if not valid_actions:
                break

            # Choose action: epsilon-greedy
            if random.random() < epsilon:
                # Explore: random action
                action = random.choice(valid_actions)
            else:
                # Exploit: use models policy
                state_tensor = ColorRampStateManager.state_to_tensor(
                    current_state, self.env_config
                )

                with torch.no_grad():
                    P_F_logits, _ = model(state_tensor)

                # Mask invalid actions
                action_mask = torch.full_like(P_F_logits, float("-inf"))
                for valid_action in valid_actions:
                    action_mask[valid_action] = 0.0

                masked_logits = P_F_logits + action_mask
                action_probs = F.softmax(masked_logits, dim=0)

                # Sample from policy
                action = torch.multinomial(action_probs, 1).item()

            # Store state-action pair
            trajectory.append((current_state.copy(), action))

            # Apply action using env_config knowledge
            try:
                current_state = ColorRampStateManager.apply_action(
                    current_state, action, self.env_config
                )
            except ValueError as e:
                print(f"Invalid action {action}: {e}")
                break

        return trajectory

    # ====================================================
    # Batch traing
    # ====================================================

    def sample_batch(
        self,
        batch_size: int,
        use_policy: bool = False,
        model: Optional[TBModel] = None,
        epsilon: float = 0.1,
    ) -> List[List[Tuple[ColorRampState, int]]]:
        """
        Sample batch of trajectories

        Args:
            batch_size: Number of trajectories
            use_policy: Whether to use models policy
            model: TBModel (required if use_policy=True)
            epsilon: Exploration rate for policy

        Returns:
            List of trajectories
        """
        trajectories = []

        for i in range(batch_size):
            try:
                if use_policy and model is not None:
                    traj = self.sample_policy_trajectory(model, epsilon)
                else:
                    traj = self.sample_random_trajectory()

                trajectories.append(traj)
            except Exception as e:
                print(f"Error sampling trajectory {i}: {e}")
                continue

        return trajectories

    # ====================================================
    # Print trajectory in Human readable format
    # ====================================================

    def print_trajectory_info(self, trajectory: List[Tuple[ColorRampState, int]]):
        """Print detailed trajectory information using env_config methods"""
        print(f"Trajectory length: {len(trajectory)}")

        for i, (state, action) in enumerate(trajectory):
            # Decode action using env_config
            action_info = self.env_config.decode_action(action)
            action_str = self.env_config.action_to_string(action)

            print(f"  Step {i}: {action_str}")
            print(f"    State: scale={state.scale}, colors={state.colors}")
            print(f"    Action info: {action_info}")

        # Show final state
        if trajectory:
            final_state = trajectory[-1][0]
            final_action = trajectory[-1][1]

            # Apply final action to see end result
            try:
                end_state = ColorRampStateManager.apply_action(
                    final_state, final_action, self.env_config
                )
                is_terminal = ColorRampStateManager.is_terminal(
                    end_state, self.env_config
                )
                print(
                    f"  Final state: scale={end_state.scale}, colors={end_state.colors}"
                )
                print(f"  Terminal: {is_terminal}")
            except Exception as e:
                print(f"  Error applying final action: {e}")


class RewardUtilities:
    """
    Utility class for reward calculations within the Blender environment.
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
        has_holes = RewardUtilities.detect_holes(tensor, threshold)
        return 0.0 if has_holes else 1.0


def state_to_tensor(state):
    """
    Convert ColorRampState to tensor representation

    Args:
        state: ColorRampState object

    Returns:
        torch.Tensor: Encoded state tensor
    """
    encoding = []

    # Scale encoding (normalized)
    scale_val = state.scale if state.scale is not None else 0.0
    encoding.append(scale_val / 20.0)  # Normalize assuming max scale ~20

    # Colors encoding (position + color_id pairs)
    max_colors = 5  # From your usage
    for pos in range(max_colors):
        if pos in state.colors:
            encoding.append((pos + 1) / max_colors)  # Position (normalized)
            encoding.append(state.colors[pos] / 32.0)  # Color ID (normalized)
        else:
            encoding.append(0.0)  # No color at this position
            encoding.append(0.0)

    # Step count (normalized)
    encoding.append(state.step_count / 6.0)  # Normalize assuming max ~6 steps

    return torch.tensor(encoding, dtype=torch.float32)
