import colorsys
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

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


def create_blender_procedure_dont_clear(config: BlenderProcedureConfig):
    clear_existing: bool = False
    BlenderProcedureBuilder.create_procedure(config, clear_existing)


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


# Simple function for Jupyter
def save_blend(name="experiment"):
    """Quick save function"""
    return BlenderFileSaverUtility.save_now(name)


# ====================================================
# Blender Experiment Environment
# ====================================================


class BlenderUtilities:
    """
    Nested utility class for Blender operations within the environment.
    """

    @staticmethod
    def create_color_ramp_procedure():
        """Create the color ramp geometry node procedure"""

        # Clear scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        for ng in bpy.data.node_groups:
            bpy.data.node_groups.remove(ng)

        # Create node group
        node_group = bpy.data.node_groups.new("ColorRampGroup", "GeometryNodeTree")

        # Add nodes
        group_input = node_group.nodes.new("NodeGroupInput")
        group_output = node_group.nodes.new("NodeGroupOutput")
        noise = node_group.nodes.new("ShaderNodeTexNoise")
        combine = node_group.nodes.new("ShaderNodeCombineXYZ")
        ramp = node_group.nodes.new("ShaderNodeValToRGB")
        set_pos = node_group.nodes.new("GeometryNodeSetPosition")

        # Set up interface
        node_group.interface.new_socket(
            "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        node_group.interface.new_socket(
            "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Connect nodes
        node_group.links.new(noise.outputs["Fac"], combine.inputs["X"])
        node_group.links.new(noise.outputs["Fac"], combine.inputs["Y"])
        node_group.links.new(noise.outputs["Fac"], combine.inputs["Z"])
        node_group.links.new(combine.outputs["Vector"], ramp.inputs["Fac"])
        node_group.links.new(ramp.outputs["Color"], set_pos.inputs["Offset"])
        node_group.links.new(group_input.outputs[0], set_pos.inputs["Geometry"])
        node_group.links.new(set_pos.outputs["Geometry"], group_output.inputs[0])

        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
        plane = bpy.context.active_object

        # Add subdivisions
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=15)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Add geometry nodes modifier
        geo_mod = plane.modifiers.new("GeometryNodes", "NODES")
        geo_mod.node_group = node_group

        return plane, {"noise": noise, "ramp": ramp}

    # ====================================================
    # Upadate Blender state with Color Ramp state Data
    # ====================================================
    @staticmethod
    def update_blender_from_state(state, nodes):
        """
        Update Blender scene from ColorRampState

        Args:
            state: ColorRampState object
            nodes: nodes dict with 'noise' and 'ramp' keys
        """
        # Update noise scale
        nodes["noise"].inputs["Scale"].default_value = state.scale

        # Update color ramp
        color_ramp = nodes["ramp"].color_ramp

        while len(color_ramp.elements) > len(state.colors):
            color_ramp.elements.remove(color_ramp.elements[-1])
        while len(color_ramp.elements) < len(state.colors):
            color_ramp.elements.new(0.5)

        for i, (pos_idx, color_id) in enumerate(sorted(state.colors.items())):
            position = ColorUtilities.calculate_position(pos_idx, 5)
            rgba = ColorUtilities.color_id_to_rgba_tuple(color_id)
            color_ramp.elements[i].position = position
            color_ramp.elements[i].color = rgba

        # Force update
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()

    @staticmethod
    def get_plane_from_scene(plane_name="Plane"):
        """
        Get the plane object from the current Blender scene.

        Args:
            plane_name: Name of the plane object to find

        Returns:
            Plane object or None if not found
        """
        try:
            # Try to get plane by name
            if plane_name in bpy.data.objects:
                return bpy.data.objects[plane_name]

            # If not found by name, look for any mesh object with geometry nodes
            for obj in bpy.data.objects:
                if obj.type == "MESH" and obj.modifiers:
                    for modifier in obj.modifiers:
                        if modifier.type == "NODES":
                            return obj

            # If still not found, get active object if it's a mesh
            if bpy.context.active_object and bpy.context.active_object.type == "MESH":
                return bpy.context.active_object

            return None

        except Exception as e:
            print(f"Error getting plane from scene: {e}")
            return None

    # ====================================================
    # Reward Helper for getting height map
    # ====================================================

    @staticmethod
    def extract_terrain_tensor(plane) -> Optional[torch.Tensor]:
        """
        Extract terrain height data from Blender plane as tensor.

        Args:
            plane: Blender plane object with geometry nodes
        Returns:
            2D tensor of height values with geometry nodes applied, or None if extraction fails
        """
        try:
            # CRITICAL FIX 1: Force scene update to ensure geometry nodes are evaluated
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()

            # CRITICAL FIX 2: Get evaluated object with all modifiers applied
            plane_eval = plane.evaluated_get(depsgraph)

            # CRITICAL FIX 3: Use proper parameters to ensure geometry nodes are included
            mesh = plane_eval.to_mesh(
                preserve_all_data_layers=True, depsgraph=depsgraph
            )

            # Check if we actually got vertices
            if len(mesh.vertices) == 0:
                print("Error: No vertices in evaluated mesh")
                plane_eval.to_mesh_clear()
                return None

            # Extract vertex heights
            verts = np.array([(v.co.x, v.co.y, v.co.z) for v in mesh.vertices])

            # CRITICAL FIX 4: Verify geometry nodes are actually working
            z_variation = verts[:, 2].std()
            if z_variation < 1e-6:
                print(
                    f"Warning: Terrain appears flat (std={z_variation:.8f}) - geometry nodes may not be applied"
                )
                # Don't return None, continue - might still be useful for debugging

            # Calculate grid size
            grid_size = int(np.sqrt(len(verts)))

            # CRITICAL FIX 5: Better handling of non-perfect squares
            expected_verts = grid_size * grid_size
            if expected_verts != len(verts):
                print(
                    f"Warning: {len(verts)} vertices don't form perfect {grid_size}x{grid_size} grid"
                )
                # Use available vertices up to perfect square
                verts = verts[:expected_verts]

            # Extract heights and reshape
            heights = verts[:, 2].reshape(grid_size, grid_size)
            tensor = torch.from_numpy(heights).float()

            # Clean up Blender mesh
            plane_eval.to_mesh_clear()

            return tensor

        except Exception as e:
            print(f"Error extracting terrain tensor: {e}")
            import traceback

            traceback.print_exc()
            return None

    @staticmethod
    def extract_color_data(plane) -> Optional[dict]:
        """
        Extract both height and color data from Blender geometry.
        FIXED VERSION - Proper mesh lifecycle and geometry nodes application.

        Args:
            plane: Blender plane object with geometry nodes applied
        Returns:
            Dictionary with height map, color map, and metadata
        """
        try:
            # CRITICAL FIX 1: Force scene update first
            bpy.context.view_layer.update()

            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()

            plane_eval = plane.evaluated_get(depsgraph)

            # CRITICAL FIX 2: Use proper parameters to ensure geometry nodes are applied
            mesh = plane_eval.to_mesh(
                preserve_all_data_layers=True, depsgraph=depsgraph
            )

            # CRITICAL FIX 3: Extract ALL data immediately before mesh becomes invalid
            # Extract vertex positions immediately
            verts = np.array([(v.co.x, v.co.y, v.co.z) for v in mesh.vertices])
            grid_size = int(np.sqrt(len(verts)))

            # Height map (Z coordinates)
            heights = verts[:, 2].reshape(grid_size, grid_size)

            # Color data - extract immediately if available
            color_data = None
            color_layers_count = 0

            if hasattr(mesh, "vertex_colors") and mesh.vertex_colors:
                color_layers_count = len(mesh.vertex_colors)
                print(f"Found {color_layers_count} vertex color layers")

                color_layer = mesh.vertex_colors.active
                if color_layer and hasattr(color_layer, "data"):
                    # Extract ALL color data immediately
                    colors = []
                    for poly in mesh.polygons:
                        for loop_idx in poly.loop_indices:
                            if loop_idx < len(color_layer.data):
                                color = color_layer.data[loop_idx].color
                                colors.append(
                                    [color[0], color[1], color[2], color[3]]
                                )  # RGBA

                    if colors:
                        colors = np.array(colors)
                        # Reshape to match grid if possible
                        if len(colors) == grid_size * grid_size:
                            color_data = colors.reshape(grid_size, grid_size, 4)
                        else:
                            color_data = colors

            # CRITICAL FIX 4: Only cleanup AFTER all data is extracted
            plane_eval.to_mesh_clear()

            return {
                "heights": heights,
                "colors": color_data,
                "grid_size": grid_size,
                "num_vertices": len(verts),
                "has_colors": color_data is not None,
                "color_layers": color_layers_count,
            }

        except Exception as e:
            print(f"Error extracting color data: {e}")
            # Ensure cleanup happens even on error
            try:
                if "plane_eval" in locals():
                    plane_eval.to_mesh_clear()
            except:
                pass
            return None

    @staticmethod
    def sample_noise_and_colors(
        noise_node, color_ramp_node, sample_size: int = 64
    ) -> Optional[dict]:
        """
        Sample the noise texture and color ramp directly from nodes.

        Args:
            noise_node: Blender noise texture node
            color_ramp_node: Blender color ramp node
            sample_size: Size of the sample grid

        Returns:
            Dictionary with sampled noise and color data
        """
        try:
            # Create sample grid
            x = np.linspace(-1, 1, sample_size)
            y = np.linspace(-1, 1, sample_size)
            X, Y = np.meshgrid(x, y)

            # Sample noise values
            noise_values = np.zeros((sample_size, sample_size))
            color_values = np.zeros((sample_size, sample_size, 4))  # RGBA

            # Get noise scale from node
            noise_scale = noise_node.inputs["Scale"].default_value

            # Sample each point
            for i in range(sample_size):
                for j in range(sample_size):
                    # Calculate noise value (simplified - Blender's actual noise is more complex)
                    # This is a rough approximation
                    noise_val = (
                        np.sin(X[i, j] * noise_scale) + np.cos(Y[i, j] * noise_scale)
                    ) * 0.5 + 0.5
                    noise_val = np.clip(noise_val, 0, 1)
                    noise_values[i, j] = noise_val

                    # Sample color ramp at this noise value
                    color = (
                        BlenderColorRampEnvironment.BlenderUtilities.sample_color_ramp(
                            color_ramp_node, noise_val
                        )
                    )
                    color_values[i, j] = color

            return {
                "noise_values": noise_values,
                "color_values": color_values,
                "sample_size": sample_size,
                "noise_scale": noise_scale,
                "sample_range": (-1, 1),
            }

        except Exception as e:
            print(f"Error sampling noise and colors: {e}")
            return None

    @staticmethod
    def sample_color_ramp(color_ramp_node, input_value: float) -> np.ndarray:
        """
        Sample a color from the color ramp at a given input value.

        Args:
            color_ramp_node: Blender color ramp node
            input_value: Value to sample (0.0 to 1.0)

        Returns:
            RGBA color array
        """
        try:
            color_ramp = color_ramp_node.color_ramp
            elements = color_ramp.elements

            # Clamp input value
            input_value = np.clip(input_value, 0.0, 1.0)

            # Find surrounding color stops
            if len(elements) == 0:
                return np.array([0.0, 0.0, 0.0, 1.0])  # Default black

            if len(elements) == 1:
                return np.array(elements[0].color[:])

            # Find the two color stops to interpolate between
            left_elem = elements[0]
            right_elem = elements[-1]

            for i in range(len(elements) - 1):
                if elements[i].position <= input_value <= elements[i + 1].position:
                    left_elem = elements[i]
                    right_elem = elements[i + 1]
                    break

            # Interpolate between the two colors
            if left_elem.position == right_elem.position:
                return np.array(left_elem.color[:])

            t = (input_value - left_elem.position) / (
                right_elem.position - left_elem.position
            )

            left_color = np.array(left_elem.color[:])
            right_color = np.array(right_elem.color[:])

            interpolated_color = (1 - t) * left_color + t * right_color

            return interpolated_color

        except Exception as e:
            print(f"Error sampling color ramp: {e}")
            return np.array([0.0, 0.0, 0.0, 1.0])

    @staticmethod
    def get_node_configuration_info(created_nodes: dict) -> dict:
        """
        Get detailed information about the node configuration.

        Args:
            created_nodes: Dictionary of created Blender nodes

        Returns:
            Dictionary with node configuration details
        """
        try:
            info = {
                "nodes": {},
                "noise_scale": None,
                "color_ramp_stops": [],
                "total_nodes": len(created_nodes),
            }

            for name, node in created_nodes.items():
                node_info = {
                    "type": node.bl_idname,
                    "name": node.name,
                    "inputs": {},
                    "outputs": {},
                }

                # Get input values
                for input_socket in node.inputs:
                    if hasattr(input_socket, "default_value"):
                        try:
                            node_info["inputs"][input_socket.name] = (
                                input_socket.default_value
                            )
                        except:
                            node_info["inputs"][input_socket.name] = "N/A"

                # Get output info
                for output_socket in node.outputs:
                    node_info["outputs"][output_socket.name] = output_socket.type

                info["nodes"][name] = node_info

                # Special handling for specific node types
                if name == "noise" and hasattr(node.inputs["Scale"], "default_value"):
                    info["noise_scale"] = node.inputs["Scale"].default_value

                if name == "ramp" and hasattr(node, "color_ramp"):
                    for i, element in enumerate(node.color_ramp.elements):
                        info["color_ramp_stops"].append(
                            {
                                "position": element.position,
                                "color": list(element.color[:]),
                                "index": i,
                            }
                        )

            return info

        except Exception as e:
            print(f"Error getting node configuration info: {e}")
            return {}

    @staticmethod
    def translate_state_to_blender(
        state: "ColorRampState", created_nodes: dict, max_colors: int
    ) -> bool:
        """
        Translate ColorRampState to Blender color ramp.
        FIXED VERSION - Properly handles Blender ColorRamp element requirements.
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

                # CRITICAL FIX: Handle color ramp elements properly
                if not state.colors:
                    # Default black to white gradient
                    # FIXED: Don't remove all elements, reset to 2 elements
                    while len(color_ramp.elements) > 2:
                        color_ramp.elements.remove(color_ramp.elements[-1])

                    # Ensure we have exactly 2 elements
                    while len(color_ramp.elements) < 2:
                        color_ramp.elements.new(1.0)

                    # Set the two default elements
                    color_ramp.elements[0].position = 0.0
                    color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black
                    color_ramp.elements[1].position = 1.0
                    color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # White

                    print("  Applied default black-to-white gradient")
                else:
                    # FIXED: Properly manage color ramp elements
                    sorted_positions = sorted(state.colors.keys())
                    needed_elements = len(sorted_positions)

                    # Adjust number of elements to match what we need
                    current_elements = len(color_ramp.elements)

                    if current_elements > needed_elements:
                        # Remove excess elements from the end
                        while len(color_ramp.elements) > needed_elements:
                            color_ramp.elements.remove(color_ramp.elements[-1])
                    elif current_elements < needed_elements:
                        # Add missing elements
                        while len(color_ramp.elements) < needed_elements:
                            # Add at a temporary position
                            color_ramp.elements.new(0.5)

                    # Now set the actual positions and colors
                    for i, position_idx in enumerate(sorted_positions):
                        if i < len(color_ramp.elements):
                            color_id = state.colors[position_idx]

                            # Calculate normalized position
                            normalized_pos = (position_idx + 1) / max_colors

                            # FIXED: Get RGBA color using proper method access
                            # Define basic colors inline to avoid class reference issues
                            if color_id == 0:  # BLACK
                                rgba_tuple = (0.0, 0.0, 0.0, 1.0)
                                color_name = "Black"
                            elif color_id == 1:  # WHITE
                                rgba_tuple = (1.0, 1.0, 1.0, 1.0)
                                color_name = "White"
                            elif color_id == 2:  # RED
                                rgba_tuple = (1.0, 0.0, 0.0, 1.0)
                                color_name = "Red"
                            elif color_id == 3:  # GREEN
                                rgba_tuple = (0.0, 1.0, 0.0, 1.0)
                                color_name = "Green"
                            elif color_id == 4:  # BLUE
                                rgba_tuple = (0.0, 0.0, 1.0, 1.0)
                                color_name = "Blue"
                            elif color_id == 5:  # YELLOW
                                rgba_tuple = (1.0, 1.0, 0.0, 1.0)
                                color_name = "Yellow"
                            elif color_id == 6:  # MAGENTA
                                rgba_tuple = (1.0, 0.0, 1.0, 1.0)
                                color_name = "Magenta"
                            elif color_id == 7:  # CYAN
                                rgba_tuple = (0.0, 1.0, 1.0, 1.0)
                                color_name = "Cyan"
                            else:  # Generated colors (8+)
                                # Simple procedural color generation
                                import colorsys

                                hue_range = 32 - 8  # Colors 8+ are procedural
                                hue = (
                                    ((color_id - 8) / hue_range) * 360
                                    if hue_range > 0
                                    else 0
                                )
                                saturation = 0.8 if color_id % 2 == 0 else 1.0
                                value = 0.9 if color_id % 3 == 0 else 0.7
                                r, g, b = colorsys.hsv_to_rgb(
                                    hue / 360, saturation, value
                                )
                                rgba_tuple = (r, g, b, 1.0)
                                color_name = f"Generated-{color_id}"

                            # Set element properties
                            element = color_ramp.elements[i]
                            element.position = normalized_pos
                            element.color = rgba_tuple

                            # Debug print
                            print(
                                f"  Added {color_name} at {normalized_pos:.1%} - {rgba_tuple}"
                            )

            # ENHANCED: Force more complete Blender update
            bpy.context.view_layer.update()

            # Force depsgraph update to ensure geometry nodes recalculate
            dg = bpy.context.evaluated_depsgraph_get()
            dg.update()

            # Additional update for geometry nodes
            for area in bpy.context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

            return True

        except Exception as e:
            print(f"Error translating state to Blender: {e}")
            import traceback

            traceback.print_exc()
            return False


class BlenderColorRampEnvironment:
    """
    Blender environment for color ramp generation - composition based.
    Similar to your HyperGrid approach.
    """

    def extract_blender_data(self, state: "ColorRampState") -> Optional[dict]:
        """
        Extract comprehensive data from Blender after applying a state.

        Args:
            state: ColorRampState to apply and extract data from

        Returns:
            Dictionary with all extracted Blender data
        """
        if not self.is_terminal(state):
            print("Warning: State is not terminal, data may be incomplete")

        try:
            # Apply state to Blender
            success = self.BlenderUtilities.translate_state_to_blender(
                state, self.created_nodes, self.max_colors
            )
            if not success:
                return None

            # Extract height data
            height_tensor = self.BlenderUtilities.extract_terrain_tensor(self.plane)

            # Extract color data
            color_data = self.BlenderUtilities.extract_color_data(self.plane)

            # Sample noise and colors directly from nodes
            noise_color_data = None
            if (
                self.created_nodes
                and "noise" in self.created_nodes
                and "ramp" in self.created_nodes
            ):
                noise_color_data = self.BlenderUtilities.sample_noise_and_colors(
                    self.created_nodes["noise"],
                    self.created_nodes["ramp"],
                    sample_size=64,
                )

            # Get node configuration info
            node_info = self.BlenderUtilities.get_node_configuration_info(
                self.created_nodes
            )

            # Calculate reward
            reward = (
                self.RewardUtilities.compute_reward(height_tensor)
                if height_tensor is not None
                else 0.0
            )

            return {
                "state": state,
                "height_tensor": height_tensor,
                "color_data": color_data,
                "noise_color_data": noise_color_data,
                "node_info": node_info,
                "reward": reward,
                "terrain_analysis": self.RewardUtilities.detect_holes(height_tensor)
                if height_tensor is not None
                else None,
                "blender_connected": self.plane is not None
                and self.created_nodes is not None,
            }

        except Exception as e:
            print(f"Error extracting Blender data: {e}")
            return None
        """
        Nested utility class for Blender operations within the environment.
        """

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

    @property
    def plane_tensor(self):
        return self.BlenderUtilities.extract_terrain_tensor(self.plane)

    def get_initial_state(self) -> "ColorRampState":
        return ColorRampState(
            max_colors=self.max_colors, num_color_choices=self.num_color_choices
        )

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
        Get reward for terminal state using BlenderUtilities.

        Example:
            >>> env = BlenderColorRampEnvironment()
            >>> plane, nodes = env.BlenderUtilities.create_color_ramp_procedure()
            >>> env.connect_blender(plane, None, nodes)
            >>> terminal_state = ColorRampState(scale=5.0, colors={0: 1, 1: 2})
            >>> reward = env.get_reward(terminal_state)
            >>> print(reward)  # 1.0 if no holes, 0.0 if holes
        """
        if not self.is_terminal(state):
            return 0.0

        try:
            # ====================================================
            # Convert state to config format
            # ====================================================
            config = {
                "noise_scale": state.scale if state.scale else 2.0,
                "noise_detail": 2.0,  # Default value
                "noise_roughness": 0.5,  # Default value
                "colors": [],
            }

            # Convert state colors to config format
            if not state.colors:
                # Default black to white
                config["colors"] = [(0.0, (0, 0, 0, 1)), (1.0, (1, 1, 1, 1))]
            else:
                # Convert state colors to config format
                for position_idx in sorted(state.colors.keys()):
                    color_id = state.colors[position_idx]
                    normalized_pos = self.ColorUtilities.calculate_position(
                        position_idx, self.max_colors
                    )
                    rgba_tuple = self.ColorUtilities.color_id_to_rgba_tuple(color_id)
                    config["colors"].append((normalized_pos, rgba_tuple))

            # ====================================================
            # Update procedure parameters
            # ====================================================
            if not self.created_nodes:
                raise ValueError("Blender nodes not connected")

            self.BlenderUtilities.update_procedure_parameters(
                self.created_nodes, config
            )

            # ====================================================
            # Extract terrain tensor
            # ====================================================
            if not self.plane:
                raise ValueError("Blender plane not connected")

            tensor = self.BlenderUtilities.extract_terrain_tensor(self.plane)
            if tensor is None:
                return 0.0

            # ====================================================
            # REWARD LOGIC: Hole detection
            # ====================================================
            threshold = 0.5

            # Use RewardUtilities for hole detection
            has_holes = self.RewardUtilities.detect_holes(tensor, threshold)

            # ====================================================
            #  Return Final reward
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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    GFLOW STUFF
# ╚══════════════════════════════════════════════════════════════════════╝


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
