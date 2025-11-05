from argparse import Action

import bpy

#TODO there are magic number for a bunch of stuff and I need to chainge that
def generate_template_blend(filepath: str = "single_color_ramp.blend"):
    """
    Generate a template .blend file with color ramp terrain setup

    Args:
        filepath: Where to save the template

    Returns:
        Path to saved template
    """
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "TerrainPlane"

    # Add subdivisions
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=15)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create geometry nodes setup
    node_group = bpy.data.node_groups.new("TerrainGenerator", "GeometryNodeTree")
    nodes = node_group.nodes
    links = node_group.links

    # Set up interface
    if hasattr(node_group, 'interface'):
        node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    # Create nodes
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "NoiseTexture"
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.name = "TerrainColorRamp"
    separate_xyz_1 = nodes.new("ShaderNodeSeparateXYZ")
    add_1 = nodes.new("ShaderNodeMath")
    add_2 = nodes.new("ShaderNodeMath")
    map_range = nodes.new("ShaderNodeMapRange")
    combine_xyz = nodes.new("ShaderNodeCombineXYZ")
    set_pos = nodes.new("GeometryNodeSetPosition")

    # Configure nodes
    noise.noise_dimensions = '4D'
    noise.noise_type = 'FBM'
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["W"].default_value = 50.0
    noise.normalize = False

    # Set default colors
    set_colors_in_ramp(color_ramp, ["Black", "White"])

    add_1.operation = 'ADD'
    add_2.operation = 'ADD'

    map_range.clamp = False
    map_range.inputs["From Min"].default_value = 0
    map_range.inputs["From Max"].default_value = 3
    map_range.inputs["To Min"].default_value = 0
    map_range.inputs["To Max"].default_value = 1

    combine_xyz.inputs["X"].default_value = 0.0
    combine_xyz.inputs["Y"].default_value = 0.0

    # Create connections
    links.new(group_input.outputs[0], set_pos.inputs["Geometry"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], separate_xyz_1.inputs["Vector"])
    links.new(separate_xyz_1.outputs["X"], add_1.inputs[0])
    links.new(separate_xyz_1.outputs["Y"], add_1.inputs[1])
    links.new(separate_xyz_1.outputs["Z"], add_2.inputs[0])
    links.new(add_1.outputs["Value"], add_2.inputs[1])
    links.new(add_2.outputs["Value"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], combine_xyz.inputs["Z"])
    links.new(combine_xyz.outputs["Vector"], set_pos.inputs["Offset"])
    links.new(set_pos.outputs["Geometry"], group_output.inputs[0])

    # Apply to plane
    geo_mod = plane.modifiers.new("GeometryNodes", "NODES")
    geo_mod.node_group = node_group

    # Force update
    bpy.context.view_layer.update()

    # Save template
    bpy.ops.wm.save_as_mainfile(filepath=filepath, compress=True)

    print(f"✓ Generated template: {filepath}")
    return filepath

from pathlib import Path

# Get the directory where THIS file (single_color_ramp.py) is located
THIS_DIR = Path(__file__).parent

def load_blend_single_color_ramp():
    """Load the single color ramp template from the same directory as this file"""
    blend_path = THIS_DIR / "single_color_ramp.blend"
    load_blend(str(blend_path))



import bpy
import pytest
import torch
from pathlib import Path
from blender_setup_utils import set_colors_in_ramp, load_blend, BlenderTensorUtility

from typing import List, Tuple, Optional
import bpy
import torch


def set_noise_params(w: float, scale: float):
    """
    Set the W and Scale parameters of the noise texture

    Args:
        w: W coordinate for 4D noise
        scale: Scale of the noise texture
    """
    node_group = bpy.data.node_groups.get("TerrainGenerator")
    if node_group is None:
        raise ValueError("TerrainGenerator node group not found")

    noise_node = node_group.nodes.get("NoiseTexture")
    if noise_node is None:
        raise ValueError("NoiseTexture node not found")

    noise_node.inputs["W"].default_value = w
    noise_node.inputs["Scale"].default_value = scale

    # Force update
    bpy.context.view_layer.update()

    print(f"✓ Set noise params: W={w}, Scale={scale}")

def get_color_ramp_state(max_colors: int) -> dict:
    """
    Get the current state of the color ramp

    Args:
        max_colors: Maximum number of colors allowed on the ramp

    Returns:
        Dictionary with:
            - filled_slots: List of slot indices that are filled
            - empty_slots: List of slot indices that are empty
            - colors: Dictionary mapping slot index -> color name
            - positions: Dictionary mapping slot index -> position value
            - is_default: Whether the color ramp is in default state (Black->White)
    """
    node_group = bpy.data.node_groups.get("TerrainGenerator")
    if node_group is None:
        raise ValueError("TerrainGenerator node group not found")

    color_ramp = node_group.nodes.get("TerrainColorRamp")
    if color_ramp is None:
        raise ValueError("TerrainColorRamp node not found")

    # Get all elements in the color ramp
    elements = color_ramp.color_ramp.elements

    # Create evenly spaced target positions
    target_positions = [i / (max_colors - 1) for i in range(max_colors)]

    # Map actual colors to slots (with tolerance for position matching)
    filled_slots = []
    colors = {}
    positions = {}

    tolerance = 0.05  # Position matching tolerance

    for slot_idx, target_pos in enumerate(target_positions):
        # Find if there's an element at this position
        for element in elements:
            if abs(element.position - target_pos) < tolerance:
                filled_slots.append(slot_idx)
                # Get color as RGBA tuple
                color_rgba = tuple(element.color)
                colors[slot_idx] = color_rgba
                positions[slot_idx] = element.position
                break

    empty_slots = [i for i in range(max_colors) if i not in filled_slots]

    # Check if default (only 2 colors: Black at 0.0 and White at 1.0)
    is_default = False
    if len(filled_slots) == 2:
        first_color = colors.get(0)
        last_color = colors.get(max_colors - 1)

        # Check if it's black and white
        black = (0.0, 0.0, 0.0, 1.0)
        white = (1.0, 1.0, 1.0, 1.0)

        def colors_match(c1, c2, tol=0.01):
            return all(abs(a - b) < tol for a, b in zip(c1, c2))

        if (first_color and last_color and
                colors_match(first_color, black) and
                colors_match(last_color, white)):
            is_default = True

    return {
        'filled_slots': filled_slots,
        'empty_slots': empty_slots,
        'colors': colors,
        'positions': positions,
        'is_default': is_default,
        'num_filled': len(filled_slots),
        'num_empty': len(empty_slots)
    }

def add_color_to_slot(slot_idx: int, color_name: str, max_colors: int):
    """
    Add a color to a specific slot on the color ramp

    Args:
        slot_idx: Index of the slot (0 to max_colors-1)
        color_name: Name of the color to add
        max_colors: Maximum number of colors allowed
    """
    from blender_setup_utils import color_df

    # Get the color ramp
    node_group = bpy.data.node_groups.get("TerrainGenerator")
    color_ramp = node_group.nodes.get("TerrainColorRamp")

    # Calculate position for this slot
    position = slot_idx / (max_colors - 1)

    # Get color from palette
    color_row = color_df[color_df['name'] == color_name]
    if color_row.empty:
        raise ValueError(f"Color '{color_name}' not found in palette")

    color = color_row.iloc[0]
    color_rgba = (color['red'], color['green'], color['blue'], color['alpha'])

    # Add the element at the position
    element = color_ramp.color_ramp.elements.new(position)
    element.color = color_rgba

    # Force update
    bpy.context.view_layer.update()

    print(f"✓ Added {color_name} to slot {slot_idx} at position {position:.3f}")




from pydantic import BaseModel, Field, ConfigDict
from typing import Dict
from datetime import datetime
import pickle

# ============================================================================
# sampling and replay buffers
# ============================================================================
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel, Field
import torch


"""
Blender Terrain API
Clean interface for interacting with the TerrainGenerator node group
"""
import bpy
from typing import List, Dict, Tuple


class BlenderTerrainAPI:
    """API for controlling terrain generation in Blender"""
    
    def reset_env(self):
        load_blend_single_color_ramp()

    def blender_env_to_tensor(self, **config) -> torch.Tensor:
        """
        Read Blender environment and serialize to tensor with color history.

        Reads the actual color ramp elements in order, not mapped to slots.
        """

        # Default config values
        max_colors = config.get('max_colors', 32)
        empty_value = config.get('empty_value', -1.0)
        w_range = config.get('w_range', (0.0, 100.0))
        scale_range = config.get('scale_range', (0.0, 50.0))

        parts = []

        # Read W and Scale from Blender
        noise_w = self.get_noise_w()
        noise_scale = self.get_noise_scale()

        # Normalize to [0, 1]
        w_normalized = (noise_w - w_range[0]) / (w_range[1] - w_range[0])
        scale_normalized = (noise_scale - scale_range[0]) / (scale_range[1] - scale_range[0])

        parts.append(torch.tensor([w_normalized], dtype=torch.float32))
        parts.append(torch.tensor([scale_normalized], dtype=torch.float32))

        # Read color ramp directly from Blender
        node_group = bpy.data.node_groups[self.node_group_name]
        color_ramp = node_group.nodes["TerrainColorRamp"]

        # Check if default (use counter)
        if "stack_call_count" not in color_ramp:
            num_colors = 0
            elements = []
        else:
            num_colors = color_ramp["stack_call_count"]
            # Get actual elements sorted by position
            elements = sorted(color_ramp.color_ramp.elements, key=lambda e: e.position)

        # Color sequence: hex values normalized
        color_sequence = torch.full((max_colors,), empty_value, dtype=torch.float32)

        # Only take unique colors (remove duplicates at same position)
        seen_positions = set()
        color_idx = 0

        for element in elements:
            pos_key = round(element.position, 3)  # Round to avoid floating point issues
            if pos_key not in seen_positions and color_idx < max_colors:
                seen_positions.add(pos_key)

                rgba = element.color
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                hex_value = (r << 16) | (g << 8) | b

                color_sequence[color_idx] = hex_value / 16777215.0
                color_idx += 1

        parts.append(color_sequence)

        # Metadata - use actual color count from counter
        parts.append(torch.tensor([num_colors / max_colors], dtype=torch.float32))

        return torch.cat(parts)

    @staticmethod
    def get_heightmap():
        # Extract heightmap after action
        bpy.context.view_layer.update()
        heightmap = BlenderTensorUtility.get_heightmap_by_name("TerrainPlane")
        return heightmap
    
    def __init__(self, node_group_name: str = "TerrainGenerator"):
        load_blend_single_color_ramp()
        self.node_group_name = node_group_name
        self._validate_setup()

    def _validate_setup(self):
        """Ensure the node group exists"""
        node_group = bpy.data.node_groups.get(self.node_group_name)
        if node_group is None:
            raise ValueError(f"Node group '{self.node_group_name}' not found")

        if node_group.nodes.get("NoiseTexture") is None:
            raise ValueError("NoiseTexture node not found")

        if node_group.nodes.get("TerrainColorRamp") is None:
            raise ValueError("TerrainColorRamp node not found")

    # ========================================================================
    # Noise Texture Control
    # ========================================================================

    def get_noise_w(self) -> float:
        """Get current W parameter of noise texture"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        return noise_node.inputs["W"].default_value

    def get_noise_scale(self) -> float:
        """Get current Scale parameter of noise texture"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        return noise_node.inputs["Scale"].default_value

    def set_noise_w(self, w: float):
        """Set W parameter of noise texture"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        noise_node.inputs["W"].default_value = w
        bpy.context.view_layer.update()

    def set_noise_scale(self, scale: float):
        """Set Scale parameter of noise texture"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        noise_node.inputs["Scale"].default_value = scale
        bpy.context.view_layer.update()

    def set_noise_params(self, w: float, scale: float):
        """Set both W and Scale parameters of noise texture"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        noise_node.inputs["W"].default_value = w
        noise_node.inputs["Scale"].default_value = scale
        bpy.context.view_layer.update()

    def is_noise_w_default(self, default_w: float = 50.0, tolerance: float = 0.01) -> bool:
        """Check if W parameter is at default value"""
        current_w = self.get_noise_w()
        return abs(current_w - default_w) < tolerance

    def is_noise_scale_default(self, default_scale: float = 5.0, tolerance: float = 0.01) -> bool:
        """Check if Scale parameter is at default value"""
        current_scale = self.get_noise_scale()
        return abs(current_scale - default_scale) < tolerance

    # ========================================================================
    # Color Ramp Control
    # ========================================================================

    def get_color_ramp_state(self, max_colors: int) -> Dict:
        """
        Get current state of the color ramp

        Args:
            max_colors: Maximum number of color slots

        Returns:
            Dictionary with filled_slots, empty_slots, colors, positions, etc.
        """
        node_group = bpy.data.node_groups[self.node_group_name]
        color_ramp = node_group.nodes["TerrainColorRamp"]
        elements = color_ramp.color_ramp.elements

        # Create evenly spaced target positions
        target_positions = [i / (max_colors - 1) for i in range(max_colors)]

        # Map actual colors to slots
        filled_slots = []
        colors = {}
        positions = {}
        tolerance = 0.05

        for slot_idx, target_pos in enumerate(target_positions):
            for element in elements:
                if abs(element.position - target_pos) < tolerance:
                    filled_slots.append(slot_idx)
                    colors[slot_idx] = tuple(element.color)
                    positions[slot_idx] = element.position
                    break

        empty_slots = [i for i in range(max_colors) if i not in filled_slots]

        # Check if default (Black at 0.0, White at 1.0)
        is_default = False
        if len(filled_slots) == 2:
            first_color = colors.get(0)
            last_color = colors.get(max_colors - 1)

            black = (0.0, 0.0, 0.0, 1.0)
            white = (1.0, 1.0, 1.0, 1.0)

            def colors_match(c1, c2, tol=0.01):
                return all(abs(a - b) < tol for a, b in zip(c1, c2))

            if (first_color and last_color and
                    colors_match(first_color, black) and
                    colors_match(last_color, white)):
                is_default = True

        return {
            'filled_slots': filled_slots,
            'empty_slots': empty_slots,
            'colors': colors,
            'positions': positions,
            'is_default': is_default,
            'num_filled': len(filled_slots),
            'num_empty': len(empty_slots)
        }

    def add_color_to_slot(self, slot_idx: int, color_rgba: Tuple[float, float, float, float], max_colors: int):
        """
        Add a color to a specific slot on the color ramp

        Args:
            slot_idx: Index of the slot (0 to max_colors-1)
            color_rgba: RGBA color tuple (values 0-1)
            max_colors: Maximum number of colors allowed
        """
        node_group = bpy.data.node_groups[self.node_group_name]
        color_ramp = node_group.nodes["TerrainColorRamp"]

        # Calculate position for this slot
        position = slot_idx / (max_colors - 1)

        # Add the element at the position
        element = color_ramp.color_ramp.elements.new(position)
        element.color = color_rgba

        bpy.context.view_layer.update()

    def stack_color_ramp(self, color_rgba: Tuple[float, float, float, float], max_colors: int):
        """
        this will put the color on the right most slot of the color ramp
        will also shift all other colors to be equadistant to each other
        """

        node_group = bpy.data.node_groups[self.node_group_name]
        color_ramp = node_group.nodes["TerrainColorRamp"]

        # Get/initialize call counter
        if "stack_call_count" not in color_ramp:
            color_ramp["stack_call_count"] = 0

        call_count = color_ramp["stack_call_count"]

        if call_count == 0:
            # First call - clear default and set single color
            color_ramp.color_ramp.elements[0].position = 0.0
            color_ramp.color_ramp.elements[0].color = color_rgba
            color_ramp.color_ramp.elements[1].position = 1.0
            color_ramp.color_ramp.elements[1].color = color_rgba
        elif call_count == 1:
            # Second call - replace RHS, keep LHS
            color_ramp.color_ramp.elements[1].position = 1.0
            color_ramp.color_ramp.elements[1].color = color_rgba
        else:
            # Subsequent calls - redistribute existing colors and add new one
            existing_colors = [(el.position, tuple(el.color)) for el in color_ramp.color_ramp.elements]

            while len(color_ramp.color_ramp.elements) > 2:
                color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])

            num_colors = len(existing_colors) + 1

            color_ramp.color_ramp.elements[0].position = 0.0
            color_ramp.color_ramp.elements[0].color = existing_colors[0][1]

            for i in range(1, len(existing_colors)):
                pos = i / (num_colors - 1)
                if i == 1:
                    color_ramp.color_ramp.elements[1].position = pos
                    color_ramp.color_ramp.elements[1].color = existing_colors[i][1]
                else:
                    el = color_ramp.color_ramp.elements.new(pos)
                    el.color = existing_colors[i][1]

            el = color_ramp.color_ramp.elements.new(1.0)
            el.color = color_rgba

        # Increment counter
        color_ramp["stack_call_count"] = call_count + 1

        bpy.context.view_layer.update()

    def clear_color_ramp(self):
        """Remove all color elements except the required minimum (2)"""
        node_group = bpy.data.node_groups[self.node_group_name]
        color_ramp = node_group.nodes["TerrainColorRamp"]

        while len(color_ramp.color_ramp.elements) > 2:
            color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])

        bpy.context.view_layer.update()

    def set_color_ramp(self, colors_dict: Dict[int, Tuple[float, float, float, float]], max_colors: int):
        """
        Set colors on the color ramp using slot-based positioning

        Args:
            colors_dict: Dictionary mapping slot_idx -> color_rgba
            max_colors: Maximum number of color slots
        """
        node_group = bpy.data.node_groups[self.node_group_name]
        color_ramp = node_group.nodes["TerrainColorRamp"]

        # Clear existing elements
        while len(color_ramp.color_ramp.elements) > 2:
            color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])

        elements = color_ramp.color_ramp.elements
        slot_positions = {i: i / (max_colors - 1) for i in range(max_colors)}
        sorted_slots = sorted(colors_dict.keys())

        if len(sorted_slots) == 0:
            raise ValueError("Must provide at least one color")

        # Set first color
        first_slot = sorted_slots[0]
        first_position = slot_positions[first_slot]
        first_color = colors_dict[first_slot]

        elements[0].position = first_position
        elements[0].color = first_color

        # Set remaining colors
        if len(sorted_slots) == 1:
            elements[1].position = first_position
            elements[1].color = first_color
        else:
            second_slot = sorted_slots[1]
            second_position = slot_positions[second_slot]
            second_color = colors_dict[second_slot]

            elements[1].position = second_position
            elements[1].color = second_color

            for slot_idx in sorted_slots[2:]:
                position = slot_positions[slot_idx]
                color = colors_dict[slot_idx]
                element = color_ramp.color_ramp.elements.new(position)
                element.color = color

        bpy.context.view_layer.update()

    def step_w(self,step_val):
        # this will incriment the value of w the specified amount.
        """Increment the W parameter by the specified amount"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        current_w = noise_node.inputs["W"].default_value
        noise_node.inputs["W"].default_value = current_w + step_val
        bpy.context.view_layer.update()

    def step_noise_scale(self,step_val):
        # this will incriment the value of w the specified amount.
        """Increment the W parameter by the specified amount"""
        node_group = bpy.data.node_groups[self.node_group_name]
        noise_node = node_group.nodes["NoiseTexture"]
        current_w = noise_node.inputs["Scale"].default_value
        noise_node.inputs["Scale"].default_value = current_w + step_val
        bpy.context.view_layer.update()
"""
Terrain Action Space Registration
Defines the action space by registering API calls to the action registry
"""
from typing import List, Dict, Any


# ============================================================================
# Pre-fill Replay Buffer
# ============================================================================
def prefill_replay_buffer():
    pass

import numpy as np


import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum




class StepWExperimentDefinition:
    """Static configuration for step-based W/Scale experiment"""

    DESCRIPTION = "Step w and scale in the noise at any time"
    STEP_W = [2]
    STEP_SCALE = [0.1]
    MAX_COLORS = 32

    # Color palette (RGBA tuples) - 32 terrain colors
    COLOR_PALETTE = [
        # Deep water (0-3)
        (0.0, 0.0, 0.2, 1.0),  # 0: Deep ocean blue
        (0.0, 0.1, 0.3, 1.0),  # 1: Deep blue
        (0.0, 0.2, 0.4, 1.0),  # 2: Ocean blue
        (0.1, 0.3, 0.5, 1.0),  # 3: Medium blue

        # Shallow water (4-7)
        (0.2, 0.4, 0.6, 1.0),  # 4: Light blue
        (0.3, 0.5, 0.7, 1.0),  # 5: Shallow water
        (0.4, 0.6, 0.7, 1.0),  # 6: Very shallow
        (0.5, 0.7, 0.8, 1.0),  # 7: Beach water

        # Beach/sand (8-11)
        (0.9, 0.9, 0.7, 1.0),  # 8: Light sand
        (0.9, 0.85, 0.6, 1.0),  # 9: Sand
        (0.85, 0.8, 0.55, 1.0),  # 10: Dark sand
        (0.8, 0.75, 0.5, 1.0),  # 11: Wet sand

        # Grassland (12-15)
        (0.4, 0.6, 0.2, 1.0),  # 12: Light grass
        (0.3, 0.5, 0.2, 1.0),  # 13: Grass
        (0.25, 0.45, 0.15, 1.0),  # 14: Dark grass
        (0.2, 0.4, 0.1, 1.0),  # 15: Forest floor

        # Forest (16-19)
        (0.15, 0.35, 0.08, 1.0),  # 16: Dense forest
        (0.2, 0.3, 0.1, 1.0),  # 17: Dark green
        (0.15, 0.25, 0.08, 1.0),  # 18: Deep forest
        (0.1, 0.2, 0.05, 1.0),  # 19: Very dark forest

        # Hills/dirt (20-23)
        (0.6, 0.5, 0.3, 1.0),  # 20: Light brown
        (0.6, 0.4, 0.2, 1.0),  # 21: Brown
        (0.5, 0.35, 0.2, 1.0),  # 22: Dark brown
        (0.45, 0.3, 0.15, 1.0),  # 23: Dirt

        # Rocky/mountain (24-27)
        (0.5, 0.5, 0.5, 1.0),  # 24: Light gray rock
        (0.4, 0.4, 0.4, 1.0),  # 25: Gray rock
        (0.3, 0.3, 0.3, 1.0),  # 26: Dark rock
        (0.25, 0.25, 0.25, 1.0),  # 27: Very dark rock

        # Snow/ice (28-31)
        (0.85, 0.85, 0.9, 1.0),  # 28: Light snow
        (0.9, 0.9, 0.95, 1.0),  # 29: Snow
        (0.95, 0.95, 0.98, 1.0),  # 30: Fresh snow
        (1.0, 1.0, 1.0, 1.0),  # 31: Pure white snow
    ]

    # Pad to 32 colors if needed
    while len(COLOR_PALETTE) < MAX_COLORS:
        COLOR_PALETTE.append((0.5, 0.5, 0.5, 1.0))

    VALID_COLOR_INDICES = list(range(len(COLOR_PALETTE)))

    # Action definitions with direct function pointers
    ACTIONS = {
        'step_w': {
            'valid_values': STEP_W,
            'description': 'Step noise W parameter',
            'execute': lambda blender_api, value: blender_api.step_w(value)
        },
        'step_scale': {
            'valid_values': STEP_SCALE,
            'description': 'Step noise Scale parameter',
            'execute': lambda blender_api, value: blender_api.step_noise_scale(value)
        },
        'add_color': {
            'valid_values': VALID_COLOR_INDICES,
            'description': 'Add color to next available slot',
            'execute': lambda blender_api, value, slot_idx: blender_api.stack_color_ramp(
                color_rgba=StepWExperimentDefinition.COLOR_PALETTE[value],
                max_colors=StepWExperimentDefinition.MAX_COLORS
            )
        },
        'stop': {
            'valid_values': [0],
            'description': 'Terminate trajectory',
            'execute': None
        }
    }

    @classmethod
    def get_total_actions(cls) -> int:
        """Total number of possible actions across all types"""
        return sum(len(info['valid_values']) for info in cls.ACTIONS.values())

    @classmethod
    def get_action_offset(cls, action_name: str) -> int:
        """Starting index for this action type in flat action space"""
        offset = 0
        for name, info in cls.ACTIONS.items():
            if name == action_name:
                return offset
            offset += len(info['valid_values'])
        raise ValueError(f"Action {action_name} not found")

    @classmethod
    def decode_action(cls, action_idx: int) -> Tuple[str, int]:
        """Convert flat action index to (action_name, value_index)"""
        current_offset = 0
        for name, info in cls.ACTIONS.items():
            num_values = len(info['valid_values'])
            if action_idx < current_offset + num_values:
                value_idx = action_idx - current_offset
                return name, value_idx
            current_offset += num_values
        raise ValueError(f"Invalid action index: {action_idx}")

    @classmethod
    def encode_action(cls, action_name: str, value_idx: int) -> int:
        """Convert (action_name, value_index) to flat action index"""
        offset = cls.get_action_offset(action_name)
        return offset + value_idx

    @classmethod
    def base_environment_state(cls, blender_api) -> 'StepWExperimentDefinition.State':
        """
        Query Blender to get the actual starting state of the environment.
        Every state must be reachable from this base state.
        """
        current_w = blender_api.get_noise_w()
        current_scale = blender_api.get_noise_scale()
        color_ramp_state = blender_api.get_color_ramp_state(max_colors=cls.MAX_COLORS)

        return cls.State(
            noise_w=current_w,
            noise_scale=current_scale,
            color_assignments={},
            num_colors_assigned=color_ramp_state['num_filled'],
            is_terminal=False
        )

    # ========================================================================
    # State Class - Pure data representation
    # ========================================================================

    class State(BaseModel):
        """
        Pure data - state representation.

        Main responsibilities:
        1. to_tensor() - Convert state to tensor for model input
        2. to_action_mask() - Generate valid action mask for model output
        """

        noise_w: float
        noise_scale: float
        color_assignments: Dict[int, int] = Field(default_factory=dict)
        num_colors_assigned: int = 0
        is_terminal: bool = False

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def to_tensor(self) -> torch.Tensor:
            """
            Convert state to tensor for model INPUT.

            Tensor structure:
            - [1]: noise_w (normalized 0-1)
            - [1]: noise_scale (normalized 0-1)
            - [MAX_COLORS]: slot occupancy (binary)
            - [1]: num_colors_assigned (normalized)
            - [1]: is_terminal (binary)

            Total: MAX_COLORS + 4
            """
            parts = []

            # Normalize W and Scale to [0, 1] range
            # Assuming reasonable bounds: W in [0, 100], Scale in [0, 50]
            parts.append(torch.tensor([self.noise_w / 100.0], dtype=torch.float32))
            parts.append(torch.tensor([self.noise_scale / 50.0], dtype=torch.float32))

            # Color slot occupancy
            slot_occupancy = torch.zeros(StepWExperimentDefinition.MAX_COLORS)
            for slot_idx in self.color_assignments.keys():
                slot_occupancy[slot_idx] = 1
            parts.append(slot_occupancy)

            # Metadata
            parts.append(torch.tensor([
                self.num_colors_assigned / StepWExperimentDefinition.MAX_COLORS,
                1.0 if self.is_terminal else 0.0
            ], dtype=torch.float32))

            return torch.cat(parts)

        def to_action_mask(self) -> torch.Tensor:
            """
            Generate valid action mask for model OUTPUT.

            Returns boolean tensor of shape [total_actions] where True = valid action.
            """
            mask = torch.zeros(StepWExperimentDefinition.get_total_actions(), dtype=torch.bool)

            if not self.is_terminal:
                # step_w: always available
                step_w_offset = StepWExperimentDefinition.get_action_offset('step_w')
                mask[step_w_offset:step_w_offset + len(StepWExperimentDefinition.STEP_W)] = True

                # step_scale: always available
                step_scale_offset = StepWExperimentDefinition.get_action_offset('step_scale')
                mask[step_scale_offset:step_scale_offset + len(StepWExperimentDefinition.STEP_SCALE)] = True

                # add_color: available if slots remaining

                if self.num_colors_assigned < StepWExperimentDefinition.MAX_COLORS:
                    color_offset = StepWExperimentDefinition.get_action_offset('add_color')
                    mask[color_offset:color_offset + len(StepWExperimentDefinition.VALID_COLOR_INDICES)] = True

                # stop: available if at least one color
                if self.num_colors_assigned > 0:
                    stop_offset = StepWExperimentDefinition.get_action_offset('stop')
                    mask[stop_offset] = True

            return mask

        @classmethod
        def get_state_dim(cls) -> int:
            """Dimension of state tensor"""
            return StepWExperimentDefinition.MAX_COLORS + 4

        @classmethod
        def get_action_dim(cls) -> int:
            """Dimension of action space"""
            return StepWExperimentDefinition.get_total_actions()

    # ========================================================================
    # Action Class - Pure data representation
    # ========================================================================

    class Action(BaseModel):
        """Pure data - action representation"""

        action_name: str
        value_idx: int

        @property
        def value(self):
            """Get the actual value from the action config"""
            return StepWExperimentDefinition.ACTIONS[self.action_name]['valid_values'][self.value_idx]

        def to_flat_index(self) -> int:
            """Convert to flat action index for model OUTPUT"""
            return StepWExperimentDefinition.encode_action(self.action_name, self.value_idx)

        @classmethod
        def from_flat_index(cls, action_idx: int) -> 'StepWExperimentDefinition.Action':
            """Create Action from flat action index"""
            action_name, value_idx = StepWExperimentDefinition.decode_action(action_idx)
            return cls(action_name=action_name, value_idx=value_idx)

    # ========================================================================
    # Trajectory Class - Orchestrator
    # ========================================================================

    class Trajectory:
        """
        Orchestrates interactions between State, Action, and Blender.
        Maintains the sequence of states and actions.
        """

        def __init__(self, blender_api, initial_state: 'StepWExperimentDefinition.State' = None):
            self.blender_api = blender_api

            # Initialize from Blender if no initial state provided
            if initial_state is None:
                initial_state = StepWExperimentDefinition.base_environment_state(blender_api)

            self.states = [initial_state]
            self.actions = []
            self.rewards = []

        def step(self, action: 'StepWExperimentDefinition.Action', reward: float = 0.0):
            """
            Take a step in the trajectory:
            1. Transition state (pure)
            2. Execute on Blender (side effect)
            3. Record state, action, reward
            """
            current_state = self.states[-1]

            # State transition logic
            new_data = current_state.model_dump()

            if action.action_name == 'step_w':
                new_data['noise_w'] = current_state.noise_w + action.value

            elif action.action_name == 'step_scale':
                new_data['noise_scale'] = current_state.noise_scale + action.value

            elif action.action_name == 'add_color':
                next_slot = current_state.num_colors_assigned
                new_data['color_assignments'][next_slot] = action.value
                new_data['num_colors_assigned'] += 1

            elif action.action_name == 'stop':
                new_data['is_terminal'] = True

            new_state = StepWExperimentDefinition.State(**new_data)

            # Execute on Blender
            action_info = StepWExperimentDefinition.ACTIONS[action.action_name]
            execute_fn = action_info['execute']

            if execute_fn is not None:
                if action.action_name == 'add_color':
                    execute_fn(self.blender_api, action.value, current_state.num_colors_assigned)
                else:
                    execute_fn(self.blender_api, action.value)

            # Record
            self.states.append(new_state)
            self.actions.append(action)
            self.rewards.append(reward)

        def step_from_flat_action(self, action_idx: int, reward: float = 0.0):
            """Convenience method: step using flat action index"""
            action = StepWExperimentDefinition.Action.from_flat_index(action_idx)
            self.step(action, reward)

        @property
        def current_state(self) -> 'StepWExperimentDefinition.State':
            """Get the current state"""
            return self.states[-1]

        def get_state_tensor(self) -> torch.Tensor:
            """Get current state as tensor"""
            return self.current_state.to_tensor()

        def get_action_mask(self) -> torch.Tensor:
            """Get valid actions for current state"""
            return self.current_state.to_action_mask()

        def is_terminal(self) -> bool:
            """Check if trajectory is complete"""
            return self.current_state.is_terminal

        def __len__(self) -> int:
            """Length of trajectory (number of actions taken)"""
            return len(self.actions)


from pydantic import BaseModel, Field
from typing import List, Tuple, Callable, Optional, Dict
import torch


class ActionDefinition(BaseModel):
    """Defines a single action with its properties"""
    action_key: str
    value: float | int
    flat_index: int
    execute_fn: Optional[Callable] = Field(default=None, exclude=True)  # Not serialized

    class Config:
        arbitrary_types_allowed = True

    def to_one_hot(self, action_dim: int) -> torch.Tensor:
        """Convert this action to a one-hot tensor"""
        one_hot = torch.zeros(action_dim)
        one_hot[self.flat_index] = 1.0
        return one_hot

    def execute(self, blender_api, **kwargs):
        """Execute this action"""
        if self.execute_fn is None:
            return  # stop action
        self.execute_fn(blender_api, self.value, **kwargs)


class v2StepWEnv(BaseModel):
    """Environment definition using Pydantic"""

    description: str = "Step w and scale in the noise at any time"
    step_w_values: List[float] = Field(default=[2.0])
    step_scale_values: List[float] = Field(default=[0.1])
    max_colors: int = 32

    # Color palette
    color_palette: List[Tuple[float, float, float, float]] = Field(default_factory=lambda: [
        # Deep water (0-3)
        (0.0, 0.0, 0.2, 1.0),  # 0: Deep ocean blue
        (0.0, 0.1, 0.3, 1.0),  # 1: Deep blue
        (0.0, 0.2, 0.4, 1.0),  # 2: Ocean blue
        (0.1, 0.3, 0.5, 1.0),  # 3: Medium blue
        # Shallow water (4-7)
        (0.2, 0.4, 0.6, 1.0),  # 4: Light blue
        (0.3, 0.5, 0.7, 1.0),  # 5: Shallow water
        (0.4, 0.6, 0.7, 1.0),  # 6: Very shallow
        (0.5, 0.7, 0.8, 1.0),  # 7: Beach water
        # Beach/sand (8-11)
        (0.9, 0.9, 0.7, 1.0),  # 8: Light sand
        (0.9, 0.85, 0.6, 1.0),  # 9: Sand
        (0.85, 0.8, 0.55, 1.0),  # 10: Dark sand
        (0.8, 0.75, 0.5, 1.0),  # 11: Wet sand
        # Grassland (12-15)
        (0.4, 0.6, 0.2, 1.0),  # 12: Light grass
        (0.3, 0.5, 0.2, 1.0),  # 13: Grass
        (0.25, 0.45, 0.15, 1.0),  # 14: Dark grass
        (0.2, 0.4, 0.1, 1.0),  # 15: Forest floor
        # Forest (16-19)
        (0.15, 0.35, 0.08, 1.0),  # 16: Dense forest
        (0.2, 0.3, 0.1, 1.0),  # 17: Dark green
        (0.15, 0.25, 0.08, 1.0),  # 18: Deep forest
        (0.1, 0.2, 0.05, 1.0),  # 19: Very dark forest
        # Hills/dirt (20-23)
        (0.6, 0.5, 0.3, 1.0),  # 20: Light brown
        (0.6, 0.4, 0.2, 1.0),  # 21: Brown
        (0.5, 0.35, 0.2, 1.0),  # 22: Dark brown
        (0.45, 0.3, 0.15, 1.0),  # 23: Dirt
        # Rocky/mountain (24-27)
        (0.5, 0.5, 0.5, 1.0),  # 24: Light gray rock
        (0.4, 0.4, 0.4, 1.0),  # 25: Gray rock
        (0.3, 0.3, 0.3, 1.0),  # 26: Dark rock
        (0.25, 0.25, 0.25, 1.0),  # 27: Very dark rock
        # Snow/ice (28-31)
        (0.85, 0.85, 0.9, 1.0),  # 28: Light snow
        (0.9, 0.9, 0.95, 1.0),  # 29: Snow
        (0.95, 0.95, 0.98, 1.0),  # 30: Fresh snow
        (1.0, 1.0, 1.0, 1.0),  # 31: Pure white snow
    ])

    # This will be populated in __init__
    tensor_to_action: Dict[int, ActionDefinition] = Field(default_factory=dict)
    n_actions: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        # Pad color palette to max_colors
        while len(self.color_palette) < self.max_colors:
            self.color_palette.append((0.5, 0.5, 0.5, 1.0))

        # Build tensor_to_action mapping
        self._build_tensor_to_action()

    def _build_tensor_to_action(self):
        """Build the mapping from flat index to ActionDefinition"""
        idx = 0

        # Import here to avoid circular dependency issues
        # You'll need to inject these functions or make them methods

        # Step W actions
        for value in self.step_w_values:
            self.tensor_to_action[idx] = ActionDefinition(
                action_key='step_w',
                value=value,
                flat_index=idx,
                execute_fn=lambda api, v: api.step_w(v)
            )
            idx += 1

        # Step Scale actions
        for value in self.step_scale_values:
            self.tensor_to_action[idx] = ActionDefinition(
                action_key='step_scale',
                value=value,
                flat_index=idx,
                execute_fn=lambda api, v: api.step_noise_scale(v)
            )
            idx += 1

        # Add Color actions
        for color_idx in range(len(self.color_palette)):
            palette_idx = color_idx
            self.tensor_to_action[idx] = ActionDefinition(
                action_key='add_color',
                value=color_idx,
                flat_index=idx,
                execute_fn=lambda api, v, **kw: api.stack_color_ramp(
                    color_rgba=self.color_palette[v],
                    max_colors=self.max_colors
                )
            )
            idx += 1

        # Stop action
        self.tensor_to_action[idx] = ActionDefinition(
            action_key='stop',
            value=0,
            flat_index=idx,
            execute_fn=None
        )
        idx += 1

        self.n_actions = idx

    def one_hot_to_action(self, one_hot_tensor: torch.Tensor) -> ActionDefinition:
        """Convert one-hot tensor to ActionDefinition"""
        idx = torch.argmax(one_hot_tensor).item()
        return self.tensor_to_action[idx]

    def action_to_one_hot(self, action_key: str, value: float | int) -> torch.Tensor:
        """Convert action key and value to one-hot tensor"""
        for idx, action_def in self.tensor_to_action.items():
            if action_def.action_key == action_key and action_def.value == value:
                return action_def.to_one_hot(self.n_actions)
        raise ValueError(f"Action {action_key}={value} not found")

    def execute_one_hot_action(self, blender_api, one_hot_tensor: torch.Tensor, **kwargs):
        """Execute action from one-hot tensor"""
        action_def = self.one_hot_to_action(one_hot_tensor)
        action_def.execute(blender_api, **kwargs)

