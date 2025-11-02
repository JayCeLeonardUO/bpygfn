import bpy
from blender_setup_utils import set_colors_in_ramp
from blender_setup_utils import load_blend

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
    map_range.inputs["From Min"].default_value = -3.0
    map_range.inputs["From Max"].default_value = -1.0
    map_range.inputs["To Min"].default_value = -3.0
    map_range.inputs["To Max"].default_value = -1.0

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


def get_valid_actions(max_colors: int, available_scales: List[float],available_w = [1,2,3]) -> dict:
    """
    Get all valid actions given the current state

    Args:
        max_colors: Maximum number of colors allowed on the ramp
        available_scales: List of available scale values

    Returns:
        Dictionary with:
            - can_set_noise: Whether noise parameters can be set
            - available_slots: List of empty slot indices
            - noise_actions: List of (w, scale) tuples if can_set_noise
            - color_actions: List of slot indices where colors can be added
    """
    from blender_setup_utils import color_df

    state = get_color_ramp_state(max_colors)

    # If default state, can set noise params
    can_set_noise = state['is_default']

    # Generate noise actions (W values from 0-100, scales from available_scales)
    noise_actions = []
    if can_set_noise:
        w_values = [10.0, 25.0, 50.0, 75.0, 100.0]  # Example W values
        for w in w_values:
            for scale in available_scales:
                noise_actions.append((w, scale))

    # Color actions are available empty slots
    color_actions = state['empty_slots']

    # Get available colors
    available_colors = color_df['name'].tolist()

    return {
        'can_set_noise': can_set_noise,
        'available_slots': state['empty_slots'],
        'noise_actions': noise_actions,
        'color_actions': color_actions,
        'available_colors': available_colors,
        'current_state': state
    }


def set_colors_in_ramp_with_slots(color_ramp_node, colors_dict: dict, max_colors: int):
    """
    Set colors on a color ramp node using slot-based positioning

    Args:
        color_ramp_node: The color ramp node
        colors_dict: Dictionary mapping slot_idx -> color_name
                    e.g., {0: "Blue", 2: "Green", 4: "White"}
        max_colors: Maximum number of color slots allowed

    Example:
        color_ramp = node_group.nodes["TerrainColorRamp"]
        # Only fill specific slots
        set_colors_in_ramp_with_slots(color_ramp, {0: "Blue", 3: "White"}, max_colors=4)
        # This creates: Blue at 0.0, empty at 0.33, empty at 0.66, White at 1.0
    """

    from blender_setup_utils import color_df

    # Create name -> color mapping from global color_df
    name_to_color = {
        row['name']: (row['red'], row['green'], row['blue'], row['alpha'])
        for _, row in color_df.iterrows()
    }

    # Clear all existing elements except the last 2 (Blender requires at least 2)
    while len(color_ramp_node.color_ramp.elements) > 2:
        color_ramp_node.color_ramp.elements.remove(color_ramp_node.color_ramp.elements[0])

    # Now we have exactly 2 elements - we'll reuse or remove one
    elements = color_ramp_node.color_ramp.elements

    # Calculate positions for each slot
    slot_positions = {i: i / (max_colors - 1) for i in range(max_colors)}

    # Sort slots to set them in order
    sorted_slots = sorted(colors_dict.keys())

    if len(sorted_slots) == 0:
        raise ValueError("Must provide at least one color")

    # Set first color (reuse first element)
    first_slot = sorted_slots[0]
    if first_slot < 0 or first_slot >= max_colors:
        raise ValueError(f"Slot index {first_slot} out of range [0, {max_colors - 1}]")

    first_color_name = colors_dict[first_slot]
    if first_color_name not in name_to_color:
        available = list(name_to_color.keys())
        raise ValueError(f"Color '{first_color_name}' not found. Available: {available}")

    first_position = slot_positions[first_slot]
    first_color = name_to_color[first_color_name]

    elements[0].position = first_position
    elements[0].color = first_color
    print(f"  Slot {first_slot} (pos {first_position:.3f}): {first_color_name}")

    # If we only have one color, set the second element to the same
    if len(sorted_slots) == 1:
        elements[1].position = first_position
        elements[1].color = first_color
    else:
        # Set second color (reuse second element)
        second_slot = sorted_slots[1]
        if second_slot < 0 or second_slot >= max_colors:
            raise ValueError(f"Slot index {second_slot} out of range [0, {max_colors - 1}]")

        second_color_name = colors_dict[second_slot]
        if second_color_name not in name_to_color:
            available = list(name_to_color.keys())
            raise ValueError(f"Color '{second_color_name}' not found. Available: {available}")

        second_position = slot_positions[second_slot]
        second_color = name_to_color[second_color_name]

        elements[1].position = second_position
        elements[1].color = second_color
        print(f"  Slot {second_slot} (pos {second_position:.3f}): {second_color_name}")

        # Add remaining colors (if any)
        for slot_idx in sorted_slots[2:]:
            if slot_idx < 0 or slot_idx >= max_colors:
                raise ValueError(f"Slot index {slot_idx} out of range [0, {max_colors - 1}]")

            color_name = colors_dict[slot_idx]
            if color_name not in name_to_color:
                available = list(name_to_color.keys())
                raise ValueError(f"Color '{color_name}' not found. Available: {available}")

            position = slot_positions[slot_idx]
            color = name_to_color[color_name]

            # Create new element
            element = color_ramp_node.color_ramp.elements.new(position)
            element.color = color

            print(f"  Slot {slot_idx} (pos {position:.3f}): {color_name}")

def build_color_ramp_incrementally(color_sequence: list, max_colors: int):
    """
    Build a color ramp incrementally, filling slots from start to end

    Args:
        color_sequence: List of color names to add in order
        max_colors: Maximum number of slots

    Returns:
        Dictionary mapping slot_idx -> color_name

    Example:
        colors = build_color_ramp_incrementally(["Blue", "Green", "White"], max_colors=5)
        # Returns: {0: "Blue", 2: "Green", 4: "White"}
        # Evenly spaces colors across available slots
    """
    if len(color_sequence) > max_colors:
        raise ValueError(f"Too many colors ({len(color_sequence)}) for max_colors ({max_colors})")

    colors_dict = {}

    if len(color_sequence) == 1:
        # Single color goes in first slot
        colors_dict[0] = color_sequence[0]
    elif len(color_sequence) == 2:
        # Two colors: first and last slot
        colors_dict[0] = color_sequence[0]
        colors_dict[max_colors - 1] = color_sequence[1]
    else:
        # Multiple colors: distribute evenly
        for i, color in enumerate(color_sequence):
            slot_idx = int(i * (max_colors - 1) / (len(color_sequence) - 1))
            colors_dict[slot_idx] = color

    return colors_dict


from action_utils.action_regestry_util import *


# Global registry
registry = ActionRegistry()


def is_noise_w_default() -> bool:
    """
    Check if the noise W parameter is still at default value

    Returns:
        True if W is at default (50.0), False otherwise
    """
    import bpy

    node_group = bpy.data.node_groups.get("TerrainGenerator")
    if node_group is None:
        raise ValueError("TerrainGenerator node group not found")

    noise_node = node_group.nodes.get("NoiseTexture")
    if noise_node is None:
        raise ValueError("NoiseTexture node not found")

    default_w = 50.0
    current_w = noise_node.inputs["W"].default_value

    # Use small tolerance for float comparison
    return abs(current_w - default_w) < 0.01


def is_noise_scale_default() -> bool:
    """
    Check if the noise Scale parameter is still at default value

    Returns:
        True if Scale is at default (5.0), False otherwise
    """
    import bpy

    node_group = bpy.data.node_groups.get("TerrainGenerator")
    if node_group is None:
        raise ValueError("TerrainGenerator node group not found")

    noise_node = node_group.nodes.get("NoiseTexture")
    if noise_node is None:
        raise ValueError("NoiseTexture node not found")

    default_scale = 5.0
    current_scale = noise_node.inputs["Scale"].default_value

    # Use small tolerance for float comparison
    return abs(current_scale - default_scale) < 0.01

# ============================================================================
# Define Action Groups with Encoding Schemes
# ============================================================================

registry.register_group(ActionGroup(
    name='noise_params',
    validator=lambda: is_noise_w_default() or is_noise_scale_default(),
    encoding_scheme=EncodingScheme.ONE_HOT,
    description='Noise texture parameters (W and Scale)'
))

registry.register_group(ActionGroup(
    name='color_ramp',
    validator=lambda: True,
    encoding_scheme=EncodingScheme.FACTORIZED,  # Use factorized encoding!
    description='Color ramp modifications'
))


# ============================================================================
# Validators
# ============================================================================

def can_set_w() -> bool:
    return is_noise_w_default()


def can_set_scale() -> bool:
    return is_noise_scale_default()


def can_add_color() -> List[dict]:
    max_colors = registry.max_colors if hasattr(registry, 'max_colors') else 5
    state = get_color_ramp_state(max_colors)
    empty_slots = state['empty_slots']

    from blender_setup_utils import color_df
    all_colors = color_df['name'].tolist()

    valid_combinations = []
    for slot_idx in empty_slots:
        for color_name in all_colors:
            valid_combinations.append({'slot_idx': slot_idx, 'color_name': color_name})

    return valid_combinations


# ============================================================================
# Decorated Actions
# ============================================================================

@registry.add_actions(
    values=[10.0, 25.0, 50.0, 75.0, 100.0],
    action_type='set_w',
    group='noise_params',
    validator=can_set_w
)
def set_w(w: float):
    import bpy
    node_group = bpy.data.node_groups["TerrainGenerator"]
    noise_node = node_group.nodes["NoiseTexture"]
    current_scale = noise_node.inputs["Scale"].default_value
    set_noise_params(w, current_scale)
    print(f"  Executed: Set W={w}")
    return {'type': 'set_w', 'w': w}


@registry.add_actions(
    values=[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0],
    action_type='set_scale',
    group='noise_params',
    validator=can_set_scale
)
def set_scale(scale: float):
    import bpy
    node_group = bpy.data.node_groups["TerrainGenerator"]
    noise_node = node_group.nodes["NoiseTexture"]
    current_w = noise_node.inputs["W"].default_value
    set_noise_params(current_w, scale)
    print(f"  Executed: Set Scale={scale}")
    return {'type': 'set_scale', 'scale': scale}


def register_color_actions(max_colors: int):
    """
    Register color actions - encoding scheme from 'color_ramp' group
    Since the group has FACTORIZED encoding, this will be two-hot!
    """
    from blender_setup_utils import color_df

    registry.max_colors = max_colors
    all_colors = color_df['name'].tolist()

    @registry.add_parameterized_actions(
        params={
            'slot_idx': list(range(max_colors)),
            'color_name': all_colors
        },
        action_type='add_color',
        group='color_ramp',  # Uses FACTORIZED encoding from group!
        validator=can_add_color
    )
    def add_color(slot_idx: int, color_name: str):
        add_color_to_slot(slot_idx, color_name, max_colors)
        print(f"  Executed: Add '{color_name}' to slot {slot_idx}")
        return {'type': 'add_color', 'slot_idx': slot_idx, 'color_name': color_name}

    print(f"\n✓ Action registry initialized")
    summary = registry.get_action_space_summary()
    print(f"  Total dimensions: {summary['total_dimensions']}")
    for group_name, group_info in summary['groups'].items():
        print(f"  Group '{group_name}': {group_info['total']} dimensions")
        for action_type_info in group_info['action_types']:
            encoding = action_type_info['encoding']
            print(f"    - {action_type_info['type']}: {action_type_info['count']} [{encoding}]")


def sample_random_trajectory(trajectory_len: int, max_colors: int = 5) -> dict:
    """
    Sample a random trajectory using the global action registry

    Args:
        trajectory_len: Maximum number of actions to take
        max_colors: Maximum number of color slots

    Returns:
        Dictionary containing trajectory information
    """

    from blender_setup_utils import BlenderTensorUtility
    import bpy

    # Load fresh template
    load_blend_single_color_ramp()

    # Register color actions if not already done
    if 'add_color' not in registry.action_type_info:
        register_color_actions(max_colors)

    # Storage for trajectory
    actions = []
    action_tensors = []
    heightmaps = []
    states = []
    action_masks = []

    print("\n" + "=" * 70)
    print("SAMPLING RANDOM TRAJECTORY")
    print("=" * 70)
    print(f"Trajectory length: {trajectory_len}")
    print(f"Max colors: {max_colors}")

    for step in range(trajectory_len):
        print(f"\n--- Step {step + 1} ---")

        # Get valid actions mask
        mask = registry.get_action_mask()
        action_masks.append(mask.clone())

        num_valid = mask.sum().item()
        print(f"Valid dimensions: {num_valid}/{registry.total_actions}")

        if num_valid == 0:
            print("No valid actions - trajectory complete")
            break

        # Sample action from mask
        action_tensor = registry.sample_from(mask)

        # Apply action
        action = registry[action_tensor]()
        print(f"Action: {action}")

        # Extract heightmap after action
        bpy.context.view_layer.update()
        heightmap = BlenderTensorUtility.get_heightmap_by_name("TerrainPlane")

        # Get current state
        current_state = get_color_ramp_state(max_colors)

        # Store
        actions.append(action)
        action_tensors.append(action_tensor)
        heightmaps.append(heightmap)
        states.append(current_state)

        print(f"Height range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")
        print(f"Filled slots: {current_state['filled_slots']}")

    result = {
        'actions': actions,
        'action_tensors': torch.stack(action_tensors) if action_tensors else torch.tensor([]),
        'heightmaps': torch.stack(heightmaps) if heightmaps else torch.tensor([]),
        'states': states,
        'action_masks': torch.stack(action_masks) if action_masks else torch.tensor([]),
        'action_space_summary': registry.get_action_space_summary(),
        'num_actions': len(actions),
        'max_colors': max_colors
    }

    print("\n" + "=" * 70)
    print(f"TRAJECTORY COMPLETE: {len(actions)} actions taken")
    print("=" * 70)

    return result

