import pandas as pd
import pytest
import json
import bpy
import torch
import numpy as np
from typing import Optional
import pytest# Load the JSON file
from pathlib import Path
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
JSON_DIR = SCRIPT_DIR / 'json'

# Load resources using relative path
with open(JSON_DIR / 'blender_config.json', 'r') as f:
    resources = json.load(f)


# Create DataFrames from the tables
color_df = pd.DataFrame(resources['color_palette'])
available_scales = resources['available_scales']

def test_load_config():
    print(color_df)
    print(f"\nAvailable scales: {available_scales}")
    assert len(color_df) == 32
    assert len(available_scales) == 7
    pass


from typing import Tuple, List
from pydantic import BaseModel, field_validator
import pytest


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

    @staticmethod
    def get_heightmap_by_name(object_name: str) -> Optional[torch.Tensor]:
        """
        Get heightmap from Blender object by name

        Args:
            object_name: Name of the Blender object

        Returns:
            2D tensor of height values, or None if object not found or extraction fails
        """
        # Find object by name
        plane = bpy.data.objects.get(object_name)

        if plane is None:
            available = [obj.name for obj in bpy.data.objects if obj.type == 'MESH']
            print(f"❌ Object '{object_name}' not found. Available mesh objects: {available}")
            return None

        if plane.type != 'MESH':
            print(f"❌ Object '{object_name}' is not a mesh (type: {plane.type})")
            return None

        return BlenderTensorUtility.extract_terrain_tensor(plane)

    @staticmethod
    def save_blender_state(
            filepath: str = "debug_test_heightmap.blend",
            compress: bool = True
    ) -> bool:
        """
        Save current Blender state to .blend file

        Args:
            filepath: Path where to save the .blend file
            compress: Whether to compress the file

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure .blend extension
            filepath_obj = Path(filepath)
            if filepath_obj.suffix != '.blend':
                filepath = str(filepath_obj.with_suffix('.blend'))

            # Save the file
            bpy.ops.wm.save_as_mainfile(filepath=filepath, compress=compress)

            print(f"✓ Saved Blender state to: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Error saving Blender state: {e}")
            return False

# Tests

def test_extract_heightmap_with_geometry_nodes():
    """Test extracting heightmap from plane with geometry nodes applied"""
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear node groups
    for node_group in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_group)

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "GeometryTerrain"

    # Add subdivisions
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=15)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create simple noise displacement geometry nodes
    node_group = bpy.data.node_groups.new("NoiseDisplacement", "GeometryNodeTree")
    nodes = node_group.nodes
    links = node_group.links

    # Create nodes
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    noise = nodes.new("ShaderNodeTexNoise")
    set_pos = nodes.new("GeometryNodeSetPosition")

    # Set up interface (Blender 4.0+)
    if hasattr(node_group, 'interface'):
        node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    # Set noise scale
    noise.inputs["Scale"].default_value = 5.0

    # Connect nodes
    links.new(group_input.outputs[0], set_pos.inputs["Geometry"])
    links.new(noise.outputs["Fac"], set_pos.inputs["Offset"])
    links.new(set_pos.outputs["Geometry"], group_output.inputs[0])

    # Apply to plane
    geo_mod = plane.modifiers.new("GeometryNodes", "NODES")
    geo_mod.node_group = node_group

    # Force update
    bpy.context.view_layer.update()

    # Extract heightmap
    heightmap = BlenderTensorUtility.get_heightmap_by_name("GeometryTerrain")

    assert heightmap is not None, "Heightmap should not be None"
    assert isinstance(heightmap, torch.Tensor)
    assert heightmap.shape[0] > 1  # Has multiple vertices

    # With noise, heights should vary
    height_std = heightmap.std().item()
    assert height_std > 0.01, "Heights should vary with noise applied"

    print(f"✓ Heightmap with geometry nodes shape: {heightmap.shape}")
    print(f"✓ Height std dev: {height_std:.3f}")
    print(f"✓ Height range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")

    # Also save heightmap as tensor
    torch.save(heightmap, "debug_test_heightmap.pt")
    print(f"✓ Saved heightmap tensor to: debug_test_heightmap.pt")




def set_colors_in_ramp(color_ramp_node, color_names: List[str], positions: List[float] = None):
    """
    Set colors on a color ramp node using color names

    Args:
        color_ramp_node: The color ramp node
        color_names: List of color names (e.g., ["Blue", "Green", "White"])
        positions: Optional positions (0.0-1.0). If None, evenly spaced

    Example:
        color_ramp = node_group.nodes["TerrainColorRamp"]
        set_colors_in_ramp(color_ramp, ["Blue", "Green", "White"])
        set_colors_in_ramp(color_ramp, ["Blue", "Green", "White"], [0.0, 0.6, 1.0])
    """
    # Create name -> color mapping from global color_df
    name_to_color = {
        row['name']: (row['red'], row['green'], row['blue'], row['alpha'])
        for _, row in color_df.iterrows()
    }

    # Clear existing elements
    while len(color_ramp_node.color_ramp.elements) > 1:
        color_ramp_node.color_ramp.elements.remove(color_ramp_node.color_ramp.elements[0])

    # Default to evenly spaced
    if positions is None:
        positions = [i / (len(color_names) - 1) for i in range(len(color_names))]

    # Set colors
    for i, (name, position) in enumerate(zip(color_names, positions)):
        if name not in name_to_color:
            available = list(name_to_color.keys())
            raise ValueError(f"Color '{name}' not found. Available: {available}")

        color = name_to_color[name]

        if i == 0:
            color_ramp_node.color_ramp.elements[0].position = position
            color_ramp_node.color_ramp.elements[0].color = color
        else:
            element = color_ramp_node.color_ramp.elements.new(position)
            element.color = color


def load_blend(filepath:str):
    """
    Load a template .blend file (resets Blender state)

    Args:
        filepath: Path to the template .blend file

    Returns:
        True if successful
    """
    from pathlib import Path

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Template not found: {filepath}")

    # Load the blend file (this resets everything)
    bpy.ops.wm.open_mainfile(filepath=filepath)
    bpy.context.view_layer.update()

    print(f"✓ Loaded template: {filepath}")
    return True




