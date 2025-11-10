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


def load_blend(filepath: str):
    """
    Load a template .blend file (resets Blender state)

    Args:
        filepath: Path to the template .blend file

    Returns:
        True if successful
    """
    from pathlib import Path
    from IPython.utils.capture import capture_output

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Template not found: {filepath}")

    # Capture all output from the load operation
    with capture_output() as captured:
        bpy.ops.wm.open_mainfile(filepath=filepath)
        bpy.context.view_layer.update()

    # captured.stdout and captured.stderr contain the output
    # but we're just discarding it

    return True



def save_blend(filepath:str):
    bpy.ops.wm.save_as_mainfile(filepath=filepath, compress=True)


def screenshot_viewport_to_png(filepath: str, resolution_x: int = 800, resolution_y: int = 600):
    import os
    """
    Quick render of the scene from a default viewpoint (no camera setup needed).

    Args:
        filepath: Path where the PNG should be saved (e.g., "./output/screenshot.png")
        resolution_x: Width of the image in pixels
        resolution_y: Height of the image in pixels
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    scene = bpy.context.scene

    # Store original settings
    original_camera = scene.camera
    original_engine = scene.render.engine

    # Create a temporary camera if none exists
    camera_data = bpy.data.cameras.new(name="TempCamera")
    camera_object = bpy.data.objects.new("TempCamera", camera_data)
    bpy.context.collection.objects.link(camera_object)

    # Position camera to view the mesh (adjust these values as needed)
    camera_object.location = (7, -7, 5)
    camera_object.rotation_euler = (1.1, 0, 0.785)

    # Set as active camera
    scene.camera = camera_object

    # Configure render settings for quick preview
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Fast rendering (Blender 4.x)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = filepath
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100

    # Quick render
    bpy.ops.render.render(write_still=True)

    # Cleanup: remove temporary camera
    bpy.data.objects.remove(camera_object, do_unlink=True)
    bpy.data.cameras.remove(camera_data)

    # Restore original settings
    scene.camera = original_camera
    scene.render.engine = original_engine

    print(f"Viewport screenshot saved to: {filepath}")

