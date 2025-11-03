from gfn_environments.single_color_ramp import *

import bpy
import os

import bpy
import os

import bpy
import os

import bpy
import os

import bpy
import os

import bpy
import os


def screenshot_viewport_to_png(filepath: str, resolution_x: int = 800, resolution_y: int = 600):
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


def test_state():
    api_instance = BlenderTerrainAPI()
    state = get_initial_environment_state()
    tensor = state.to_state_tensor()
    state.apply_to_blender(blender_api=api_instance)

    from blender_setup_utils import save_blend
    save_blend(filepath="./tests/file_dump/test_state.blend")

    # Screenshot the initial viewport
    screenshot_viewport_to_png(filepath="./tests/file_dump/test_state_initial.png")

    print("Initial state:", state)

    # Apply an action (e.g., set noise_w parameter)
    action_name = 'set_w'
    value_idx = 2  # Pick an index from VALID_W
    new_state = state.apply_action(action_name, value_idx)
    new_state.execute_action_on_blender(api_instance, action_name, value_idx)

    # Screenshot after first action
    screenshot_viewport_to_png(filepath="./tests/file_dump/test_state_after_w.png")

    print("After set_w:", new_state)

    # Apply another action (e.g., set noise_scale parameter)
    action_name = 'set_scale'
    value_idx = 1  # Pick an index from VALID_SCALE
    new_state = new_state.apply_action(action_name, value_idx)
    new_state.execute_action_on_blender(api_instance, action_name, value_idx)

    # Screenshot after second action
    screenshot_viewport_to_png(filepath="./tests/file_dump/test_state_after_scale.png")

    print("After set_scale:", new_state)

    # Apply a color action
    action_name = 'add_color'
    value_idx = 5  # Pick a color from COLOR_PALETTE
    new_state = new_state.apply_action(action_name, value_idx)
    new_state.execute_action_on_blender(api_instance, action_name, value_idx)

    # Screenshot after adding color
    screenshot_viewport_to_png(filepath="./tests/file_dump/test_state_after_color.png")

    print("After add_color:", new_state)

