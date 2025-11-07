from gfn_environments.single_color_ramp import *


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

from gfn_environments.single_color_ramp import *


def test_serialize_blender_env():
    blender_api = BlenderTerrainAPI()
    blender_api.reset_env()
    print(blender_api.blender_env_to_tensor())
    s_wstep = v2StepWEnv()

    one_hot_2 = s_wstep.action_to_one_hot(2)
    one_hot_3 = s_wstep.action_to_one_hot(3)
    one_hot_4 = s_wstep.action_to_one_hot(4)
    for i in range(s_wstep.n_actions):
        s_wstep.execute_idx(blender_api, i)

    print('='*70)

    print(blender_api.blender_env_to_tensor())

def test_stop_state():
    """
    this should show after that there are now enums for -2 aka end
    """
    blender_api = BlenderTerrainAPI()
    blender_api.reset_env()
    s_wstep = v2StepWEnv()
    s_wstep.execute_idx(blender_api, 0)
    s_wstep.execute_idx(blender_api, 2)

