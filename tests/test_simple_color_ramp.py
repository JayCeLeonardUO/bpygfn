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


def test_state_serialization():
    """
    Test to inspect the tensor serialization of states.
    Shows exactly what goes IN and OUT of the model.
    """
    from blender_setup_utils import save_blend

    load_blend_single_color_ramp()
    blender_api = BlenderTerrainAPI()

    # Create trajectory from initial Blender state
    trajectory = StepWExperimentDefinition.Trajectory(blender_api)

    print("\n" + "=" * 80)
    print("MODEL INPUT/OUTPUT SERIALIZATION TEST")
    print("=" * 80)

    # Print action encoding map at the beginning
    print("\n" + "🗺️  ACTION ENCODING MAP")
    print("─" * 80)
    print("Flat Index → Action")
    print("─" * 80)

    action_dim = StepWExperimentDefinition.State.get_action_dim()
    for flat_idx in range(action_dim):
        action_name, value_idx = StepWExperimentDefinition.decode_action(flat_idx)
        action_info = StepWExperimentDefinition.ACTIONS[action_name]

        # Get the actual value if available
        if 'valid_values' in action_info and value_idx < len(action_info['valid_values']):
            actual_value = action_info['valid_values'][value_idx]
            print(f"  [{flat_idx:3d}] → {action_name:15s} value_idx={value_idx:2d}  (value={actual_value})")
        else:
            print(f"  [{flat_idx:3d}] → {action_name:15s} value_idx={value_idx:2d}")

    print("─" * 80)
    print(f"Total action space size: {action_dim}")
    print("=" * 80)

    def print_tensor_compact(tensor, name="Tensor"):
        """Print tensor in compact format, summarizing zeros"""
        nonzero_mask = tensor != 0
        num_nonzero = nonzero_mask.sum().item()
        num_zeros = len(tensor) - num_nonzero

        print(f"  Shape: {tensor.shape}")
        print(f"  Non-zero: {num_nonzero}, Zeros: {num_zeros}")

        if num_nonzero > 0:
            nonzero_indices = torch.where(nonzero_mask)[0]
            print(f"  Non-zero elements:")
            for idx in nonzero_indices:
                print(f"    [{idx.item():3d}] = {tensor[idx].item():.6f}")
        else:
            print(f"  All zeros")

    def print_io(trajectory, step_num, action_taken=None):
        """Print what goes into and out of the model"""

        print(f"\n{'─' * 80}")
        print(f"STEP {step_num}")
        if action_taken:
            print(f"Action taken: {action_taken.action_name} (value_idx={action_taken.value_idx})")
        print(f"{'─' * 80}")

        # MODEL INPUT
        state_tensor = trajectory.get_state_tensor()
        print(f"\n📥 MODEL INPUT (state_tensor):")
        print_tensor_compact(state_tensor, "state_tensor")

        # MODEL OUTPUT
        if action_taken:
            flat_idx = StepWExperimentDefinition.encode_action(
                action_taken.action_name,
                action_taken.value_idx
            )

            # Create example action tensor with this action selected
            action_tensor = torch.zeros(StepWExperimentDefinition.State.get_action_dim())
            action_tensor[flat_idx] = 1.0  # Mark which action was taken

            print(f"\n📤 MODEL OUTPUT (action_tensor):")
            print(f"  What model would output for this action:")
            print_tensor_compact(action_tensor, "action_tensor")
            print(f"  ")
            print(f"  Action '{action_taken.action_name}' (value_idx={action_taken.value_idx})")
            print(f"    → Encoded as flat index [{flat_idx}]")

            # Verify encoding
            decoded_name, decoded_value_idx = StepWExperimentDefinition.decode_action(flat_idx)
            match = decoded_name == action_taken.action_name and decoded_value_idx == action_taken.value_idx
            print(f"    → Decoding verification: {'✓ PASS' if match else '✗ FAIL'}")
        else:
            action_dim = StepWExperimentDefinition.State.get_action_dim()
            print(f"\n📤 MODEL OUTPUT (action_tensor):")
            print(f"  Shape: torch.Size([{action_dim}])")
            print(f"  Model would output probability distribution over all {action_dim} actions")

        return state_tensor, action_tensor if action_taken else None

    # Initial state
    print_io(trajectory, step_num=0)

    # Step 1: Step W
    action = StepWExperimentDefinition.Action(action_name='step_w', value_idx=0)
    trajectory.step(action, reward=0.0)
    print_io(trajectory, step_num=1, action_taken=action)

    # Step 2: Step Scale
    action = StepWExperimentDefinition.Action(action_name='step_scale', value_idx=0)
    trajectory.step(action, reward=0.0)
    print_io(trajectory, step_num=2, action_taken=action)

    # Step 3: Add first color
    action = StepWExperimentDefinition.Action(action_name='add_color', value_idx=5)
    trajectory.step(action, reward=0.0)
    print_io(trajectory, step_num=3, action_taken=action)

    # Step 4: Add second color
    action = StepWExperimentDefinition.Action(action_name='add_color', value_idx=12)
    trajectory.step(action, reward=0.0)
    print_io(trajectory, step_num=4, action_taken=action)

    # Step 5: Add third color
    action = StepWExperimentDefinition.Action(action_name='add_color', value_idx=20)
    trajectory.step(action, reward=0.0)
    print_io(trajectory, step_num=5, action_taken=action)

    # Step 6: Stop
    action = StepWExperimentDefinition.Action(action_name='stop', value_idx=0)
    trajectory.step(action, reward=1.0)
    print_io(trajectory, step_num=6, action_taken=action)

    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)


from gfn_environments.single_color_ramp import *
import torch
import random







# Now run your test
from gfn_environments.single_color_ramp import *
import torch
import random


def test_blender_env_to_tensor():
    """Test serializing Blender environment to tensor with hex color values"""
    load_blend_single_color_ramp()
    blender_api = BlenderTerrainAPI()

    print("=" * 80)
    print("TEST: blender_env_to_tensor() with HEX colors")
    print("=" * 80)

    def rgba_to_hex(rgba):
        """Convert RGBA tuple to hex integer"""
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        return (r << 16) | (g << 8) | b

    def print_tensor_readable(tensor, max_colors=32):
        """Print tensor in human-readable format"""
        w = tensor[0].item() * 100
        scale = tensor[1].item() * 50
        num_colors = int(tensor[-1].item() * max_colors)
        print(f"raw_tensor:{tensor}")
        print(f"[W={w:.2f}, Scale={scale:.2f}, ", end="")

        # Print colors as hex
        colors = []
        for i in range(num_colors):
            color_val = tensor[2 + i].item()
            if color_val != -1.0:
                # Denormalize to hex
                hex_int = int(color_val * 16777215)
                colors.append(f"0x{hex_int:06X}")
            else:
                colors.append("EMPTY")

        print(f"Colors=[{', '.join(colors)}], NumColors={num_colors}]")

    # Test 1: Initial state
    print("\n--- Test 1: Initial state ---")
    tensor = blender_api.blender_env_to_tensor()
    print_tensor_readable(tensor)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Expected: (35,) = [W, Scale, 32 colors, NumColors]")

    # Test 2: After stepping W
    print("\n--- Test 2: After stepping W by 0.5 ---")
    blender_api.step_w(0.5)
    tensor = blender_api.blender_env_to_tensor()
    print_tensor_readable(tensor)

    # Test 3: Add first color
    print("\n--- Test 3: After adding first color (palette idx 5) ---")
    first_color = StepWExperimentDefinition.COLOR_PALETTE[5]  # (0.3, 0.5, 0.7, 1.0)
    first_hex = rgba_to_hex(first_color)
    print(f"Adding color: RGB{first_color[:3]} = 0x{first_hex:06X}")
    blender_api.stack_color_ramp(first_color, max_colors=32)
    tensor = blender_api.blender_env_to_tensor()
    print_tensor_readable(tensor)

    # Test 4: Add second color
    print("\n--- Test 4: After adding second color (palette idx 12) ---")
    second_color = StepWExperimentDefinition.COLOR_PALETTE[12]  # (0.4, 0.6, 0.2, 1.0)
    second_hex = rgba_to_hex(second_color)
    print(f"Adding color: RGB{second_color[:3]} = 0x{second_hex:06X}")
    blender_api.stack_color_ramp(second_color, max_colors=32)
    tensor = blender_api.blender_env_to_tensor()
    print_tensor_readable(tensor)

    # Test 5: Add third color
    print("\n--- Test 5: After adding third color (palette idx 20) ---")
    third_color = StepWExperimentDefinition.COLOR_PALETTE[20]  # (0.6, 0.5, 0.3, 1.0)
    third_hex = rgba_to_hex(third_color)
    print(f"Adding color: RGB{third_color[:3]} = 0x{third_hex:06X}")
    blender_api.stack_color_ramp(third_color, max_colors=32)
    tensor = blender_api.blender_env_to_tensor()
    print_tensor_readable(tensor)

    # Test 6: Verify hex values are correct
    print("\n--- Test 6: Verify hex conversion ---")
    print(f"Color 1 normalized: {tensor[2].item():.6f}")
    print(f"Color 1 hex: 0x{int(tensor[2].item() * 16777215):06X}")
    print(f"Expected: 0x{first_hex:06X}")
    print(f"Color 2 normalized: {tensor[3].item():.6f}")
    print(f"Color 2 hex: 0x{int(tensor[3].item() * 16777215):06X}")
    print(f"Expected: 0x{second_hex:06X}")

    # Test 7: Custom max_colors
    print("\n--- Test 7: With max_colors=8 ---")
    tensor = blender_api.blender_env_to_tensor(max_colors=8)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Expected: (11,) = [W, Scale, 8 colors, NumColors]")
    print_tensor_readable(tensor, max_colors=8)

    # Test 8: Custom empty_value
    print("\n--- Test 8: With empty_value=0.0 ---")
    tensor = blender_api.blender_env_to_tensor(empty_value=0.0)
    print(f"Empty slot values (should be 0.0): {tensor[5:10].tolist()}")



    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("State representation: [W, Scale, Color1_hex, Color2_hex, ..., NumColors]")
    print("Each color is a single normalized hex value (0xRRGGBB)")
    print("=" * 80)




