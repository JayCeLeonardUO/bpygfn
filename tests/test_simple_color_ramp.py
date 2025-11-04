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
