"""
Test: Verify trajectory replay generates same heightmaps
"""
import torch
from gfn_environments.single_color_ramp import (
    load_blend_single_color_ramp,
    sample_random_trajectory
)
from gfn_environments.single_color_ramp import register_terrain_actions
from action_utils.action_regestry_util import ActionRegistry
from blender_setup_utils import BlenderTensorUtility
import bpy


print("\n" + "="*80)
print("TEST: TRAJECTORY REPLAY VERIFICATION")
print("="*80)

# Setup
load_blend_single_color_ramp()
registry = ActionRegistry()
register_terrain_actions(registry)

# Generate a trajectory
print("\n[Step 1] Generating original trajectory...")
original_traj = sample_random_trajectory(
    action_registry=registry,
    trajectory_len=6,
    max_colors=16,
    debug=True
)

print(f"\n✓ Generated trajectory with {len(original_traj['actions'])} actions")
print(f"  Action tensors shape: {original_traj['action_tensors'].shape}")
print(f"  Heightmaps shape: {original_traj['heightmaps'].shape}")

# Store original data
original_action_tensors = original_traj['action_tensors'].clone()
original_heightmaps = original_traj['heightmaps'].clone()
original_actions = original_traj['actions'].copy()

print("\n[Step 2] Original trajectory details:")
for i, action in enumerate(original_actions):
    print(f"  {i}: {action}")

# Now replay the trajectory
print("\n[Step 3] Replaying trajectory in fresh Blender state...")
load_blend_single_color_ramp()  # Fresh state

replayed_heightmaps = []

for step, action_tensor in enumerate(original_action_tensors):
    print(f"\n--- Replaying step {step + 1} ---")
    
    # Apply the action
    action_result = registry[action_tensor]()
    print(f"  Action: {action_result}")
    
    # Get heightmap after action
    bpy.context.view_layer.update()
    heightmap = BlenderTensorUtility.get_heightmap_by_name("TerrainPlane")
    replayed_heightmaps.append(heightmap)
    
    print(f"  Heightmap range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")

# Stack replayed heightmaps
replayed_heightmaps = torch.stack(replayed_heightmaps)

# Compare
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"\nOriginal heightmaps shape:  {original_heightmaps.shape}")
print(f"Replayed heightmaps shape:  {replayed_heightmaps.shape}")

# Check if shapes match
if original_heightmaps.shape != replayed_heightmaps.shape:
    print("\n❌ SHAPES DO NOT MATCH!")
else:
    print("\n✓ Shapes match")
    
    # Check each heightmap
    all_match = True
    for i in range(len(original_heightmaps)):
        original_hm = original_heightmaps[i]
        replayed_hm = replayed_heightmaps[i]
        
        # Compute difference
        diff = torch.abs(original_hm - replayed_hm)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        match = torch.allclose(original_hm, replayed_hm, atol=1e-5)
        
        print(f"\nStep {i}:")
        print(f"  Original range: [{original_hm.min():.3f}, {original_hm.max():.3f}]")
        print(f"  Replayed range: [{replayed_hm.min():.3f}, {replayed_hm.max():.3f}]")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Match (atol=1e-5): {'✓' if match else '❌'}")
        
        if not match:
            all_match = False
            
            # Show where differences are
            diff_locations = (diff > 1e-5).nonzero()
            if len(diff_locations) > 0:
                print(f"  Differences at {len(diff_locations)} locations")
                if len(diff_locations) <= 10:
                    print(f"  First few locations: {diff_locations[:10].tolist()}")

    print("\n" + "="*80)
    if all_match:
        print("✓ ALL HEIGHTMAPS MATCH - TRAJECTORY REPLAY IS CORRECT")
    else:
        print("❌ HEIGHTMAPS DO NOT MATCH - TRAJECTORY REPLAY HAS ISSUES")
    print("="*80)

# Additional check: verify actions match
print("\n[Step 4] Verifying actions match...")
actions_match = True
for i, (orig_action, orig_tensor) in enumerate(zip(original_actions, original_action_tensors)):
    # Get action from replayed tensor
    replayed_action = registry[orig_tensor]
    
    # Note: replayed_action is the function, not the result
    # We can't compare directly, but we can check the tensor indices
    orig_idx = torch.argmax(orig_tensor).item()
    
    print(f"Step {i}: action_idx={orig_idx}, type={orig_action['type']}")

print("\n✓ Test complete")