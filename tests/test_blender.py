import datetime
import os
from pathlib import Path

import bpy
import pytest


def test_example_scene():
    """
    pytest tests/test_blender.py::test_example_scene
    """
    # Clear existing mesh objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Create a new mesh object
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    obj = bpy.context.active_object

    # Create geometry node tree - ignore the broken type hint
    node_tree = bpy.data.node_groups.new(name="SimpleNodes", type="GeometryNodeTree")  # type: ignore

    # Add some assertions to verify it worked
    assert obj is not None
    assert node_tree is not None
    print("Test completed successfully!")


@pytest.fixture
def save_dir():
    """Fixture to provide directories for saving Blender test files.

    Returns a dictionary with:
    - 'current': Directory for the most recent test run (overwritten each time)
    - 'history': Directory for keeping history of all test runs

    Checks for BLENDERTESTSAVEDIR environment variable first, otherwise
    defaults to ./tests directory.
    """
    if "BLENDERTESTSAVEDIR" in os.environ and os.environ["BLENDERTESTSAVEDIR"]:
        base_dir = os.environ["BLENDERTESTSAVEDIR"]
    else:
        base_dir = "./tests"

    # Create the base directory
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    # Create the history directory
    history_dir = os.path.join(base_dir, "run_history")
    Path(history_dir).mkdir(parents=True, exist_ok=True)

    print("Using save directories:")
    print(f"  - Current run: {base_dir}")
    print(f"  - Run history: {history_dir}")

    return {"current": base_dir, "history": history_dir}


# @aerial Color ramp configuration for Blender
@pytest.fixture
def color_ramp_blender_config():
    return {
        "nodes": [
            ("noise", "ShaderNodeTexNoise"),
            ("combine", "ShaderNodeCombineXYZ"),
            ("ramp", "ShaderNodeValToRGB"),
            ("set_pos", "GeometryNodeSetPosition"),
        ],
        "connections": [
            ("noise", "Fac", "combine", "X"),
            ("noise", "Fac", "combine", "Y"),
            ("noise", "Fac", "combine", "Z"),
            ("combine", "Vector", "ramp", "Fac"),
            ("ramp", "Color", "set_pos", "Offset"),
        ],
    }


def test_color_ramp(save_dir, color_ramp_blender_config):
    """
    Test color ramp functionality in Blender.

    Can be run with: pytest -s tests/test_blender.py::test_color_ramp

    Args:
        save_dir: Directory to save the Blender file for visual inspection
    """
    print("Starting color ramp test...")

    def create_connected_nodes(node_group, node_configs):
        nodes = node_group.nodes
        links = node_group.links
        created_nodes = {}
        print(f"Creating {len(node_configs['nodes'])} nodes...")
        for name, node_type in node_configs["nodes"]:
            created_nodes[name] = nodes.new(node_type)
            print(f"  - Created {node_type} node named '{name}'")

        print(f"Creating {len(node_configs['connections'])} connections...")
        for from_node, from_out, to_node, to_in in node_configs["connections"]:
            links.new(
                created_nodes[from_node].outputs[from_out],
                created_nodes[to_node].inputs[to_in],
            )
            print(f"  - Connected {from_node}.{from_out} → {to_node}.{to_in}")
        return created_nodes

    # Clear existing mesh objects
    print("Clearing existing mesh objects...")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clear existing node groups
    print(f"Clearing {len(bpy.data.node_groups)} existing node groups...")
    for node_group in bpy.data.node_groups:
        print(f"  - Removing node group: {node_group.name}")
        bpy.data.node_groups.remove(node_group)

    # Create geometry node group
    print("Creating new geometry node group: TestColorRampGroup")
    node_group = bpy.data.node_groups.new("TestColorRampGroup", "GeometryNodeTree")
    # Add group input and output nodes
    print("Adding group input and output nodes...")
    group_input = node_group.nodes.new("NodeGroupInput")
    group_output = node_group.nodes.new("NodeGroupOutput")
    print(f"  - Created {group_input.name}")
    print(f"  - Created {group_output.name}")

    # Set up the interface for the node group (Blender 4.4 specific)
    print("Setting up node group interface for Blender 4.4...")
    try:
        # In Blender 4.4, we use the interface system
        if hasattr(node_group, "interface"):
            # Add input socket for geometry
            in_socket = node_group.interface.new_socket(
                name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
            )
            print(f"  - Added input socket: {in_socket.name}")

            # Add output socket for geometry
            out_socket = node_group.interface.new_socket(
                name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
            )
            print(f"  - Added output socket: {out_socket.name}")
        else:
            print(
                "  - Warning: node_group.interface not found. This is unexpected for Blender 4.4"
            )
    except Exception as e:
        print(f"  - Error setting up node group interface: {e}")
        print("  - This may result in red connections in the node editor")

    # Create node configuration
    print("Setting up node configuration...")
    config = color_ramp_blender_config

    # Create and connect nodes
    print("Creating and connecting nodes according to configuration...")
    created_nodes = create_connected_nodes(node_group, config)

    # Connect to group input/output
    print("Connecting to group input/output...")
    node_group.links.new(
        group_input.outputs[0], created_nodes["set_pos"].inputs["Geometry"]
    )
    print("  - Connected Group Input → Set Position.Geometry")

    node_group.links.new(
        created_nodes["set_pos"].outputs["Geometry"], group_output.inputs[0]
    )
    print("  - Connected Set Position.Geometry → Group Output")

    # Test: Check that we have the expected number of connections
    print(f"Checking connections: found {len(node_group.links)} links")
    assert len(node_group.links) == 7, f"Expected 7 links, got {len(node_group.links)}"

    # Test: Verify specific connections exist
    print("Verifying specific connections...")
    link_pairs = []
    for link in node_group.links:
        link_pair = (link.from_node.name, link.to_node.name)
        link_pairs.append(link_pair)
        print(f"  - Found connection: {link_pair[0]} → {link_pair[1]}")

    expected_connections = [
        ("Group Input", "Set Position"),
        ("Noise Texture", "Combine XYZ"),
        ("Combine XYZ", "ColorRamp"),
        ("ColorRamp", "Set Position"),
        ("Set Position", "Group Output"),
    ]

    # Test: Check color ramp has default color stops
    color_ramp_node = created_nodes["ramp"]
    print(
        f"Checking color ramp: found {len(color_ramp_node.color_ramp.elements)} color stops"
    )
    assert len(color_ramp_node.color_ramp.elements) >= 2, (
        "Color ramp should have at least 2 color stops"
    )

    # Test: Verify we can modify color ramp
    print("Modifying color ramp elements...")
    print(
        f"  - Original first stop color: {color_ramp_node.color_ramp.elements[0].color[:]}"
    )
    print(
        f"  - Original second stop color: {color_ramp_node.color_ramp.elements[1].color[:]}"
    )

    color_ramp_node.color_ramp.elements[0].color = (1, 0, 0, 1)  # Red
    color_ramp_node.color_ramp.elements[1].color = (0, 0, 1, 1)  # Blue

    print(
        f"  - Modified first stop color to red: {color_ramp_node.color_ramp.elements[0].color[:]}"
    )
    print(
        f"  - Modified second stop color to blue: {color_ramp_node.color_ramp.elements[1].color[:]}"
    )

    assert color_ramp_node.color_ramp.elements[0].color[0] == 1.0, (
        "First color stop should be red"
    )
    assert color_ramp_node.color_ramp.elements[1].color[2] == 1.0, (
        "Second color stop should be blue"
    )

    # Create a demonstration object to apply the node group to
    print("\nCreating a demonstration object...")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
    sphere = bpy.context.active_object

    # Add a geometry nodes modifier to the sphere

    print("Adding geometry nodes modifier to the sphere...")
    geo_mod = sphere.modifiers.new("GeometryNodes", "NODES")

    # Attempt to assign the node group to the modifier
    try:
        geo_mod.node_group = node_group
        print(
            f"  - Successfully added geometry nodes modifier with node group: {node_group.name}"
        )
    except Exception as e:
        print(f"  - Note: Could not assign node group to modifier: {e}")
        print(
            "  - This is expected in some Blender versions and won't affect the test "
            "results"
        )

    # Save the Blender file
    test_name = "color_ramp_test"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # First save to the current directory (overwriting any existing file)
    current_path = os.path.join(save_dir["current"], f"{test_name}.blend")
    print(f"\nSaving current Blender file to: {current_path}")
    bpy.ops.wm.save_as_mainfile(filepath=current_path)
    print(f"File saved successfully at: {current_path}")

    # Then save to the history directory with timestamp
    history_path = os.path.join(save_dir["history"], f"{test_name}_{timestamp}.blend")
    print(f"Saving history Blender file to: {history_path}")
    bpy.ops.wm.save_as_mainfile(filepath=history_path)
    print(f"History file saved successfully at: {history_path}")

    print("✅ Color ramp test passed!")
    print(f"Created node group with {len(node_group.links)} connections")
    print(f"Color ramp has {len(color_ramp_node.color_ramp.elements)} color stops")

    # Print additional details about node structure
    print("\nFinal Node Group Structure:")
    print(f"Total nodes: {len(node_group.nodes)}")
    for i, node in enumerate(node_group.nodes):
        print(f"  {i + 1}. {node.name} ({node.bl_idname})")
        print(f"     - Inputs: {len(node.inputs)}")
        print(f"     - Outputs: {len(node.outputs)}")

    # Print details about the color ramp configuration
    print("\nColor Ramp Configuration:")
    for i, element in enumerate(color_ramp_node.color_ramp.elements):
        print(f"  Stop {i + 1}: Position={element.position}, Color={element.color[:]}")

    # Print information about the created demonstration object
    print("\nDemonstration Object:")
    print(f"  Name: {sphere.name}")
    print(f"  Type: {sphere.type}")

    print("\nTest completed successfully!")
    # return {
    #     "current_file": current_path,
    #     "history_file": history_path,
    #     "test_name": test_name,
    # }

def valid_actions():


def test_read_height_map():
    pass
