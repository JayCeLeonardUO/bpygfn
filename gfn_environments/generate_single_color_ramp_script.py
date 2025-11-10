import bpy

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

    # Set default colors directly
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # White

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


def main():
    """
    Generate the template .blend file and save it to the files directory
    """
    from pathlib import Path

    # Define the files directory path
    # Adjust this path based on your project structure
    files_dir = Path(__file__).parent / "files"

    # Create the files directory if it doesn't exist
    files_dir.mkdir(parents=True, exist_ok=True)

    # Define the output filepath
    output_path = files_dir / "single_color_ramp.blend"

    print(f"Generating template blend file...")
    print(f"Output path: {output_path}")

    # Generate the template
    template_path = generate_template_blend(filepath=str(output_path))

    print(f"✓ Template successfully saved to: {template_path}")
    return template_path


if __name__ == "__main__":
    import bpy
    main()
