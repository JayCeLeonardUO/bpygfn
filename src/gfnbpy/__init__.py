import bpy

def create_cube():
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    cube = bpy.context.object
    cube.name = "MyCube"
    return cube
