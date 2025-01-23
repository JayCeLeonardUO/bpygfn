#%%
import sys
from dataclasses import dataclass


"""
bpy HAS to be imported before bmesh
"""
import bpy
import bmesh
from mathutils import Vector

# action definitions - complonents of the action space
action_vocab = ["left", "right", "up", "down", "jump", "shoot"]

def reset_context():
    pass

def log(thing):
    print(thing)
    
def cube_volume(cube):
    bm = bmesh.new()
    bm.from_mesh(cube.data)
    # this fn is expensive but gflow nets are not 
    volume = bm.calc_volume()
    bm.free()
    return volume

def get_context_objects():
    return [obj for obj in bpy.context.scene.objects]

def print_context_objects():
    """
    context is the state  ...
    im pretty sure that the is some kind of error that I should be handling here
    """
    for object in bpy.context.scene.objects:
        print(object.data)

def compare_vols(mesh1,mesh2):
    """
    compares two mesh vols
    with an opinionated theashold
    """
    return True if cube_volume(mesh1) == cube_volume(mesh2) else False 

# reward for the trajectory
def reward_cond_2x_scale_and_rot(mesh) -> bool:
    """
    check the passed in mesh in comparison to the baseline
    """ 
    return
def create_test_cube():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    return bpy.context.active_object

def print_mesh(mesh):
    print([vertex.co for vertex in mesh.data.vertices])

def scale_2x(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    bmesh.ops.scale(bm,
                    vec=Vector((2.0,2.0,2.0)),
                    verts=bm.verts)
    bm.to_mesh(mesh.data)
    bm.free()

def assemble_composition(mesh):
    """
    """
    def create():
        return bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    
    def scale_halfx():
        bm = bmesh.new()
        bm.from_mesh(mesh.data)
        bmesh.ops.scale(bm,
                        vec=Vector((2.0,2.0,2.0)),
                        verts=bm.verts)
        bm.to_mesh(mesh.data)
        bm.free()
 
    def rot():
        return 0

def state_masking(state):
    # if there is a cube then then mask out creat cube
    return

#%%
def main(args):
    # Process the command line arguments
    print(f"Arguments received: {args}")
    intitial_cube  = create_test_cube()
    print(intitial_cube)
    print_mesh(intitial_cube)
    assemble_composition()
    print_mesh(intitial_cube)

    return 0


if __name__ == "__main__":
    log("entering main") 
    main(sys.argv[1:])
# %%
