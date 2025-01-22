#%%
import sys
import gfn
import bpy
import bmesh


# action definitions - complonents of the action space
action_vocab = ["left", "right", "up", "down", "jump", "shoot"]

def vol_cube(cube):
    bm = bmesh.new()
    bm.from_mesh(cube.data)
    # this fn is expensive but gflow nets are not 
    volume = bm.calc_volume()
    bm.free()
    return volume

# reward for the trajectory
def reward_condition_2x_cube(initial_cube,final_cube): 
    initial_volume  = initial_cube.data
    return 0


def create_test_cube():
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    return bpy.context.active_object

def main(args):
    # Process the command line arguments
    print(f"Arguments received: {args}")
    intitial_cube  = create_test_cube()
    print(intitial_cube)

    print([vertex.co for vertex in intitial_cube.data.vertices])
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
# %%
