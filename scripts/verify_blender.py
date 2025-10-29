import bpy

def main():
    # Quick test - if this runs without error, bpy is working
    print(f"Blender {bpy.app.version_string} - bpy working!")
    bpy.ops.mesh.primitive_cube_add()
    print(f"Created cube: {bpy.context.active_object.name}")

if __name__ == "__main__":
    main()