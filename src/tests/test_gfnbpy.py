import unittest
import bpy
from gfnbpy import create_cube

class TestCreateCube(unittest.TestCase):
    def setUp(self):
        # Clear existing objects
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_create_cube(self):
        # Create a cube
        cube = create_cube()

        # Check if the cube was created
        self.assertIsNotNone(cube)
        self.assertEqual(cube.name, "MyCube")

        # Check if the cube is in the scene
        self.assertIn(cube.name, bpy.context.scene.objects)
        # Print the cube's values
        print(f"Cube location: {cube.location}")
        print(f"Cube dimensions: {cube.dimensions}")
        # Check the cube's properties
        self.assertEqual(tuple(cube.location), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(cube.dimensions.x, 2.0, places=5)
        self.assertAlmostEqual(cube.dimensions.y, 2.0, places=5)
        self.assertAlmostEqual(cube.dimensions.z, 2.0, places=5)

if __name__ == '__main__':
    unittest.main()