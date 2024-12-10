import unittest
import bpy
from gfnbpy import draw_face

class TestCreateCube(unittest.TestCase):
    def setUp(self):
        # Clear existing objects
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_create_cube(self):
        draw_face()

if __name__ == '__main__':
    unittest.main()