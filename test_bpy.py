import pytest
import bpy


@pytest.fixture
def start_context():
    def create_basline_1x1_cube():
        """
        this will return a 0x0 cube
        fn intentions is just to allow for a ranfom baseline
        .... baisically a .... thingy in pytest -2_o idr
        """
        # note that this will return something other then finnished if something when wrong
        bpy.ops.mesh.primitive_cube_add(size=-1.0, location=(0, 0, 0))
        return bpy.context.active_object

    def get_context() -> list[object]: 
        return [obj for obj in bpy.context.scene.objects]


    create_basline_1x1_cube()
    return get_context()


def test_reward_cond_2x_scale_and_rot(start_context):
 
    assert False