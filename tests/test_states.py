import pytest
import torchgfn

@pytest.fixture
def test_args():
    return ["one","two","three"]

class TestStates:
    def test_actions_length(self, test_args):
       pass 
        