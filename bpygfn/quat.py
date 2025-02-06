import torch

# TODO: make the state more like a sruct then how i do it how
QUAT_SLICE = slice(0, 4)
VOLUME_SLICE = 5


def init_state(state_shape):
    state = torch.zeros(state_shape)
    state[VOLUME_SLICE] = 1
    return state


def get_quaternion_slice(state: torch.Tensor):
    quaternion = state[QUAT_SLICE]
    print(quaternion)
    return state


def get_volume_slice(state: torch.Tensor):
    volume = state[VOLUME_SLICE]
    return volume


def scale_state_up(state: torch.Tensor):
    state[VOLUME_SLICE] *= 2
    return state


def scale_state_down(state: torch.Tensor):
    state[VOLUME_SLICE] *= 0.5
    return state
