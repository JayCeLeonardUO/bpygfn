from dataclasses import dataclass

import torch

# TODO: make the state more like a sruct then how i do it how
QUAT_SLICE = slice(0, 3)
VOLUME_SLICE = 4


@dataclass
class StateView:
    tensor: torch.Tensor  # The full state tensor

    @staticmethod
    def _quaternion_dims():
        return 4  # Or however many dimensions your quaternion has

    @staticmethod
    def _volume_dims():
        return 1  # Or however many dimensions your volume has

    @staticmethod
    def _base_size():
        return StateView._quaternion_dims() + StateView._volume_dims()

    @property
    def quaternion(self):
        return self.tensor[..., : StateView._quaternion_dims()]

    @property
    def volume(self):
        return self.tensor[..., StateView._quaternion_dims() : StateView._base_size()]

    @property
    def action_history(self):
        return self.tensor[..., StateView._base_size() :]

    @action_history.setter
    def action_history(self, new_history):
        self.tensor[..., StateView._base_size() :] = new_history


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
