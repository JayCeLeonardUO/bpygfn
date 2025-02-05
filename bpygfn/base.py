from collections import defaultdict
from typing import Callable, Dict, Iterator, Optional, Union

import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.states import States
from torch.types import Number

QUATERNION_SIZE = 4
VOLUME_SIZE = 1


def my_rotate(target):
    print(target)
    return


# Define a type alias for our action function
StateFunction = Callable[[torch.Tensor], torch.Tensor]
ActionList = Dict[Union[int, Number], StateFunction]


def move_forward(state: torch.Tensor) -> torch.Tensor:
    new_state = state.clone()
    new_state[3] = 1.0
    return new_state


def move_backward(state: torch.Tensor) -> torch.Tensor:
    new_state = state.clone()
    new_state[3] = -1.0
    return new_state


def turn_left(state: torch.Tensor) -> torch.Tensor:
    new_state = state.clone()
    new_state[0:3] = torch.tensor([0.0, -1.0, 0.0])
    return new_state


ACTION_LIST: ActionList = {
    0: move_forward,
    1: move_backward,
    2: turn_left,
}


def action_generator(
    states: torch.Tensor, actions: torch.Tensor, action_list
) -> Iterator[torch.Tensor]:
    """
    Generates new states by applying actions one at a time

    Args:
        states: torch.Tensor of shape (batch_size, state_dim)
        actions: torch.Tensor of shape (batch_size,)

    Yields:
        torch.Tensor of shape (state_dim,) for each state
    """
    for state, action in zip(states, actions):
        yield action_list[action.item()](state)


class SuperSimpleEnv(DiscreteEnv):
    # should take in the actions dict
    def __init__(
        self,
        history_size: int,
        action_list: ActionList = ACTION_LIST,
        device_str: Optional[str] = "cpu",
    ):
        """
        NOTE: the states that this env output are what the MLP will see
        As in --they get put directally into the model with no preprocessing

        Arguments:
            n_actions: int - number of action
            history_size: int, - len of the history of prev actions you want in the state
            action_list: ActionList, - dict of fn pointers, this internal defindes n_actions
            device_str: Optional[str] = None, - can be "cpu" or "cuda"

        """
        action_shape = (1,)
        n_actions = len(action_list)
        state_shape = ((QUATERNION_SIZE + VOLUME_SIZE + (n_actions * history_size)),)
        s0 = torch.zeros(state_shape)
        self.action_list = defaultdict(None, action_list)

        # Pre-define the vectorized step function
        def apply_single_step(
            state: torch.Tensor, action: torch.Tensor
        ) -> torch.Tensor:
            return self.action_list[action.item()](state)

        self.batch_step = torch.vmap(apply_single_step, in_dims=(0, 0))
        super().__init__(
            n_actions=len(self.action_list),
            s0=s0,
            state_shape=state_shape,
            device_str=device_str,
            action_shape=action_shape,
        )

    def apply_batched_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(list(action_generator(states, actions, self.action_list)))

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.

        action[0] one will be applied to state[0] and so on
        returns the new batch of states
        Arguements:
            states: States, states you want to apply the action to
            actions: Actions action you want to apply the states
        Returns:
            torch.Tensor
        """
        # verify that states are in the right shape
        # apply the action on the state
        # First create a tensor of the right shape
        new_states = states.tensor.clone()

        # Iterate through batch and apply actions using tensor operations
        # TODO: this could get costly in traing, possably look for a way to use vmap
        for idx, (state, action) in enumerate(zip(states.tensor, actions.tensor)):
            action_key = int(action)
            new_states[idx] = self.action_list[action_key](state)

        return new_states  # pyright: ignore
        return self.batch_step(states.tensor, actions.tensor)

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        return super().is_action_valid(states, actions, backward)

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        return super().backward_step(states, actions)

    def update_masks(self, states: States) -> None:
        return super().update_masks(states)

    """ 
    #from the Debuger I was not able to find any where this i called other then testing
    def make_random_states_tensor(self, batch_shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randint(
            0, self.height, batch_shape + self.s0.shape, device=self.device
        )
    """
