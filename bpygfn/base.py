from collections import defaultdict
from typing import Callable, Dict, Optional, Union

import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates, States
from torch.types import Number

from bpygfn.quat import init_state

# a quaternion will aways be 4.... cuz its a quaternion
QUATERNION_DIMS = 4
# This is the volume for a mesh... for now it is a single integer in the state
# TODO: I don't thing this global is apropreate
VOLUME_DIMS = 1


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


# TODO: make this dependency injection
ACTION_LIST: ActionList = {
    0: move_forward,
    1: move_backward,
    2: turn_left,
}


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
        self.base_shape = QUATERNION_DIMS + VOLUME_DIMS

        # welp... looks like from the hypergrid env that n_action should account for the exit action
        # hence we add one to the len of the actions list...
        n_actions = len(action_list)
        state_shape = ((QUATERNION_DIMS + VOLUME_DIMS + (n_actions * history_size)),)

        self.state_action_history_slices = [
            slice(i, i + n_actions)
            for i in range(self.base_shape, *state_shape, n_actions)
        ]
        self.quat_slice = slice(0, QUATERNION_DIMS)
        s0 = init_state(state_shape)
        self.action_list = defaultdict(None, action_list)

        super().__init__(
            n_actions=len(self.action_list),
            s0=s0,
            state_shape=state_shape,
            device_str=device_str,
            action_shape=action_shape,
        )

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.

        action[0] one will be applied to state[0] and so on
        returns the new batch of states

        Note- it is step that will alter the action history of the state (for now)

        Arguements:
            states: States, states you want to apply the action to
            actions: Actions action you want to apply the states
        Returns:
            torch.Tensor
        """

        # First create a tensor of the right shape
        new_states = states.tensor.clone()

        # Iterate through batch and apply actions using tensor operations
        # TODO: this could get costly in traing, possably look for a way to use vmap
        for idx, (state, action) in enumerate(zip(states.tensor, actions.tensor)):
            action_key = int(action)
            new_states[idx] = self.action_list[action_key](state)

            # Get the existing action history, excluding oldest action
            last_states = new_states[
                idx,
                slice(
                    self.state_action_history_slices[0].start,
                    self.state_action_history_slices[1].stop,
                ),
            ]
            import pudb

            pudb.set_trace()
            # Create one-hot encoding for the new action
            action_onehot = torch.zeros(self.n_actions, device=new_states.device)

            action_onehot[action_key] = 1
            print(action_onehot)
            # Update the action history - new action goes at start, drop oldest
            new_states[idx, self.base_shape :] = torch.cat([action_onehot, last_states])
        # why does this fn return a tensoe and not states????
        return new_states  # pyright: ignore

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        """
        as per the domentation... is_action_valid must be difined by the user...
        But im pretty sure this fn is taken care of by the DiscreteEnv ABC
        """
        # TODO: investigate if I have to impliment this or if using the partent is ok
        return super().is_action_valid(states, actions, backward)

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        """
        This is a trajectory of a Backwards step.

        In This env provents repeated actions, so That means that it will match the forward masks logic
        """
        return super().backward_step(states, actions)

    def update_masks(self, states: DiscreteStates) -> None:  # pyright: ignore
        """
        This is more for just validation that masks actually prevent actions from being taken

        the state that this fn is preventing is "consecutive" actions
        Mind you this is called on the entire batch shape... not just one given shape
        """

        import pudb

        pudb.set_trace()
        valid_actions_mask = torch.stack(
            [
                ~state[self.state_action_history_slices[0]].bool()
                for state in states.tensor
            ]
        )
        states.forward_masks = valid_actions_mask
        # TODO: early termination logic
        # Intionnally NOT using this functtion. ... helper fn is not helpfull or intuitive...
        # this funtion expects invers logic for allowed states.... wtf???
        # states.set_nonexit_action_masks(my_cond, allow_exit=True)

    def reward(self, final_states: States) -> torch.Tensor:
        """
        the ACB if the env class wants the reward to call in state

        Possable Implimentation Err?
            Discrete enve should have made it so the reward takes in descrete state
            you have to overide the virtual fn for that to happen....
            Discrete evironment does not over ride the type hinting

            But hypergrid overrides this with DiscreteState... for litrally no reasion

            its only type hinting but now I wonder why there is even the distinction
        Arguments:
            final_states: these are the final states
                you note that this will be the result of the trajectory...
        Returns:
            This rewards orentations (quaternions) that are up right and 2x > starting volume
            OR
            rewards orentations that are "up-side-down" and <1.5x the scale of the original
        """

        return final_states.tensor
