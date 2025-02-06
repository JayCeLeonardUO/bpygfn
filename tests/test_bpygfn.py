from typing import Iterable, Iterator

import numpy as np
import numpy.typing as npt
import pytest
import torch
from gfn.states import States

from bpygfn.base import ActionList, SuperSimpleEnv
from bpygfn.quat import get_quaternion_slice, scale_state_up


@pytest.fixture
def dummy_actions():
    # Trying the step function starting from 3 instances of s_0
    return [
        [0, 1, 2],
        [2, 0, 1],
        [2, 0, 1],
    ]


@pytest.fixture
def dummy_failing_actions():
    return [0, 0, 0]


@pytest.fixture
def simple_env_test_action():
    return []


def generate_dummy_compositions(
    num_compositions: int, n: int, min_size: int = 1, max_size: int = 10
) -> Iterator[list[npt.NDArray]]:
    """
    Generate specific number of random compositions.

    Args:
        num_compositions: How many compositions to generate
        n: Length of each numpy array
        min_size: Minimum length of the list
        max_size: Maximum length of the list
    """
    for _ in range(num_compositions):
        # Random length between min_size and max_size
        list_length = np.random.randint(min_size, max_size + 1)

        # Generate list of random arrays
        composition = [np.random.random(n) for _ in range(list_length)]

        yield composition


def expected_mask_logic(state: States):
    def get_most_resent_state(composition) -> np.ndarray:
        """
        given a compostion.
        Since the state is a trajectory of actions applied to states

        each action taken is seen a direction
        each action will incriment the state's Action coll by one
        therefor

        we return the abs element wise difference
        (as it so happens that abs differnce will be a 1 for the action taken and 0 everywhere else)
        """
        if len(composition) == 0:
            return np.asarray(composition[1])
        return composition[-1] - composition[-2]

    # This function as not practival purpose other then making sure my class wont crash
    def valid_trasitions(last_action: Iterable[int]):
        last_action = np.asarray(last_action)
        # no double actions so just negate the most resent state
        return last_action ^ last_action

    dummy_mask = valid_trasitions(get_most_resent_state(state))
    return dummy_mask


@pytest.fixture
def dummy_actions_n_actions_10():
    return np.identity(10)


@pytest.fixture
def dummy_actions_list() -> ActionList:
    def print_action(state: torch.Tensor) -> torch.Tensor:
        print("hello")
        return state

    def print_action_2(state: torch.Tensor) -> torch.Tensor:
        print("hello 2")
        return state

    def print_action_3(state: torch.Tensor) -> torch.Tensor:
        print("hello 3")
        return state

    def print_action_4(state: torch.Tensor) -> torch.Tensor:
        print("hello 4")
        return state

    return {
        1: print_action,
        2: print_action_2,
        3: print_action_3,
        4: scale_state_up,
        5: get_quaternion_slice,
    }


def test_SuperSimpleEnv_step(dummy_actions_list):
    """
    Args:
        height (int): number of unique values per dimension.
        ndim (int): number of dimensions.
    """

    # each each "dimention" will have different heigh of states...
    # rotation is rotation, is will have a different height say the volume dimention
    # enumeration it the best
    # - sate dimentions are not uniform in possable values
    # - states are discrete

    env = SuperSimpleEnv(
        history_size=3, device_str="cpu", action_list=dummy_actions_list
    )

    random_states = env.reset(
        batch_shape=10,
    )

    print(random_states.tensor)
    env.preprocessor.preprocess(random_states)
    test_actions = [1, 2, 5, 5, 1, 2, 3, 4, 1, 2]

    def format_tensor(list_, discrete=True):
        """
        If discrete, returns a long tensor with a singleton batch dimension from list
        ``list_``. Otherwise, casts list to a float tensor without unsqueezing
        """
        if discrete:
            return torch.tensor(list_, dtype=torch.long).unsqueeze(-1)
        else:
            return torch.tensor(list_, dtype=torch.float)

    actions = env.actions_from_tensor(format_tensor(test_actions))
    steped_states = env.step(states=random_states, actions=actions)  # pyright: ignore
    print(steped_states)
    # preprocessed_grid = env.preprocessor.preprocess(random_states)
    """
    # this is yanked from hypergrid.py
    actions = env.actions_from_tensor(format_tensor([2, 1, 0, 1]))

    print(f"Reset states: {states.s0}")
    print(f"state Tensor: {states.tensor}")
    print(f"Actions: {actions.tensor}")

    steps = env.step(states=states, actions=actions)

    print(f"Steps: {steps}")
    """


#    for actions_list in passing_actions_lists:
#        # WHAT PATTERN IS THIS SHIT????
#        actions = env.actions_from_tensor(format_tensor(actions_list))
#
#        # this process is such a night mare
#        # 1. you have to pass in a tensor of actions, to get actions?
#        # 2. then you have to pass in states and actions BACK INTO ENV?!?!?!?
#        steps = env.step(states, actions)
#        print(steps)
#
#
#
#        # verify that step applied the logic that I want
