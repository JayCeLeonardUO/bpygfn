from typing import Iterable, Iterator, List

import numpy as np
import numpy.typing as npt
import pytest
import torch
from gfn.preprocessors import EnumPreprocessor
from gfn.states import DiscreteStates, States

from bpygfn.base import SuperSimpleEnv


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


def test_SuperSimpleEnv_step(dummy_actions_n_actions_10):
    NDIM = 4
    HEIGHT = 3
    BATCH_SIZE = 3
    SEED = 42

    def test_actions_and_states(dummy_actions_n_actions_10):
        N_ACTIONS = 10
        BATCH_SHAPE = 10

        dummy_comps: List[List[int]] = [[1, 0, 0], [1, 1, 0], [2, 1, 0]]

        # Convert list to tensor
        tensor: torch.Tensor = torch.tensor(dummy_comps)

        ds = DiscreteStates(tensor)

        # succsesfully sep
        return ds

    def get_states_indices(states: States) -> torch.Tensor:
        """Get the indices of the states in the canonical ordering.

        Args:
            states: The states to get the indices of.

        Returns the indices of the states in the canonical ordering as a tensor of shape `batch_shape`.
        """
        states_raw = states.tensor

        canonical_base = HEIGHT ** torch.arange(
            NDIM - 1, -1, -1, device=states_raw.device
        )

        indices = (canonical_base * states_raw).sum(-1).long()
        assert indices.shape == states.batch_shape
        return indices

    preproc = EnumPreprocessor(
        get_states_indices=get_states_indices,
    )

    s0 = torch.zeros(NDIM, dtype=torch.long, device=torch.device("cpu"))
    env = SuperSimpleEnv(
        n_actions=4,
        s0=s0,
        state_shape=(NDIM,),
        preprocessor=preproc,
        height=HEIGHT,
        device_str="cpu",
    )

    # Utilities.
    def format_tensor(list_, discrete=True):
        """
        If discrete, returns a long tensor with a singleton batch dimension from list
        ``list_``. Otherwise, casts list to a float tensor without unsqueezing
        """
        if discrete:
            return torch.tensor(list_, dtype=torch.long).unsqueeze(-1)
        else:
            return torch.tensor(list_, dtype=torch.float)

    # Testing the backward method from a batch of random (seeded) state.
    states = env.reset(
        batch_shape=(NDIM),
    )

    # this is yanked from hypergrid.py
    # s0 = torch.zeros(NDIM, dtype=torch.long, device=torch.device("cuda"))
    actions = env.actions_from_tensor(format_tensor([2, 1, 0, 1]))

    print(f"\nstate 0: {s0}")
    print(f"Reset states: {states.s0}")
    print(f"state Tensor: {states.tensor}")
    print(f"Actions: {actions.tensor}")

    steps = env.step(states=states, actions=actions)

    print(f"Steps: {steps}")


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
