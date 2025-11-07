# remind pyright States
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import numpy.typing as npt
import pytest
import torch
from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.modules import MLP
from gfn.utils.training import validate
from tqdm import tqdm

from bpygfn.base import ActionList, SuperSimpleEnv
from bpygfn.quat import scale_state_down, scale_state_up

warnings.filterwarnings(
    "ignore", message="urwid.listbox is moved to urwid.widget.listbox"
)


@dataclass
class TrainingArgs:
    no_cuda: bool = False
    ndim: int = 4
    height: int = 16
    seed: int = 0
    lr: float = 1e-3
    lr_logz: float = 1e-1
    n_iterations: int = 1000
    validation_interval: int = 100
    validation_samples: int = 100000
    batch_size: int = 16
    epsilon: float = 0.1


@pytest.fixture
def training_args(request) -> TrainingArgs:
    """
    Fixture that provides training arguments with default values.
    Can be overridden using @pytest.mark.parametrize or by passing custom values.

    Example usage:
        def test_training(training_args):
            assert training_args.batch_size == 16

        @pytest.mark.parametrize('training_args',
                               [{'batch_size': 32, 'lr': 1e-4}],
                               indirect=True)
        def test_custom_training(training_args):
            assert training_args.batch_size == 32
            assert training_args.lr == 1e-4
    """
    # Get custom parameters if provided through parametrize
    if hasattr(request, "param"):
        return TrainingArgs(**{**vars(TrainingArgs()), **request.param})
    return TrainingArgs()


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
        print("hello 0")
        return state

    def print_action_1(state: torch.Tensor) -> torch.Tensor:
        print("hello 1")
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
        0: print_action,
        1: print_action_1,
        2: print_action_2,
        3: scale_state_down,
        4: scale_state_up,
    }


@pytest.fixture
def Blender_actions() -> ActionList:
    def print_action(state: torch.Tensor) -> torch.Tensor:
        print("hello 0")
        return state

    def print_action_1(state: torch.Tensor) -> torch.Tensor:
        print("hello 1")
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
        0: print_action,
        1: print_action_1,
        2: print_action_2,
        3: scale_state_down,
        4: scale_state_up,
    }


@pytest.fixture
def dummy_env(dummy_actions_list):
    return SuperSimpleEnv(
        history_size=3, device_str="cpu", action_list=dummy_actions_list
    )


def format_tensor(list_, discrete=True):
    """
    If discrete, returns a long tensor with a singleton batch dimension from list
    ``list_``. Otherwise, casts list to a float tensor without unsqueezing
    """
    if discrete:
        return torch.tensor(list_, dtype=torch.long).unsqueeze(-1)
    else:
        return torch.tensor(list_, dtype=torch.float)


@pytest.fixture
def args_blah():
    return


def test_SuperSimpleEnv_step(dummy_actions_list, dummy_env):  # {{{
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
    env.preprocessor.preprocess(random_states)
    test_actions = [1, 2, 0, 0, 1, 2, 3, 4, 1, 2]

    actions = env.actions_from_tensor(format_tensor(test_actions))
    # how do i check that the states what where called here are right?
    steped_states = env._step(states=random_states, actions=actions)  # pyright: ignore
    # valid_masks = torch.tensor(
    #     [
    #         [True, False, True, True, True],
    #         [True, True, False, True, True],
    #         [False, True, True, True, True],
    #         [False, True, True, True, True],
    #         [True, False, True, True, True],
    #         [True, True, False, True, True],
    #         [True, True, True, False, True],
    #         [False, False, False, False, False],
    #         [True, False, True, True, True],
    #         [True, True, False, True, True],
    #     ]
    # )
    print(env.reward(steped_states))  # }}}

    # assert valid_masks == steped_states.forward_masks  # pyright: ignore


def test_training(training_args, dummy_env):  # {{{
    args = training_args
    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Setup the Environment.
    env = dummy_env
    # Build the GFlowNet.
    module_PF = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    module_PB = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions - 1,
        trunk=module_PF.trunk,
    )
    pf_estimator = DiscretePolicyEstimator(
        module_PF, env.n_actions, is_backward=False, preprocessor=env.preprocessor
    )
    pb_estimator = DiscretePolicyEstimator(
        module_PB, env.n_actions, is_backward=True, preprocessor=env.preprocessor
    )
    gflownet = TBGFlowNet(pf=pf_estimator, pb=pb_estimator, logZ=0.0)

    # Feed pf to the sampler.
    sampler = Sampler(estimator=pf_estimator)

    # Move the gflownet to the GPU.
    gflownet = gflownet.to(device_str)

    # Policy parameters have their own LR. Log Z gets dedicated learning rate
    # (typically higher).
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=args.lr)
    optimizer.add_param_group(
        {"params": gflownet.logz_parameters(), "lr": args.lr_logz}
    )

    validation_info = {"l1_dist": float("inf")}
    visited_terminating_states = env.states_from_batch_shape((0,))
    for it in (pbar := tqdm(range(args.n_iterations), dynamic_ncols=True)):  # pyright: ignore
        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )
        visited_terminating_states.extend(trajectories.last_states)

        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        if (it + 1) % args.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )
            print(f"Iter {it + 1}: L1 distance {validation_info['l1_dist']:.8f}")
        pbar.set_postfix({"loss": loss.item()})  # }}}


def test_hypergridbmesh():
    print(" What is going on")
