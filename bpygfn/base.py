# %%
from inspect import indentsize
from typing import Literal

from gfn import preprocessors
import torch
from gfn.actions import Actions
from gfn.containers import Trajectories, Transitions
from gfn.env import DiscreteEnv
from gfn.modules import DiscretePolicyEstimator
from gfn.preprocessors import EnumPreprocessor
from gfn.states import DiscreteStates, States
from gfn.utils.modules import DiscreteUniform, Tabular
from torch import TensorType, nn
from gfn.gflownet import TBGFlowNet
from gfn.utils.modules import MLP


# %%
class SimpleEnv(DiscreteEnv):
    """
    holds all the information I will 'apparently' need for
    """

    def __init__(
        self,
        ndim: int,
        height: int,
        sf=None,
        preprocessor=None,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"] = "KHot",
    ):
        self.ndim = 10
        self.height = height
        s0 = torch.zeros(ndim, dtype=torch.long, device=torch.device(device_str))
        device_str = "cuda"
        torch.zeros(self.ndim, dtype=torch.long, device=torch.device(device_str))
        n_actions = ndim + 1
        super().__init__(
            n_actions=n_actions,
            s0=s0,  # type: ignore
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def get_states_indices(self, states: DiscreteStates) -> torch.Tensor:
        """
        for some reason I have to set the behavior of states in the env class .... wtf
        """
        states_raw = states.tensor
        canonical_base = self.height ** torch.arange(
            self.ndim - 1, -1, -1, device=states_raw.device
        )
        indices = (canonical_base * states_raw).sum(-1).long()
        assert indices.shape == states.batch_shape
        return indices

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """
        Args:
            states: The current states.
            actions: The actions to take
        """

        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        assert new_states_tensor == states.tensor.shape
        return new_states_tensor

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        return super().backward_step(states, actions)

    def update_masks(self, states: States) -> None:
        """
        based on the current mask. mask out invalid selections for step
        """
        # update the forwards and backwards masks here
        return super().update_masks(states)

    def reward(self, final_states: States) -> torch.Tensor:
        """
        this will be a script from the blender_helpers lib
        """
        return super().reward(final_states)

#%%
def examples_states(
    height: int = 10, ndim: int = 10, device: torch.device = torch.device("cpu")
):
    """
    This will be a constructions of 7 numbers
    think of the addition of a state as a step is a direction in a hyper grid
    This is why I will have to calculate masks for each step
    """
    # construct some states to pass into preproc
    
    # are teh trajectories not an array of steps
    canonical_base = height ** torch.arange(ndim - 1, -1, -1, device=device)

    # so is this not enumeration the actions?
    def preproc_fn(states: DiscreteStates) -> torch.Tensor:
        data = states.tensor
        indices = (canonical_base * data).sum(-1).long()
        return indices 

    def reward(final_states):
        pass
        # rewards a batch of trajectories
        # reward bands

        """
        cos
            R0 + # base
            (0.25 < ax) * R1 prod mastk out ones that dont meet critieria
            ((0.3<ax)) * (ax *) ... second ring
        In the cosine case we are rewarding
        """
    """
    myStates = ["one", "two", "three"]
    n_actions = len(myStates)
    device_str = "cpu"
    device = torch.device(device_str)
    s0 = []  # source state
    sf = [-1]  # I assme this is the sink
    # EnumPreprocessor
    """

# %%
def create_gflownet(env: SimpleEnv, hidden_dim, n_hidden):
    """
    - gflownet needs
    - forward prob
    - backwards -  prob
    - state information
    """

    def get_states_indices(states: DiscreteStates) -> torch.Tensor:
        return torch.tensor([-1])

    # i guess pyright is not seeing descrete states inherit from states
    preprocessor = EnumPreprocessor(get_states_indices=get_states_indices)  # pyright: ignore

    pf_module = MLP(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
    )

    pb_module = DiscreteUniform(env.n_actions - 1)

    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )

    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=env.preprocessor,
    )
    # env is not even used in the loss??!?!? then what is the point
    gflownet = TBGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
    )

    return gflownet


if __name__ == "__main__":
    env = SimpleEnv(ndim=9, height=10)
    print("SimpleEnv initialized with ndims:", env.ndim)
# %%
