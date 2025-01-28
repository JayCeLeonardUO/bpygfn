# %%
from typing import Literal

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
#%%
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
        print(states)
        return super().update_masks(states)

    def reward(self, final_states: States) -> torch.Tensor:
        """
        this will be a script from the blender_helpers lib
        """
        return super().reward(final_states)


# %%
def create_gflownet(env: DiscreteEnv, hidden_dim,n_hidden):
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

    gflownet = TBGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
    )

    return gflownet

if __name__ == "__main__":
    env = SimpleEnv(ndim=9, height=10)
    print("SimpleEnv initialized with ndims:", env.ndim)
# %%
