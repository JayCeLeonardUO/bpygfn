# %%
from typing import Literal
from gfn.states import States
import torch
from gfn.containers import Trajectories, Transitions
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import EnumPreprocessor
from torch import TensorType, nn


class SimpleEnv(DiscreteEnv):
    def __init__(
        self,
        ndim: int,
        n_actions,
        sf=None,
        preprocessor=None,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot", "OneHot", "Identity", "Enum"] = "KHot",
    ):
        self.ndim = 10
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
        
    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """
        
        """
        return super().step(states, actions)
    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        return super().backward_step(states, actions)
    def update_masks(self, states: States) -> None:
        """
        based on the current mast. mask out invalid selections for step
        """
        return super().update_masks(states)

"""
testing strings
"""
if __name__ == "__main__":
    env = SimpleEnv(ndim=10, n_actions=11)
    print("SimpleEnv initialized with ndims:", env.ndim)

# %%


class SimpleSmileFlowModel(nn.Module):
    """
    - 6 layers (one for each face part)
    """

    def __init__(self, num_hid, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(6, num_hid), nn.LeakyReLU(), nn.Linear(num_hid, 6)
        )

    def forward(self, x):
        return self.mlp(x).exp()


class SimpleSequenceFlowModel(nn.Module):
    """
    the seq_len would be arbitraty
    """

    def __init__(self, num_hid, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimpleTransitions(Transitions):
    def __init__(
        self,
        env,
        states=None,
        actions=None,
        is_done=None,
        next_states=None,
        is_backward=False,
        log_rewards=None,
        log_probs=None,
    ):
        super().__init__(
            env,
            states,
            actions,
            is_done,
            next_states,
            is_backward,
            log_rewards,
            log_probs,
        )


class SimpleTrajectories(Trajectories):
    def __init__(
        self,
        env,
        states=None,
        actions=None,
        when_is_done=None,
        is_backward=False,
        log_rewards=None,
        log_probs=None,
    ):
        super().__init__(
            env, states, actions, when_is_done, is_backward, log_rewards, log_probs
        )
