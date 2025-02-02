from typing import Optional, Tuple, Union

import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import Preprocessor
from gfn.states import States


class SuperSimpleEnv(DiscreteEnv):
    def __init__(
        self,
        n_actions: int,
        s0: torch.Tensor,
        state_shape: Tuple,
        height: int,
        action_shape: Tuple = (1,),
        dummy_action: Optional[torch.Tensor] = None,
        exit_action: Optional[torch.Tensor] = None,
        sf: Optional[torch.Tensor] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        self.height = height
        super().__init__(
            n_actions,
            s0,
            state_shape,
            action_shape,
            dummy_action,
            exit_action,
            sf,
            device_str,
            preprocessor,
        )

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.

        Args:
            states: The current states.i
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, *state_shape).
        """
        new_states_tensor = states.tensor.scatter(-1, actions.tensor, 1, reduce="add")
        assert new_states_tensor.shape == states.tensor.shape
        return new_states_tensor

    def is_action_valid(
        self, states: States, actions: Actions, backward: bool = False
    ) -> bool:
        return super().is_action_valid(states, actions, backward)

    def backward_step(self, states: States, actions: Actions) -> torch.Tensor:
        return super().backward_step(states, actions)

    def update_masks(self, states: States) -> None:
        return super().update_masks(states)

    def make_random_states_tensor(self, batch_shape: Tuple[int, ...]) -> torch.Tensor:
        """ this does not quite work for this env
        Random states make no sense in this case....
        make sure this is not called by the gflow net
        """
        return torch.randint(
            0, self.height, batch_shape + self.s0.shape, device=self.device
        )

    # In some cases overwritten by the user to support specific use-cases.
    def reset(
        self,
        batch_shape: Optional[Union[int, Tuple[int]]] = None,
        random: bool = False,
        sink: bool = False,
        seed: int = None,  # pyright: ignore
    ) -> States:
        """Instantiates a batch of initial states.

        `random` and `sink` cannot be both True. When `random` is `True` and `seed` is
            not `None`, environment randomization is fixed by the submitted seed for
            reproducibility.
        """
        assert not (random and sink)

        if random and seed is not None:
            torch.manual_seed(seed)  # TODO: Improve seeding here?

        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        states = self.states_from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )

        self.update_masks(states)

        return states

    # env = SuperSimpleEnvSimpleEnv(ndim=9, height=10)
    # print("SimpleEnv initialized with ndims:", env.ndim)
