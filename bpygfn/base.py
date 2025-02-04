from collections import defaultdict
from typing import Optional, Tuple

import torch
from gfn.actions import Actions
from gfn.env import DiscreteEnv
from gfn.preprocessors import EnumPreprocessor
from gfn.states import DiscreteStates, States


def my_rotate(target):
    print(target)
    return


"""
class MyState:
    roation = ["i", "j", "k"]
    actionHistory: List["MyState"] = []

    def __init__(self):
        return

    @property
    def to_tensor(self):
        return torch.Tensor([-1])
"""

"""
When a key that is not in the dict is called it will just result in the
Exit action
"""


def temp_exit():
    return None


# dynamic list of pointers to actions
# TODO: this is a global ... fix it
actions_dictionary: dict = defaultdict(
    temp_exit, {"printhello": lambda: print("hello"), "rot": my_rotate}
)


class myAction(DiscreteStates):
    def __init__(self):
        pass


# this will be the fn that the preproc class gets
def my_calc_ind(states: States) -> torch.Tensor:
    print(states)
    return torch.Tensor([-1])


class SuperSimpleEnv(DiscreteEnv):
    # should take in the actions dict
    def __init__(
        self,
        device_str: Optional[str] = None,
    ):
        """
        Args:
            TODO - n_actions is hard coded... fix it
            dummy_action, I have no idea what this is or why it is here
        """

        preprocessor = EnumPreprocessor(self.get_states_indices)
        # should not_ be passed in
        state_shape: Tuple = (1,)

        # this will be relevent
        # exit_action = torch.Tensor([-1])  # actions_dictionary['exit']

        # value for undefined actions?
        # dummy_action: torch.Tensor = torch.Tensor([-1])

        s0 = torch.Tensor([0])
        sf = torch.Tensor([0])

        super().__init__(
            n_actions=len(actions_dictionary),
            s0=s0,
            state_shape=state_shape,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def make_random_states_tensor(self, batch_shape: Tuple) -> torch.Tensor:
        """Optional method inherited by all States instances to emit a random tensor."""
        print("this was called")
        return torch.Tensor([-1])

    import pudb

    pudb.set_trace()

    # there is no way around it... I have to do the make indecies and make class fn's FML
    def get_states_indices(self, states: States) -> torch.Tensor:
        """Get the indices of the states in the canonical ordering.

        Args:
            states: The states to get the indices of.

        Returns the indices of the states in the canonical ordering as a tensor of shape `batch_shape`.
        """
        # is this
        raise NotImplementedError
        return torch.Tensor([-1])

    def step(self, states: States, actions: Actions) -> torch.Tensor:
        """Take a step in the environment.
        The Actions in this case will be a tensor of operations to apply to the batch

        As i Under stand it,

        - Step is trajectory egnostic

        - the trainin proses is applying actions to states and seeing which ones lead to the final step

        TODO for now just to quaternions and scalling volume direcly

        Args:
            states: The current states.i
            actions: The actions to take.

        Returns the new states after taking the actions as a tensor of shape (*batch_shape, *state_shape).
        """
        raise NotImplementedError
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

    """ 
    #from the Debuger I was not able to find any where this i called other then testing
    def make_random_states_tensor(self, batch_shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randint(
            0, self.height, batch_shape + self.s0.shape, device=self.device
        )
    """
