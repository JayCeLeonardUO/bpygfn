import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ActionRegistry:
    """Static configuration defining all possible actions and their valid values"""

    class Phase(str, Enum):
        """Phases of the generation process"""
        PARAM_SELECTION = "param_selection"
        COLOR_SELECTION = "color_selection"

    # Define all valid values for each action type
    VALID_W = [1, 2, 4, 8, 16]
    VALID_SCALE = [0.1, 0.5, 1.0, 2.0, 5.0]
    VALID_COLORS = list(range(32))

    # Action definitions
    ACTIONS = {
        'w': {
            'valid_values': VALID_W,
            'phase': Phase.PARAM_SELECTION,
            'description': 'Noise width parameter'
        },
        'scale': {
            'valid_values': VALID_SCALE,
            'phase': Phase.PARAM_SELECTION,
            'description': 'Noise scale parameter'
        },
        'color': {
            'valid_values': VALID_COLORS,
            'phase': Phase.COLOR_SELECTION,
            'description': 'Add color to ramp'
        },
        'stop': {
            'valid_values': [0],  # Dummy
            'phase': Phase.COLOR_SELECTION,
            'description': 'Terminate trajectory'
        }
    }

    @classmethod
    def get_total_actions(cls) -> int:
        """Total number of possible actions across all types"""
        return sum(len(info['valid_values']) for info in cls.ACTIONS.values())

    @classmethod
    def get_action_offset(cls, action_name: str) -> int:
        """Starting index for this action type in flat action space"""
        offset = 0
        for name, info in cls.ACTIONS.items():
            if name == action_name:
                return offset
            offset += len(info['valid_values'])
        raise ValueError(f"Action {action_name} not found")

    @classmethod
    def decode_action(cls, action_idx: int) -> Tuple[str, int]:
        """Convert flat action index to (action_name, value_index)"""
        current_offset = 0
        for name, info in cls.ACTIONS.items():
            num_values = len(info['valid_values'])
            if action_idx < current_offset + num_values:
                value_idx = action_idx - current_offset
                return name, value_idx
            current_offset += num_values
        raise ValueError(f"Invalid action index: {action_idx}")

    @classmethod
    def encode_action(cls, action_name: str, value_idx: int) -> int:
        """Convert (action_name, value_index) to flat action index"""
        offset = cls.get_action_offset(action_name)
        return offset + value_idx


class State(BaseModel):
    """
    Pydantic model for GFlowNet state.
    Represents the current state of generation and provides tensor representations.
    """

    # Parameters
    w: Optional[int] = Field(None, description="Noise width parameter")
    scale: Optional[float] = Field(None, description="Noise scale parameter")

    # Color selection
    selected_colors: List[bool] = Field(
        default_factory=lambda: [False] * len(ActionRegistry.VALID_COLORS),
        description="Binary vector of selected colors"
    )
    num_colors_selected: int = Field(0, ge=0)

    # State management
    current_phase: ActionRegistry.Phase = Field(ActionRegistry.Phase.PARAM_SELECTION)
    is_terminal: bool = Field(False)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('w')
    @classmethod
    def validate_w(cls, v):
        if v is not None and v not in ActionRegistry.VALID_W:
            raise ValueError(f"w must be one of {ActionRegistry.VALID_W}")
        return v

    @field_validator('scale')
    @classmethod
    def validate_scale(cls, v):
        if v is not None and v not in ActionRegistry.VALID_SCALE:
            raise ValueError(f"scale must be one of {ActionRegistry.VALID_SCALE}")
        return v

    @model_validator(mode='after')
    def update_phase(self):
        """Auto-transition to color selection when params are set"""
        if self.w is not None and self.scale is not None:
            self.current_phase = ActionRegistry.Phase.COLOR_SELECTION
        return self

    def to_state_tensor(self) -> torch.Tensor:
        """
        Convert to STATE tensor (network INPUT).
        Includes metadata to help network make decisions.
        """
        parts = []

        # W parameter (one-hot)
        w_onehot = torch.zeros(len(ActionRegistry.VALID_W))
        if self.w is not None:
            w_idx = ActionRegistry.VALID_W.index(self.w)
            w_onehot[w_idx] = 1
        parts.append(w_onehot)

        # Scale parameter (one-hot)
        scale_onehot = torch.zeros(len(ActionRegistry.VALID_SCALE))
        if self.scale is not None:
            scale_idx = ActionRegistry.VALID_SCALE.index(self.scale)
            scale_onehot[scale_idx] = 1
        parts.append(scale_onehot)

        # Selected colors (binary)
        parts.append(torch.tensor(self.selected_colors, dtype=torch.float32))

        # Metadata - helps network understand state context
        parts.append(torch.tensor([
            self.num_colors_selected / len(ActionRegistry.VALID_COLORS),  # Normalized count
            1.0 if self.w is not None else 0.0,  # w set?
            1.0 if self.scale is not None else 0.0,  # scale set?
            1.0 if self.current_phase == ActionRegistry.Phase.COLOR_SELECTION else 0.0  # phase
        ], dtype=torch.float32))

        return torch.cat(parts)

    def to_action_mask(self) -> torch.Tensor:
        """
        Convert to ACTION mask (valid actions for network OUTPUT).
        Derived from current state - what actions are valid?
        """
        mask = torch.zeros(ActionRegistry.get_total_actions(), dtype=torch.bool)

        if self.current_phase == ActionRegistry.Phase.PARAM_SELECTION:
            # Can set w if not already set
            if self.w is None:
                offset = ActionRegistry.get_action_offset('w')
                mask[offset:offset + len(ActionRegistry.VALID_W)] = True

            # Can set scale if not already set
            if self.scale is None:
                offset = ActionRegistry.get_action_offset('scale')
                mask[offset:offset + len(ActionRegistry.VALID_SCALE)] = True

        elif self.current_phase == ActionRegistry.Phase.COLOR_SELECTION:
            # Can select unselected colors
            color_offset = ActionRegistry.get_action_offset('color')
            for i, selected in enumerate(self.selected_colors):
                if not selected:
                    mask[color_offset + i] = True

            # Can always stop (if at least some colors selected)
            if self.num_colors_selected > 0:
                stop_offset = ActionRegistry.get_action_offset('stop')
                mask[stop_offset] = True

        return mask

    def apply_action(self, action_name: str, value_idx: int) -> 'State':
        """Apply action and return new state (immutable)"""
        value = ActionRegistry.ACTIONS[action_name]['valid_values'][value_idx]
        new_data = self.model_dump()

        if action_name == 'w':
            new_data['w'] = value
        elif action_name == 'scale':
            new_data['scale'] = value
        elif action_name == 'color':
            new_data['selected_colors'][value] = True
            new_data['num_colors_selected'] += 1
        elif action_name == 'stop':
            new_data['is_terminal'] = True

        return State(**new_data)

    @classmethod
    def get_state_tensor_dim(cls) -> int:
        """Dimension of state tensor (network INPUT)"""
        return (len(ActionRegistry.VALID_W) +
                len(ActionRegistry.VALID_SCALE) +
                len(ActionRegistry.VALID_COLORS) +
                4)  # metadata

    @classmethod
    def get_action_tensor_dim(cls) -> int:
        """Dimension of action tensor (network OUTPUT)"""
        return ActionRegistry.get_total_actions()


class ColorRampGFlowNet(nn.Module):
    """GFlowNet that reads tensor dimensions from State"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        input_dim = State.get_state_tensor_dim()
        output_dim = State.get_action_tensor_dim()

        # Build policy network
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        print(f"Initialized GFlowNet:")
        print(f"  Input dimension (state tensor):  {input_dim}")
        print(f"  Output dimension (action tensor): {output_dim}")
        print(f"  Hidden dimension: {hidden_dim}")

    def forward_policy(self, state: State) -> torch.Tensor:
        """Get action probabilities for current state"""
        # Convert state to input tensor
        state_tensor = state.to_state_tensor().unsqueeze(0)  # Add batch dim

        # Get logits from network
        with torch.no_grad():
            logits = self.policy(state_tensor).squeeze(0)  # Remove batch dim

        # Get valid action mask from state
        mask = state.to_action_mask()

        # Apply mask and convert to probabilities
        masked_logits = torch.where(mask, logits, torch.tensor(-1e9))
        probs = torch.softmax(masked_logits, dim=0)

        return probs

    def sample_action(self, state: State) -> Tuple[int, str, int, float]:
        """Sample action from policy"""
        probs = self.forward_policy(state)
        action_idx = torch.multinomial(probs, 1).item()
        action_name, value_idx = ActionRegistry.decode_action(action_idx)
        return action_idx, action_name, value_idx, probs[action_idx].item()

    def sample_trajectory(self, max_steps: int = 20) -> Tuple[List[Dict], State]:
        """Sample complete trajectory"""
        state = State()
        trajectory = []

        for step in range(max_steps):
            if state.is_terminal:
                break

            # Sample action
            action_idx, action_name, value_idx, prob = self.sample_action(state)

            # Store trajectory step
            trajectory.append({
                'state': state.model_copy(deep=True),
                'action_idx': action_idx,
                'action_name': action_name,
                'value_idx': value_idx,
                'prob': prob
            })

            # Apply action to get next state
            state = state.apply_action(action_name, value_idx)

        return trajectory, state
