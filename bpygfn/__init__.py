# native python
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# external
import bpy
import numpy as np
import torch


@dataclass
class ActionEncoder:
    max_sequence_length: int = 30
    action_vocab: Optional[List[str]] = None
    action_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_action: Dict[int, str] = field(default_factory=dict)
    positional_dim: int = field(default=4)  # Added positional dimension parameter

    def __post_init__(self):
        if self.action_vocab:
            self.build_vocab(self.action_vocab)

    def build_vocab(self, actions: List[str]) -> None:
        unique_actions = sorted(set(actions))
        self.action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.action_to_idx)

    def one_hot_encode(self, action_sequence: List[str]) -> torch.Tensor:
        if not self.action_to_idx:
            self.build_vocab(action_sequence)

        encoded = torch.zeros(self.max_sequence_length, self.vocab_size)

        for t, action in enumerate(action_sequence[: self.max_sequence_length]):
            if action in self.action_to_idx:
                encoded[t, self.action_to_idx[action]] = 1

        return encoded

    def integer_encode(self, action_sequence: List[str]) -> torch.Tensor:
        if not self.action_to_idx:
            self.build_vocab(action_sequence)

        encoded = torch.full((self.max_sequence_length,), -1)
        sequence_length = min(len(action_sequence), self.max_sequence_length)

        for t, action in enumerate(action_sequence[:sequence_length]):
            if action in self.action_to_idx:
                encoded[t] = self.action_to_idx[action]

        return encoded

    def positional_encode(self, action_sequence: List[str]) -> torch.Tensor:
        one_hot = self.one_hot_encode(action_sequence)

        # Create positional encoding with matching dimensions
        position = torch.arange(
            0, self.max_sequence_length, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.positional_dim, 2).float()
            * (-np.log(10000.0) / self.positional_dim)
        )
        pos_encoding = torch.zeros(self.max_sequence_length, self.positional_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return torch.cat([one_hot, pos_encoding], dim=1)

    def decode(self, encoded_sequence: torch.Tensor) -> List[str]:
        return [
            self.idx_to_action[idx.item()]
            for idx in encoded_sequence
            if idx.item() != -1 and idx.item() in self.idx_to_action
        ]
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

class StateFlowModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build the flow network
        layers = [
            nn.Linear(self.state_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ]
        
        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            
        layers.extend([
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        ])
        
        self.flow_network = nn.Sequential(*layers)
        
    def compute_pairwise_flows(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute flows between all pairs of states in a trajectory.
        
        Args:
            states: Tensor of shape (sequence_length, state_dim)
            
        Returns:
            flow_matrix: Tensor of shape (sequence_length, sequence_length) containing flow values
            state_pairs: Tensor of shape (sequence_length * (sequence_length-1), 2, state_dim)
        """
        seq_length = states.size(0)
        
        # Create all possible pairs of states
        i, j = torch.triu_indices(seq_length, seq_length, offset=1)
        state_pairs = torch.stack([states[i], states[j]], dim=1)
        
        # Concatenate state pairs for network input
        network_input = state_pairs.view(-1, self.state_dim * 2)
        
        # Compute flows
        flows = self.flow_network(network_input)
        
        # Create flow matrix
        flow_matrix = torch.zeros((seq_length, seq_length))
        flow_matrix[i, j] = flows.squeeze()
        flow_matrix = flow_matrix + flow_matrix.t()  # Make symmetric
        
        return flow_matrix, state_pairs
    
    def get_trajectory_flows(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze flows in a trajectory and return various flow metrics.
        
        Args:
            states: Tensor of shape (sequence_length, state_dim)
            
        Returns:
            Dictionary containing:
                - flow_matrix: Pairwise flows between states
                - total_flow: Sum of all flows
                - forward_flow: Flows to future states
                - backward_flow: Flows to past states
                - local_flow: Flows between adjacent states
        """
        flow_matrix, _ = self.compute_pairwise_flows(states)
        seq_length = states.size(0)
        
        # Calculate different types of flows
        total_flow = flow_matrix.sum() / 2  # Divide by 2 since matrix is symmetric
        
        # Forward and backward flows
        triu_indices = torch.triu_indices(seq_length, seq_length, offset=1)
        forward_flow = flow_matrix[triu_indices[0], triu_indices[1]].sum()
        backward_flow = flow_matrix[torch.tril_indices(seq_length, seq_length, offset=-1)].sum()
        
        # Local flows (between adjacent states)
        local_indices = torch.arange(seq_length-1)
        local_flow = flow_matrix[local_indices, local_indices+1].sum()
        
        return {
            'flow_matrix': flow_matrix,
            'total_flow': total_flow,
            'forward_flow': forward_flow,
            'backward_flow': backward_flow,
            'local_flow': local_flow
        }
    
    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the flow model.
        
        Args:
            states: Tensor of shape (sequence_length, state_dim)
            
        Returns:
            Flow metrics dictionary
        """
        return self.get_trajectory_flows(states)

# Example usage
def example_usage():
    # Create sample trajectory
    sequence_length = 10
    state_dim = 8
    states = torch.randn(sequence_length, state_dim)
    
    # Initialize model
    flow_model = StateFlowModel(state_dim=state_dim)
    
    # Get flows
    flow_metrics = flow_model(states)
    
    return flow_metrics