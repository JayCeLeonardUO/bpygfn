# native python
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import torch
from gfn.env import DiscreteEnv


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

class testEnv(DiscreteEnv):
    pass