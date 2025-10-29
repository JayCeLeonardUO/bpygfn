from typing import List, Callable, Dict, Any, Optional, Literal
import torch
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum


class EncodingScheme(Enum):
    """Encoding schemes for action groups"""
    ONE_HOT = "one_hot"  # Standard one-hot encoding
    FACTORIZED = "factorized"  # Multi-hot with separate dimensions per factor
    BINARY = "binary"  # Binary encoding (for future use)


@dataclass
class ActionGroup:
    """Defines a group of related actions with shared validation and encoding"""
    name: str
    validator: Optional[Callable] = None
    encoding_scheme: EncodingScheme = EncodingScheme.ONE_HOT
    description: str = ""


class ActionRegistry:
    """Registry for action functions with group-based validation and encoding"""

    def __init__(self):
        self.action_functions = {}
        self.action_types = {}  # Maps action_idx -> action_type
        self.action_groups = {}  # Maps action_idx -> group_name

        self.action_type_info = defaultdict(lambda: {
            'indices': [],
            'count': 0,
            'validator': None,
            'group': None,
            'encoding_scheme': EncodingScheme.ONE_HOT
        })

        self.groups = {}  # Maps group_name -> ActionGroup
        self.current_index = 0

    def register_group(self, group: ActionGroup):
        """Register an action group with its validator and encoding scheme"""
        self.groups[group.name] = group
        print(f"  Registered group: '{group.name}' - {group.description} [{group.encoding_scheme.value}]")

    def add_actions(
            self,
            values: List[Any],
            action_type: str,
            group: Optional[str] = None,
            validator: Optional[Callable] = None
    ):
        """
        Decorator to register simple actions with values
        Uses encoding scheme from group if specified
        """

        def decorator(func: Callable):
            # Get encoding scheme from group
            encoding_scheme = EncodingScheme.ONE_HOT
            if group and group in self.groups:
                encoding_scheme = self.groups[group].encoding_scheme

            action_indices = []

            for value in values:
                action_idx = self.current_index
                self.action_functions[action_idx] = lambda v=value: func(v)
                self.action_types[action_idx] = action_type
                if group:
                    self.action_groups[action_idx] = group
                action_indices.append(action_idx)
                self.current_index += 1

            # Determine validator
            final_validator = validator
            if final_validator is None and group and group in self.groups:
                final_validator = self.groups[group].validator

            # Store metadata
            self.action_type_info[action_type]['indices'] = action_indices
            self.action_type_info[action_type]['count'] = len(action_indices)
            self.action_type_info[action_type]['validator'] = final_validator
            self.action_type_info[action_type]['group'] = group
            self.action_type_info[action_type]['values'] = values
            self.action_type_info[action_type]['encoding_scheme'] = encoding_scheme

            return func

        return decorator

    def add_parameterized_actions(
            self,
            params: Dict[str, List[Any]],
            action_type: str,
            group: Optional[str] = None,
            validator: Optional[Callable] = None
    ):
        """
        Decorator for parameterized actions
        Uses encoding scheme from group:
        - ONE_HOT: Creates one action per combination (standard)
        - FACTORIZED: Creates separate dimensions per parameter (multi-hot)
        """

        def decorator(func: Callable):
            # Get encoding scheme from group
            encoding_scheme = EncodingScheme.ONE_HOT
            if group and group in self.groups:
                encoding_scheme = self.groups[group].encoding_scheme

            if encoding_scheme == EncodingScheme.FACTORIZED:
                return self._add_factorized(params, action_type, group, validator, func)
            else:
                return self._add_one_hot_parameterized(params, action_type, group, validator, func)

        return decorator

    def sample_from(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Sample a valid action tensor from the mask

        Args:
            mask: Boolean mask of valid actions

        Returns:
            Action tensor with appropriate hot bits set
        """
        action_tensor = torch.zeros(self.total_actions)

        # Find which action types have valid dimensions
        for action_type, info in self.action_type_info.items():
            indices = info['indices']
            type_mask = mask[indices]

            if type_mask.sum() == 0:
                continue  # No valid actions of this type

            encoding = info['encoding_scheme']

            if encoding == EncodingScheme.ONE_HOT:
                # Sample one index from valid indices
                valid_indices = torch.nonzero(type_mask, as_tuple=True)[0]
                local_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                global_idx = indices[local_idx]
                action_tensor[global_idx] = 1.0
                return action_tensor  # Return immediately for one-hot

            elif encoding == EncodingScheme.FACTORIZED:
                # Sample one value per factor
                factor_ranges = info['factor_ranges']
                can_sample = True

                for param_name, range_info in factor_ranges.items():
                    start = range_info['start']
                    end = range_info['end']
                    factor_mask = mask[start:end]

                    if factor_mask.sum() == 0:
                        can_sample = False
                        break

                    # Sample one value from this factor
                    valid_factor_indices = torch.nonzero(factor_mask, as_tuple=True)[0]
                    local_idx = valid_factor_indices[torch.randint(len(valid_factor_indices), (1,))].item()
                    action_tensor[start + local_idx] = 1.0

                if can_sample:
                    return action_tensor

        raise ValueError("Could not sample valid action from mask")
    def _add_one_hot_parameterized(self, params, action_type, group, validator, func):
        """Add parameterized actions with one-hot encoding"""
        action_indices = []
        param_combinations = []

        import itertools
        param_names = list(params.keys())
        param_values_list = list(params.values())

        for combination in itertools.product(*param_values_list):
            param_dict = dict(zip(param_names, combination))
            action_idx = self.current_index

            self.action_functions[action_idx] = lambda pd=param_dict: func(**pd)
            self.action_types[action_idx] = action_type
            if group:
                self.action_groups[action_idx] = group
            action_indices.append(action_idx)
            param_combinations.append(param_dict)
            self.current_index += 1

        # Determine validator
        final_validator = validator
        if final_validator is None and group and group in self.groups:
            final_validator = self.groups[group].validator

        # Store metadata
        self.action_type_info[action_type]['indices'] = action_indices
        self.action_type_info[action_type]['count'] = len(action_indices)
        self.action_type_info[action_type]['validator'] = final_validator
        self.action_type_info[action_type]['group'] = group
        self.action_type_info[action_type]['params'] = params
        self.action_type_info[action_type]['combinations'] = param_combinations
        self.action_type_info[action_type]['encoding_scheme'] = EncodingScheme.ONE_HOT

        return func

    def _add_factorized(self, params, action_type, group, validator, func):
        """Add parameterized actions with factorized (multi-hot) encoding"""
        # Calculate index ranges for each parameter/factor
        factor_ranges = {}
        current_start = self.current_index

        for param_name, param_values in params.items():
            factor_ranges[param_name] = {
                'start': current_start,
                'end': current_start + len(param_values),
                'values': param_values
            }
            current_start += len(param_values)

        total_indices = list(range(self.current_index, current_start))

        # Mark all indices
        for idx in total_indices:
            self.action_types[idx] = action_type
            if group:
                self.action_groups[idx] = group

        # Generate all combinations for validation
        import itertools
        param_names = list(params.keys())
        param_values_list = list(params.values())
        action_combinations = []
        for combination in itertools.product(*param_values_list):
            param_dict = dict(zip(param_names, combination))
            action_combinations.append(param_dict)

        # Determine validator
        final_validator = validator
        if final_validator is None and group and group in self.groups:
            final_validator = self.groups[group].validator

        # Store metadata
        self.action_type_info[action_type]['indices'] = total_indices
        self.action_type_info[action_type]['count'] = len(total_indices)
        self.action_type_info[action_type]['validator'] = final_validator
        self.action_type_info[action_type]['group'] = group
        self.action_type_info[action_type]['params'] = params
        self.action_type_info[action_type]['factor_ranges'] = factor_ranges
        self.action_type_info[action_type]['combinations'] = action_combinations
        self.action_type_info[action_type]['function'] = func
        self.action_type_info[action_type]['encoding_scheme'] = EncodingScheme.FACTORIZED

        self.current_index = current_start

        return func

    def factorized_tensor_to_params(self, action_tensor: torch.Tensor, action_type: str) -> dict:
        """Convert factorized tensor to parameters"""
        info = self.action_type_info[action_type]
        factor_ranges = info['factor_ranges']

        params = {}
        for param_name, range_info in factor_ranges.items():
            start = range_info['start']
            end = range_info['end']
            values = range_info['values']

            factor_slice = action_tensor[start:end]
            hot_idx = torch.argmax(factor_slice).item()
            params[param_name] = values[hot_idx]

        return params

    def __getitem__(self, action_tensor: torch.Tensor) -> Callable:
        """Get action function from tensor (handles all encoding schemes)"""
        # Check factorized actions first
        for atype, info in self.action_type_info.items():
            if info['encoding_scheme'] == EncodingScheme.FACTORIZED:
                indices = info['indices']
                if len(indices) > 0 and action_tensor[indices[0]:indices[-1] + 1].sum() > 0:
                    params = self.factorized_tensor_to_params(action_tensor, atype)
                    func = info['function']
                    return lambda: func(**params)

        # Otherwise one-hot
        action_idx = torch.argmax(action_tensor).item()

        if action_idx not in self.action_functions:
            raise ValueError(f"Invalid action index: {action_idx}")

        return self.action_functions[action_idx]

    def get_action_mask(self) -> torch.Tensor:
        """Generate action mask (handles all encoding schemes)"""
        mask = torch.zeros(self.total_actions, dtype=torch.bool)

        for action_type, info in self.action_type_info.items():
            validator = info['validator']
            encoding = info['encoding_scheme']

            if encoding == EncodingScheme.FACTORIZED:
                if validator is None:
                    for idx in info['indices']:
                        mask[idx] = True
                else:
                    valid_combinations = validator()

                    if isinstance(valid_combinations, bool):
                        for idx in info['indices']:
                            mask[idx] = valid_combinations
                    elif isinstance(valid_combinations, list):
                        factor_ranges = info['factor_ranges']

                        for param_name, range_info in factor_ranges.items():
                            start = range_info['start']
                            values = range_info['values']

                            for i, value in enumerate(values):
                                for valid_combo in valid_combinations:
                                    if isinstance(valid_combo, dict):
                                        if valid_combo.get(param_name) == value:
                                            mask[start + i] = True
                                            break
            else:  # ONE_HOT
                if validator is None:
                    for idx in info['indices']:
                        mask[idx] = True
                else:
                    is_valid = validator()

                    if isinstance(is_valid, bool):
                        for idx in info['indices']:
                            mask[idx] = is_valid
                    elif isinstance(is_valid, (list, torch.Tensor)):
                        for idx, valid in zip(info['indices'], is_valid):
                            mask[idx] = valid

        return mask

    @property
    def total_actions(self) -> int:
        return self.current_index

    def get_action_space_summary(self) -> Dict[str, Any]:
        """Get summary organized by groups with encoding info"""
        summary = {
            'total_dimensions': self.total_actions,
            'groups': {}
        }

        grouped_types = defaultdict(list)
        for action_type, info in self.action_type_info.items():
            group = info['group'] or 'ungrouped'
            grouped_types[group].append({
                'type': action_type,
                'count': info['count'],
                'encoding': info['encoding_scheme'].value
            })

        for group_name, types in grouped_types.items():
            total = sum(t['count'] for t in types)
            summary['groups'][group_name] = {
                'action_types': types,
                'total': total
            }

        return summary