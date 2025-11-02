from typing import Callable, Dict, Any, Union
import torch
import numpy as np
from scipy import ndimage


class RewardRegistry:
    """Simple registry to map strings to reward functions"""

    def __init__(self):
        self._functions: Dict[str, Callable] = {}

    def register(self, name: str):
        """
        Decorator to register a reward function by name

        Usage:
            @reward_registry.register('no_holes')
            def detect_holes_reward(tensor, threshold=0.5):
                return reward_value
        """

        def decorator(func: Callable):
            self._functions[name] = func
            return func

        return decorator

    def __getitem__(self, name: str) -> Callable:
        """Get function by name: reward_registry['no_holes']"""
        if name not in self._functions:
            available = list(self._functions.keys())
            raise KeyError(f"Reward '{name}' not found. Available: {available}")
        return self._functions[name]

    def __call__(self, name: str) -> Callable:
        """Get function by call: reward_registry('no_holes')"""
        return self[name]

    def list_rewards(self) -> list:
        """List all registered reward names"""
        return list(self._functions.keys())


# Global registry
reward_registry = RewardRegistry()


# ============================================================================
# Register Reward Functions
# ============================================================================

@reward_registry.register('no_holes')
def detect_holes_reward(
        tensor: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
) -> float:
    """Reward for terrain without holes"""
    if torch.is_tensor(tensor):
        noise = tensor.numpy()
    else:
        noise = tensor

    binary_mask = noise > threshold
    filled = ndimage.binary_fill_holes(binary_mask)
    has_holes = (filled.sum() - binary_mask.sum()) > 0

    return 0.0 if has_holes else 1.0


@reward_registry.register('height_variance')
def height_variance_reward(
        tensor: Union[np.ndarray, torch.Tensor],
        target_variance: float = 0.1,
        tolerance: float = 0.05
) -> float:
    """Reward based on height variance"""
    if torch.is_tensor(tensor):
        variance = tensor.var().item()
    else:
        variance = np.var(tensor)

    deviation = abs(variance - target_variance)

    if deviation <= tolerance:
        return 1.0
    else:
        return max(0.0, 1.0 - (deviation - tolerance) / target_variance)


@reward_registry.register('smoothness')
def smoothness_reward(
        tensor: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.1
) -> float:
    """Reward for smooth terrain"""
    if torch.is_tensor(tensor):
        tensor_np = tensor.numpy()
    else:
        tensor_np = tensor

    grad_y, grad_x = np.gradient(tensor_np)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    avg_gradient = grad_magnitude.mean()

    if avg_gradient <= threshold:
        return 1.0
    else:
        return max(0.0, 1.0 - (avg_gradient - threshold) / threshold)


@reward_registry.register('height_range')
def height_range_reward(
        tensor: Union[np.ndarray, torch.Tensor],
        min_height: float = -1.0,
        max_height: float = 1.0
) -> float:
    """Reward for keeping heights within range"""
    if torch.is_tensor(tensor):
        min_val = tensor.min().item()
        max_val = tensor.max().item()
    else:
        min_val = np.min(tensor)
        max_val = np.max(tensor)

    min_violation = max(0.0, min_height - min_val)
    max_violation = max(0.0, max_val - max_height)
    total_violation = min_violation + max_violation

    if total_violation == 0:
        return 1.0
    else:
        return max(0.0, 1.0 - total_violation)


# ============================================================================
# Usage Examples
# ============================================================================

def test_simple_usage():
    """Test simple string -> function pointer usage"""

    from single_color_ramp import sample_random_trajectory

    print("\n" + "=" * 70)
    print("TESTING SIMPLE REWARD USAGE")
    print("=" * 70)

    # Sample trajectory
    trajectory = sample_random_trajectory(trajectory_len=3, max_colors=5)
    heightmap = trajectory['heightmaps'][-1]

    # Example 1: Using config string
    config = {
        "reward": "no_holes",
        "threshold": 0.5
    }

    print("\n1. Using config dictionary:")
    print(f"   Config: {config}")

    # Get function from string
    reward_fn = reward_registry[config["reward"]]
    reward = reward_fn(heightmap, threshold=config["threshold"])

    print(f"   Reward: {reward:.3f}")

    # Example 2: Direct usage
    print("\n2. Direct usage:")
    reward = reward_registry['smoothness'](heightmap)
    print(f"   smoothness reward: {reward:.3f}")

    # Example 3: Using __call__
    print("\n3. Using call syntax:")
    reward = reward_registry('height_range')(heightmap, min_height=-2.0, max_height=2.0)
    print(f"   height_range reward: {reward:.3f}")

    # Example 4: List all rewards
    print("\n4. Available rewards:")
    for name in reward_registry.list_rewards():
        print(f"   - {name}")

    print("\n✓ Simple usage test passed!")


