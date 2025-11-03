"""
GFlowNet Training Experiment with Replay Buffer
- Pre-fills replay buffer once at start
- Trains only on replay buffer
- Validates by generating new trajectories from model
- Saves generated trajectories as artifacts
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import json
from datetime import datetime

from models.cgfn_model import ConvTBModel
from gfn_environments.single_color_ramp import (
    load_blend_single_color_ramp,
    register_terrain_actions
)
from gfn_environments.single_color_ramp import (
    prefill_replay_buffer,
    ReplayBuffer
)
from action_utils.action_regestry_util import ActionRegistry


# ============================================================================
# Reward Function
# ============================================================================

def target_variance_reward(
        tensor: torch.Tensor,
        min_variance: float = 0.3,
        max_variance: float = 0.7
) -> float:
    """Reward for heightmaps with variance in target range"""
    variance = tensor.var().item()
    return 1.0 if min_variance <= variance <= max_variance else 0.0


# ============================================================================
# GFlowNet Trainer
# ============================================================================

class GFlowNetTrainer:
    """Trainer for GFlowNet with Trajectory Balance objective"""

    def __init__(
            self,
            model: ConvTBModel,
            replay_buffer: ReplayBuffer,
            action_registry: ActionRegistry,
            reward_fn,
            reward_params: Dict,
            max_colors: int = 16,
            learning_rate: float = 1e-3,
            device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.replay_buffer = replay_buffer
        self.action_registry = action_registry
        self.reward_fn = reward_fn
        self.reward_params = reward_params
        self.max_colors = max_colors
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.action_dim = action_registry.total_actions
        self.action_history_size = model.action_history_size

    def compute_trajectory_logprobs(self, trajectory: Dict) -> tuple:
        """Compute forward and backward log probabilities for a trajectory"""
        action_tensors = trajectory['action_tensors']
        heightmaps = trajectory['heightmaps']

        log_probs_forward = []
        log_probs_backward = []

        for step in range(len(action_tensors)):
            heightmap = heightmaps[step].unsqueeze(0).to(self.device)

            # Build action history
            if step == 0:
                action_history = torch.zeros(1, self.action_history_size, self.action_dim)
            else:
                prev_actions = action_tensors[:step]
                if len(prev_actions) < self.action_history_size:
                    padding = torch.zeros(
                        self.action_history_size - len(prev_actions),
                        self.action_dim
                    )
                    action_history = torch.cat([padding, prev_actions], dim=0).unsqueeze(0)
                else:
                    action_history = prev_actions[-self.action_history_size:].unsqueeze(0)

            action_history = action_history.to(self.device)

            # Get model predictions
            P_F_logits, P_B_logits = self.model(heightmap, action_history)

            # Get action that was taken
            action_tensor = action_tensors[step]
            action_idx = torch.argmax(action_tensor).item()

            # Compute log probabilities
            P_F_probs = torch.softmax(P_F_logits, dim=-1)
            log_pf = torch.log(P_F_probs[0, action_idx] + 1e-10)

            P_B_probs = torch.softmax(P_B_logits, dim=-1)
            log_pb = torch.log(P_B_probs[0, action_idx] + 1e-10)

            log_probs_forward.append(log_pf)
            log_probs_backward.append(log_pb)

        return log_probs_forward, log_probs_backward

    def compute_tb_loss(self, trajectories: List[Dict]) -> torch.Tensor:
        """Compute Trajectory Balance loss"""
        losses = []

        for traj in trajectories:
            log_probs_forward, log_probs_backward = self.compute_trajectory_logprobs(traj)

            log_pf_sum = sum(log_probs_forward)
            log_pb_sum = sum(log_probs_backward)

            reward = traj['reward']
            log_reward = torch.log(torch.tensor(reward + 1e-10))

            loss = (self.model.logZ + log_pf_sum - log_pb_sum - log_reward) ** 2
            losses.append(loss)

        return torch.stack(losses).mean()

    def train_step(self, batch_size: int) -> Dict:
        """Single training step - samples from replay buffer"""
        self.model.train()

        # Sample from replay buffer
        df = self.replay_buffer.df
        if len(df) < batch_size:
            batch_size = len(df)

        sampled_df = df.sample(n=batch_size)

        # Load trajectories
        import pickle
        trajectories = []
        for _, row in sampled_df.iterrows():
            traj_data = pickle.loads(row['trajectory_data'])
            traj_data['reward'] = row['reward']
            traj_data['variance'] = row['variance']
            trajectories.append(traj_data)

        # Compute loss
        loss = self.compute_tb_loss(trajectories)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'logZ': self.model.logZ.item(),
        }

    def generate_trajectories(self, num_samples: int, trajectory_len: int) -> List[Dict]:
        """
        Generate trajectories from current policy

        Returns list of dicts with: actions, variance, reward, num_steps
        """
        self.model.eval()

        from blender_setup_utils import BlenderTensorUtility
        import bpy

        generated_trajectories = []

        with torch.no_grad():
            for i in range(num_samples):
                load_blend_single_color_ramp()

                action_tensors = []
                heightmaps = []
                actions = []

                for step in range(trajectory_len):
                    if step == 0:
                        bpy.context.view_layer.update()
                        heightmap = BlenderTensorUtility.get_heightmap_by_name("TerrainPlane")
                        action_history = torch.zeros(1, self.action_history_size, self.action_dim)
                    else:
                        heightmap = heightmaps[-1]
                        prev_actions = torch.stack(action_tensors)
                        if len(prev_actions) < self.action_history_size:
                            padding = torch.zeros(
                                self.action_history_size - len(prev_actions),
                                self.action_dim
                            )
                            action_history = torch.cat([padding, prev_actions], dim=0).unsqueeze(0)
                        else:
                            action_history = prev_actions[-self.action_history_size:].unsqueeze(0)

                    mask = self.action_registry.get_action_mask()
                    if mask.sum() == 0:
                        break

                    heightmap_batch = heightmap.unsqueeze(0).to(self.device)
                    action_history = action_history.to(self.device)

                    P_F_logits, _ = self.model(heightmap_batch, action_history)

                    # Mask invalid actions
                    masked_logits = P_F_logits.clone()
                    masked_logits[0, ~mask] = float('-inf')

                    # Sample action
                    P_F_probs = torch.softmax(masked_logits, dim=-1)
                    action_idx = torch.multinomial(P_F_probs[0], 1).item()

                    # Create one-hot action tensor
                    action_tensor = torch.zeros(self.action_dim)
                    action_tensor[action_idx] = 1.0

                    # Apply action
                    action_result = self.action_registry[action_tensor]()
                    actions.append(action_result)

                    # Get new heightmap
                    bpy.context.view_layer.update()
                    new_heightmap = BlenderTensorUtility.get_heightmap_by_name("TerrainPlane")

                    action_tensors.append(action_tensor)
                    heightmaps.append(new_heightmap)

                if len(heightmaps) > 0:
                    final_heightmap = heightmaps[-1]
                    variance = final_heightmap.var().item()
                    reward = self.reward_fn(final_heightmap, **self.reward_params)

                    generated_trajectories.append({
                        'actions': actions,
                        'variance': variance,
                        'reward': reward,
                        'num_steps': len(actions)
                    })

        return generated_trajectories

    def evaluate_policy(self, num_samples: int, trajectory_len: int) -> Dict:
        """Evaluate policy by generating trajectories"""
        generated = self.generate_trajectories(num_samples, trajectory_len)

        if len(generated) == 0:
            return {
                'policy/success_rate': 0.0,
                'policy/avg_reward': 0.0,
                'policy/avg_variance': 0.0,
                'policy/std_variance': 0.0,
                'policy/min_variance': 0.0,
                'policy/max_variance': 0.0,
                'policy/avg_steps': 0.0,
            }

        variances = [t['variance'] for t in generated]
        rewards = [t['reward'] for t in generated]
        steps = [t['num_steps'] for t in generated]

        in_range = [
            self.reward_params['min_variance'] <= v <= self.reward_params['max_variance']
            for v in variances
        ]
        success_rate = np.mean(in_range)

        return {
            'policy/success_rate': success_rate,
            'policy/avg_reward': np.mean(rewards),
            'policy/avg_variance': np.mean(variances),
            'policy/std_variance': np.std(variances),
            'policy/min_variance': min(variances),
            'policy/max_variance': max(variances),
            'policy/avg_steps': np.mean(steps),
            'generated_trajectories': generated  # Include for artifact saving
        }


# ============================================================================
# Artifact Saving Utilities
# ============================================================================

def save_trajectories_as_artifact(trajectories: List[Dict], filename: str = "generated_trajectories.json"):
    """
    Save generated trajectories as human-readable JSON artifact

    Args:
        trajectories: List of trajectory dicts
        filename: Name for artifact file
    """
    # Convert to human-readable format
    readable_trajectories = []

    for i, traj in enumerate(trajectories):
        readable_traj = {
            'trajectory_id': i,
            'num_steps': traj['num_steps'],
            'reward': float(traj['reward']),
            'variance': float(traj['variance']),
            'actions': traj['actions']  # Already human-readable dicts
        }
        readable_trajectories.append(readable_traj)

    # Save as JSON
    filepath = Path(filename)
    with open(filepath, 'w') as f:
        json.dump(readable_trajectories, f, indent=2)

    # Log to MLflow
    mlflow.log_artifact(str(filepath))

    # Clean up local file
    filepath.unlink()


def save_trajectory_summary_as_artifact(trajectories: List[Dict], filename: str = "trajectory_summary.csv"):
    """
    Save trajectory summary statistics as CSV artifact

    Args:
        trajectories: List of trajectory dicts
        filename: Name for artifact file
    """
    summary_data = []

    for i, traj in enumerate(trajectories):
        # Count action types
        action_types = [a['type'] for a in traj['actions']]
        type_counts = {}
        for atype in action_types:
            type_counts[atype] = type_counts.get(atype, 0) + 1

        summary_data.append({
            'trajectory_id': i,
            'num_steps': traj['num_steps'],
            'reward': traj['reward'],
            'variance': traj['variance'],
            'num_set_w': type_counts.get('set_w', 0),
            'num_set_scale': type_counts.get('set_scale', 0),
            'num_add_color': type_counts.get('add_color', 0),
        })

    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    filepath = Path(filename)
    df.to_csv(filepath, index=False)

    # Log to MLflow
    mlflow.log_artifact(str(filepath))

    # Clean up
    filepath.unlink()


# ============================================================================
# Training Function
# ============================================================================

def train_gflownet(config: Dict) -> float:
    """
    Train GFlowNet with given config

    Args:
        config: Training configuration dictionary

    Returns:
        Best success rate achieved
    """
    print(f"\n{'=' * 70}")
    print(f"Training Configuration:")
    print(f"  Iterations: {config['num_iterations']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']:.5f}")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Action History: {config['action_history_size']}")
    print(f"  Buffer Size: {config['num_samples']}")
    print(f"{'=' * 70}\n")

    # Initialize environment
    load_blend_single_color_ramp()

    # Register actions
    action_registry = ActionRegistry()
    from gfn_environments.single_color_ramp import register_terrain_actions
    register_terrain_actions(action_registry)

    # Get dimensions
    from gfn_environments.single_color_ramp import sample_random_trajectory
    sample_traj = sample_random_trajectory(
        action_registry=action_registry,
        trajectory_len=1,
        max_colors=config['max_colors']
    )
    heightmap_size = sample_traj['heightmaps'][0].shape[0]
    action_dim = action_registry.total_actions

    print(f"Environment Info:")
    print(f"  Heightmap size: {heightmap_size}x{heightmap_size}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Max colors: {config['max_colors']}\n")

    # Pre-fill replay buffer
    replay_buffer = prefill_replay_buffer(
        action_registry=action_registry,
        num_samples=config['num_samples'],
        max_colors=config['max_colors'],
        trajectory_len=config['trajectory_len'],
        reward_fn=target_variance_reward,
        reward_params=config['reward_params'],
        buffer_capacity=config['buffer_capacity']
    )

    # Log buffer statistics
    df = replay_buffer.df
    baseline_success_rate = df['reward'].mean()

    mlflow.log_metrics({
        'buffer/size': len(replay_buffer),
        'buffer/success_rate': baseline_success_rate,
        'buffer/avg_variance': df['variance'].mean(),
        'buffer/std_variance': df['variance'].std(),
    }, step=0)

    print(f"Buffer Statistics:")
    print(f"  Size: {len(replay_buffer)}")
    print(f"  Baseline success rate: {baseline_success_rate:.2%}\n")

    # Create model
    model = ConvTBModel(
        heightmap_channels=1,
        heightmap_size=heightmap_size,
        action_dim=action_dim,
        hidden_size=config['hidden_size'],
        action_history_size=config['action_history_size']
    )

    # Create trainer
    trainer = GFlowNetTrainer(
        model=model,
        replay_buffer=replay_buffer,
        action_registry=action_registry,
        reward_fn=target_variance_reward,
        reward_params=config['reward_params'],
        max_colors=config['max_colors'],
        learning_rate=config['learning_rate'],
        device=config['device']
    )

    best_success_rate = 0.0

    # Training loop
    print(f"Starting training...\n")
    for iteration in range(config['num_iterations']):
        stats = trainer.train_step(config['batch_size'])

        # Log training metrics
        mlflow.log_metrics({
            'train/loss': stats['loss'],
            'train/logZ': stats['logZ'],
        }, step=iteration)

        # Evaluate policy
        if (iteration + 1) % config['eval_interval'] == 0:
            eval_stats = trainer.evaluate_policy(
                config['eval_samples'],
                config['trajectory_len']
            )

            # Extract generated trajectories
            generated = eval_stats.pop('generated_trajectories')

            success_rate = eval_stats['policy/success_rate']
            best_success_rate = max(best_success_rate, success_rate)

            # Compute improvement
            improvement = success_rate - baseline_success_rate
            improvement_pct = (improvement / max(baseline_success_rate, 0.01)) * 100

            # Log metrics
            mlflow.log_metrics({
                **eval_stats,
                'comparison/improvement_absolute': improvement,
                'comparison/improvement_percent': improvement_pct,
                'comparison/best_success_rate': best_success_rate,
            }, step=iteration)

            print(f"Iter {iteration + 1:4d}: "
                  f"Loss={stats['loss']:.4f}, "
                  f"Policy={success_rate:.2%}, "
                  f"Best={best_success_rate:.2%}, "
                  f"Improve={improvement:+.2%}")

            # Save generated trajectories as artifacts
            save_trajectories_as_artifact(
                generated,
                filename=f"generated_trajectories_iter_{iteration + 1}.json"
            )
            save_trajectory_summary_as_artifact(
                generated,
                filename=f"trajectory_summary_iter_{iteration + 1}.csv"
            )

    # Final evaluation
    print(f"\nFinal evaluation...")
    final_eval_stats = trainer.evaluate_policy(
        num_samples=config['eval_samples'],
        trajectory_len=config['trajectory_len']
    )

    final_generated = final_eval_stats.pop('generated_trajectories')
    final_success_rate = final_eval_stats['policy/success_rate']
    best_success_rate = max(best_success_rate, final_success_rate)

    # Log final metrics
    mlflow.log_metrics({
        'final/success_rate': final_success_rate,
        'best/success_rate': best_success_rate,
        'final/improvement': final_success_rate - baseline_success_rate,
    })

    # Save final trajectories
    save_trajectories_as_artifact(
        final_generated,
        filename="final_generated_trajectories.json"
    )
    save_trajectory_summary_as_artifact(
        final_generated,
        filename="final_trajectory_summary.csv"
    )

    print(f"\n{'=' * 70}")
    print(f"Training Complete!")
    print(f"  Final success rate: {final_success_rate:.2%}")
    print(f"  Best success rate: {best_success_rate:.2%}")
    print(f"  Baseline: {baseline_success_rate:.2%}")
    print(f"  Improvement: {final_success_rate - baseline_success_rate:+.2%}")
    print(f"{'=' * 70}\n")

    return best_success_rate


# ============================================================================
# Optuna Objective
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function"""

    config = {
        'reward_params': {
            'min_variance': 0.3,
            'max_variance': 0.7
        },
        'max_colors': 16,
        'trajectory_len': 8,
        'num_iterations': trial.suggest_int('num_iterations', 500, 2000, step=250),
        'batch_size': trial.suggest_int('batch_size', 16, 128, step=16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
        'action_history_size': trial.suggest_int('action_history_size', 0, 15),
        'num_samples': trial.suggest_int('num_samples', 200, 1000, step=200),
        'buffer_capacity': 10000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_interval': 50,
        'eval_samples': 50,
    }

    mlflow.set_experiment("gflownet_replay_buffer_training")

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        try:
            mlflow.log_params(config)
            mlflow.log_param('trial_number', trial.number)

            best_success_rate = train_gflownet(config)

            mlflow.log_param('status', 'completed')
            return best_success_rate

        except optuna.TrialPruned:
            mlflow.log_param('status', 'pruned')
            raise

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            mlflow.log_param('status', 'failed')
            mlflow.log_param('error', str(e))
            return 0.0


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Single training run
    config = {
        'reward_params': {'min_variance': 0.3, 'max_variance': 0.7},
        'max_colors': 16,
        'trajectory_len': 8,
        'num_iterations': 1000,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'hidden_size': 256,
        'action_history_size': 5,
        'num_samples': 500,
        'buffer_capacity': 10000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_interval': 50,
        'eval_samples': 50,
    }

    mlflow.set_experiment("gflownet_replay_buffer_training")

    with mlflow.start_run(run_name="single_run"):
        mlflow.log_params(config)
        best_rate = train_gflownet(config)
        print(f"\nBest success rate: {best_rate:.2%}")