import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import numpy as np
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt

from models.cgfn_model import ConvTBModel
from gfn_environments.single_color_ramp import (
    sample_random_trajectory,
    registry as action_registry,
    register_color_actions,
    load_blend_single_color_ramp
)
from gfn_environments.heightmap_reward_utils import reward_registry


# ============================================================================
# Register Target Reward
# ============================================================================

@reward_registry.register('target_variance')
def target_variance_reward(
        tensor: torch.Tensor,
        min_variance: float = 0.3,
        max_variance: float = 0.7
) -> float:
    """Reward for heightmaps with variance in target range [0.3, 0.7]"""
    if torch.is_tensor(tensor):
        variance = tensor.var().item()
    else:
        variance = np.var(tensor)

    if min_variance <= variance <= max_variance:
        return 1.0
    else:
        return 0.0


# ============================================================================
# GFlowNet Training
# ============================================================================

class GFlowNetTrainer:
    """Trainer for GFlowNet with Trajectory Balance objective"""

    def __init__(
            self,
            model: ConvTBModel,
            reward_fn_name: str,
            reward_params: Dict,
            max_colors: int = 5,
            learning_rate: float = 1e-3,
            device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.reward_fn = reward_registry[reward_fn_name]
        self.reward_params = reward_params
        self.max_colors = max_colors
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if 'add_color' not in action_registry.action_type_info:
            register_color_actions(max_colors)

        self.action_dim = action_registry.total_actions
        self.action_history_size = model.action_history_size

    def sample_trajectory(self, trajectory_len: int) -> Dict:
        """Sample a trajectory from current policy"""
        load_blend_single_color_ramp()

        actions = []
        action_tensors = []
        heightmaps = []
        log_probs_forward = []
        log_probs_backward = []

        for step in range(trajectory_len):
            if step == 0:
                from blender_setup_utils import BlenderTensorUtility
                import bpy
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

            mask = action_registry.get_action_mask()

            if mask.sum() == 0:
                break

            heightmap_batch = heightmap.unsqueeze(0).to(self.device)
            action_history = action_history.to(self.device)

            P_F_logits, P_B_logits = self.model(heightmap_batch, action_history)

            masked_logits = P_F_logits.clone()
            masked_logits[0, ~mask] = float('-inf')

            P_F_probs = torch.softmax(masked_logits, dim=-1)
            action_tensor = action_registry.sample_from(mask)

            action_idx = torch.argmax(action_tensor).item()

            log_pf = torch.log(P_F_probs[0, action_idx] + 1e-10)

            P_B_probs = torch.softmax(P_B_logits, dim=-1)
            log_pb = torch.log(P_B_probs[0, action_idx] + 1e-10)

            action = action_registry[action_tensor]()

            import bpy
            from blender_setup_utils import BlenderTensorUtility
            bpy.context.view_layer.update()
            new_heightmap = BlenderTensorUtility.get_heightmap_by_name("TerrainPlane")

            actions.append(action)
            action_tensors.append(action_tensor)
            heightmaps.append(new_heightmap)
            log_probs_forward.append(log_pf)
            log_probs_backward.append(log_pb)

        final_heightmap = heightmaps[-1]
        reward = self.reward_fn(final_heightmap, **self.reward_params)
        variance = final_heightmap.var().item()

        return {
            'actions': actions,
            'action_tensors': action_tensors,
            'heightmaps': heightmaps,
            'log_probs_forward': log_probs_forward,
            'log_probs_backward': log_probs_backward,
            'reward': reward,
            'variance': variance,
            'num_steps': len(actions)
        }

    def compute_tb_loss(self, trajectories: List[Dict]) -> torch.Tensor:
        """Compute Trajectory Balance loss"""
        losses = []

        for traj in trajectories:
            log_pf_sum = sum(traj['log_probs_forward'])
            log_pb_sum = sum(traj['log_probs_backward'])

            reward = traj['reward']
            log_reward = torch.log(torch.tensor(reward + 1e-10))

            loss = (self.model.logZ + log_pf_sum - log_pb_sum - log_reward) ** 2
            losses.append(loss)

        return torch.stack(losses).mean()

    def train_step(self, num_trajectories: int, trajectory_len: int) -> Dict:
        """
        Single training step

        Args:
            num_trajectories: Batch size (number of trajectories to sample)
            trajectory_len: Maximum length of each trajectory

        Returns:
            Dictionary with training statistics
        """
        self.model.train()

        # Sample batch of trajectories
        trajectories = []
        for _ in range(num_trajectories):
            traj = self.sample_trajectory(trajectory_len)
            trajectories.append(traj)

        # Compute loss
        loss = self.compute_tb_loss(trajectories)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute statistics on GENERATED samples
        rewards = [traj['reward'] for traj in trajectories]
        variances = [traj['variance'] for traj in trajectories]
        steps = [traj['num_steps'] for traj in trajectories]

        # Check if generated samples are in target range
        in_range = [
            self.reward_params['min_variance'] <= v <= self.reward_params['max_variance']
            for v in variances
        ]
        success_rate = np.mean(in_range)

        return {
            'loss': loss.item(),
            'generated_success_rate': success_rate,  # Success rate of generated samples
            'generated_avg_reward': np.mean(rewards),
            'generated_min_reward': min(rewards),
            'generated_max_reward': max(rewards),
            'generated_avg_variance': np.mean(variances),
            'generated_min_variance': min(variances),
            'generated_max_variance': max(variances),
            'generated_std_variance': np.std(variances),
            'generated_avg_steps': np.mean(steps),
            'logZ': self.model.logZ.item(),
        }

    def validate(self, num_samples: int, trajectory_len: int) -> Dict:
        """Simple validation without visualizations"""
        self.model.eval()

        test_trajectories = []

        with torch.no_grad():
            for i in range(num_samples):
                traj = self.sample_trajectory(trajectory_len)
                test_trajectories.append(traj)

        variances = [traj['variance'] for traj in test_trajectories]
        rewards = [traj['reward'] for traj in test_trajectories]
        steps = [traj['num_steps'] for traj in test_trajectories]

        in_range = [
            self.reward_params['min_variance'] <= v <= self.reward_params['max_variance']
            for v in variances
        ]
        success_rate = np.mean(in_range)

        val_stats = {
            'val/success_rate': success_rate,
            'val/avg_reward': np.mean(rewards),
            'val/avg_variance': np.mean(variances),
            'val/std_variance': np.std(variances),
            'val/min_variance': min(variances),
            'val/max_variance': max(variances),
            'val/avg_steps': np.mean(steps),
        }

        return val_stats



# ============================================================================
# Random Baseline
# ============================================================================

def compute_random_baseline(
        max_colors: int,
        trajectory_len: int,
        num_samples: int,
        reward_fn,
        reward_params: Dict
) -> Dict:
    """
    Compute random baseline performance

    Args:
        max_colors: Number of color slots
        trajectory_len: Maximum trajectory length
        num_samples: Number of random trajectories to sample
        reward_fn: Reward function
        reward_params: Reward function parameters

    Returns:
        Dictionary with random baseline statistics
    """
    print(f"\nComputing random baseline ({num_samples} samples)...")

    random_trajectories = []

    for i in range(num_samples):
        if (i + 1) % 20 == 0:
            print(f"  Random sample {i + 1}/{num_samples}...", end='\r')

        traj = sample_random_trajectory(trajectory_len, max_colors)
        random_trajectories.append(traj)

    print(" " * 80, end='\r')  # Clear line

    # Compute statistics
    variances = [traj['heightmaps'][-1].var().item() for traj in random_trajectories]
    rewards = [reward_fn(traj['heightmaps'][-1], **reward_params) for traj in random_trajectories]

    in_range = [
        reward_params['min_variance'] <= v <= reward_params['max_variance']
        for v in variances
    ]
    success_rate = np.mean(in_range)

    baseline_stats = {
        'random/success_rate': success_rate,
        'random/avg_variance': np.mean(variances),
        'random/std_variance': np.std(variances),
        'random/min_variance': min(variances),
        'random/max_variance': max(variances),
        'random/avg_reward': np.mean(rewards),
    }

    print(f"✓ Random baseline success rate: {success_rate:.2%}")

    return baseline_stats


# ============================================================================
# Optuna Objective Function
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function to optimize GFlowNet hyperparameters

    Args:
        trial: Optuna trial object

    Returns:
        Validation success rate (metric to maximize)
    """

    # Sample hyperparameters
    config = {
        'reward': 'target_variance',
        'reward_params': {
            'min_variance': 0.3,
            'max_variance': 0.7
        },
        'max_colors': 16,
        'trajectory_len':8,
        'num_iterations': trial.suggest_int('num_iterations', 500, 2000, step=250),
        'num_trajectories': trial.suggest_int('num_trajectories', 16, 128, step=16),  # Batch size
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
        'action_history_size': trial.suggest_int('action_history_size', 0, 15),
        'device': 'cuda',
        'val_interval': 50,
        'val_samples': 100,
    }

    mlflow.set_experiment("gflownet_optuna_optimization")

    with mlflow.start_run(run_name=f"trial_{trial.number}"):

        try:
            # Log all hyperparameters
            mlflow.log_params(config)
            mlflow.log_param('trial_number', trial.number)

            # Initialize environment
            load_blend_single_color_ramp()

            if 'add_color' not in action_registry.action_type_info:
                register_color_actions(config['max_colors'])

            # Get dimensions
            sample_traj = sample_random_trajectory(1, config['max_colors'])
            heightmap_size = sample_traj['heightmaps'][0].shape[0]
            action_dim = action_registry.total_actions

            print(f"\nTrial {trial.number}:")
            print(f"  Iterations={config['num_iterations']}, ValInterval={config['val_interval']}")
            print(f"  BatchSize={config['num_trajectories']}, LR={config['learning_rate']:.5f}")
            print(f"  Hidden={config['hidden_size']}, ActionHistory={config['action_history_size']}")
            print(f"  TrajLen={config['trajectory_len']}, Device={config['device']}")
            print(f"  MaxColors={config['max_colors']}, ActionDim={action_dim}")

            # Compute random baseline
            reward_fn = reward_registry[config['reward']]
            random_baseline = compute_random_baseline(
                max_colors=config['max_colors'],
                trajectory_len=config['trajectory_len'],
                num_samples=config['val_samples'],
                reward_fn=reward_fn,
                reward_params=config['reward_params']
            )

            # Log random baseline
            mlflow.log_metrics(random_baseline, step=0)
            random_success_rate = random_baseline['random/success_rate']

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
                reward_fn_name=config['reward'],
                reward_params=config['reward_params'],
                max_colors=config['max_colors'],
                learning_rate=config['learning_rate'],
                device=config['device']
            )

            best_success_rate = 0.0

            # Training loop
            for iteration in range(config['num_iterations']):
                stats = trainer.train_step(config['num_trajectories'], config['trajectory_len'])

                # Log training metrics (from generated samples during training)
                mlflow.log_metrics({
                    'train/loss': stats['loss'],
                    'train/generated_success_rate': stats['generated_success_rate'],
                    'train/generated_avg_reward': stats['generated_avg_reward'],
                    'train/generated_avg_variance': stats['generated_avg_variance'],
                    'train/logZ': stats['logZ'],
                }, step=iteration)

                # Validation
                if (iteration + 1) % config['val_interval'] == 0:
                    val_stats = trainer.validate(
                        config['val_samples'],
                        config['trajectory_len']
                    )

                    success_rate = val_stats['val/success_rate']
                    best_success_rate = max(best_success_rate, success_rate)

                    # Compute improvement over random
                    improvement = success_rate - random_success_rate
                    improvement_pct = (improvement / max(random_success_rate, 0.01)) * 100

                    # Log validation metrics
                    mlflow.log_metrics({
                        **val_stats,
                        'comparison/improvement_absolute': improvement,
                        'comparison/improvement_percent': improvement_pct,
                        'comparison/success_vs_random': success_rate / max(random_success_rate, 0.01),
                    }, step=iteration)

                    print(f"    Iter {iteration + 1}: "
                          f"Val={success_rate:.2%}, Train={stats['generated_success_rate']:.2%}, "
                          f"Best={best_success_rate:.2%}, vs Random: {improvement:+.2%}")

                    # Report for pruning
                    trial.report(success_rate, iteration)

                    if trial.should_prune():
                        print(f"    Pruned at iteration {iteration + 1}")
                        raise optuna.TrialPruned()

            # Final validation
            final_val_stats = trainer.validate(
                num_samples=config['val_samples'],
                trajectory_len=config['trajectory_len']
            )

            final_success_rate = final_val_stats['val/success_rate']
            best_success_rate = max(best_success_rate, final_success_rate)

            # Compute final improvement over random
            final_improvement = final_success_rate - random_success_rate
            final_improvement_pct = (final_improvement / max(random_success_rate, 0.01)) * 100

            # Log final metrics
            mlflow.log_metrics({
                'final/success_rate': final_success_rate,
                'final/avg_variance': final_val_stats['val/avg_variance'],
                'best/success_rate': best_success_rate,
                'final/improvement_absolute': final_improvement,
                'final/improvement_percent': final_improvement_pct,
                'final/success_vs_random': final_success_rate / max(random_success_rate, 0.01),
                'best/improvement_absolute': best_success_rate - random_success_rate,
                'best/improvement_percent': ((best_success_rate - random_success_rate) / max(random_success_rate,
                                                                                             0.01)) * 100,
            })

            mlflow.log_param('status', 'completed')

            print(f"\n  Completed!")
            print(f"    Final success rate: {final_success_rate:.2%}")
            print(f"    Best success rate: {best_success_rate:.2%}")
            print(f"    Random baseline: {random_success_rate:.2%}")
            print(f"    Improvement: {final_improvement:+.2%} ({final_improvement_pct:+.1f}%)")
            print()

            return best_success_rate

        except optuna.TrialPruned:
            mlflow.log_param('status', 'pruned')
            raise

        except Exception as e:
            print(f"  Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            mlflow.log_param('status', 'failed')
            mlflow.log_param('error', str(e))
            return 0.0


# ============================================================================
# Optuna Study Functions
# ============================================================================

def run_optimization(
        n_trials: int = 50,
        timeout: int = None,
        study_name: str = "gflownet_variance_optimization"
):
    """Run Optuna hyperparameter optimization"""

    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print("=" * 70 + "\n")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=200,
            interval_steps=50
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,)
    )

    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    # Filter out failed trials
    completed_trials = [t for t in study.trials if t.value is not None and t.value > 0]

    if len(completed_trials) == 0:
        print("\n⚠️  No trials completed successfully!")
        return study

    print(f"\nCompleted trials: {len(completed_trials)}/{len(study.trials)}")

    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (success rate): {trial.value:.2%}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Print top 5 trials
    print(f"\nTop 5 trials:")
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
    for i, trial in enumerate(top_trials):
        print(f"\n  {i + 1}. Trial {trial.number} - Success rate: {trial.value:.2%}")
        print(f"     Batch size: {trial.params['num_trajectories']}")
        print(f"     LR: {trial.params['learning_rate']:.5f}")
        print(f"     Hidden: {trial.params['hidden_size']}")
        print(f"     Action history: {trial.params['action_history_size']}")
        print(f"     Iterations: {trial.params['num_iterations']}")

    # Save study
    study_path = Path(f"optuna_study_{study_name}.pkl")
    import joblib
    joblib.dump(study, study_path)
    print(f"\n✓ Study saved to: {study_path}")

    # Create visualizations
    try:
        visualize_optimization(study)
    except Exception as e:
        print(f"\n⚠️  Could not create visualizations: {e}")

    return study


def visualize_optimization(study: optuna.Study):
    """Create Optuna visualizations"""
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
    )

    print("\nCreating optimization visualizations...")

    try:
        fig = plot_optimization_history(study)
        fig.write_image("optuna_history.png")
        print("  ✓ Saved: optuna_history.png")
    except Exception as e:
        print(f"  ⚠️  Could not create history plot: {e}")

    try:
        fig = plot_param_importances(study)
        fig.write_image("optuna_param_importance.png")
        print("  ✓ Saved: optuna_param_importance.png")
    except Exception as e:
        print(f"  ⚠️  Could not create param importance plot: {e}")

    try:
        fig = plot_parallel_coordinate(study)
        fig.write_image("optuna_parallel_coordinate.png")
        print("  ✓ Saved: optuna_parallel_coordinate.png")
    except Exception as e:
        print(f"  ⚠️  Could not create parallel coordinate plot: {e}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":

    # Run optimization
    study = run_optimization(
        n_trials=30,
        study_name="gflownet_variance_v1"
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    if study.best_value:
        print(f"Best success rate: {study.best_value:.2%}")
    print(f"View results in MLflow UI: mlflow ui")
    print("=" * 70)