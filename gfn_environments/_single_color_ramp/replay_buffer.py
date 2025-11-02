# replay_buffer.py
"""Dead simple trajectory replay buffer with Pydantic schema"""

from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
from datetime import datetime
import pickle
from pathlib import Path



# ============================================================================
# BUFFER
# ============================================================================

def load_buffer(csv_path: str = "trajectory_buffer.csv") -> pd.DataFrame:
    """Load buffer or create empty one"""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=['id', 'timestamp', 'trajectory_data', 'trajectory_len', 'num_actions'])


def add_trajectory(df: pd.DataFrame, trajectory: Dict, max_size: int | None = None) -> pd.DataFrame:
    """Add trajectory to buffer, return updated DataFrame"""
    trajectory_id = len(df)
    record = TrajectoryRecord.from_trajectory(trajectory_id, trajectory)

    new_row = pd.DataFrame([record.to_dict()])
    df = pd.concat([df, new_row], ignore_index=True)

    # Enforce max size
    if max_size and len(df) > max_size:
        df = df.tail(max_size).reset_index(drop=True)
        df['id'] = range(len(df))

    return df


def save_buffer(df: pd.DataFrame, csv_path: str = "trajectory_buffer.csv"):
    """Save buffer to disk"""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def get_trajectory(df: pd.DataFrame, trajectory_id: int) -> Dict | None:
    """Get trajectory by ID"""
    row = df[df['id'] == trajectory_id]
    if row.empty:
        return None
    return pickle.loads(row.iloc[0]['trajectory_data'])


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    from gfn_environments.single_color_ramp import sample_random_trajectory

    # Load or create
    df = load_buffer("experiments/buffer.csv")

    # Add trajectories
    print("Adding trajectories...")
    for i in range(10):
        traj = sample_random_trajectory(trajectory_len=8, max_colors=5)
        df = add_trajectory(df, traj, max_size=100)
        print(f"  Added trajectory {i}, buffer size: {len(df)}")

    # Save
    save_buffer(df, "experiments/buffer.csv")

    # Use it like any DataFrame
    print("\n" + "=" * 70)
    print("Buffer contents:")
    print(df[['id', 'timestamp', 'trajectory_len', 'num_actions']])

    print("\nTrajectories with >5 actions:")
    print(df[df['num_actions'] > 5])

    print("\nStats:")
    print(df['trajectory_len'].describe())

    # Get specific trajectory
    traj = get_trajectory(df, 0)
    print(f"\nTrajectory 0 has {traj['num_actions']} actions")

    # Sample random batch
    batch = df.sample(n=3)
    print(f"\nSampled {len(batch)} trajectories")