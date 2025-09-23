#!/usr/bin/env python3
"""
Script to calculate normalization statistics from lerobot-pi0-bridge data.
Calculates mean, variance (std), and quantiles (q01, q99) for state and action data.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_parquet_files(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all parquet files and extract state and action data."""
    data_path = Path(data_dir)

    all_states = []
    all_actions = []

    # Find all chunk directories
    chunk_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('chunk-')])

    print(f"Found {len(chunk_dirs)} chunk directories")

    for chunk_dir in chunk_dirs:
        print(f"Processing {chunk_dir.name}...")

        # Get all parquet files in this chunk
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))

        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)

                # Extract state and action columns
                if 'observation.state' in df.columns:
                    state_data = np.vstack(df['observation.state'].values)
                    all_states.append(state_data)

                if 'action' in df.columns:
                    action_data = np.vstack(df['action'].values)
                    all_actions.append(action_data)

            except Exception as e:
                print(f"Error processing {parquet_file}: {e}")
                continue

    # Concatenate all data
    states = np.vstack(all_states) if all_states else np.array([])
    actions = np.vstack(all_actions) if all_actions else np.array([])

    print(f"Loaded {states.shape[0]} state samples with {states.shape[1]} dimensions")
    print(f"Loaded {actions.shape[0]} action samples with {actions.shape[1]} dimensions")

    return states, actions


def calculate_statistics(data: np.ndarray) -> Dict[str, List[float]]:
    """Calculate mean, std, q01, and q99 for each dimension."""
    if data.size == 0:
        return {"mean": [], "std": [], "q01": [], "q99": []}

    stats = {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "q01": np.percentile(data, 1, axis=0).tolist(),
        "q99": np.percentile(data, 99, axis=0).tolist()
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Calculate normalization statistics from lerobot data")
    parser.add_argument("--data_dir", type=str, default="/opt/tiger/openpi/lerobot-pi0-bridge/data",
                        help="Path to the lerobot data directory")
    parser.add_argument("--output", type=str, default="norm_stats.json",
                        help="Output JSON file path")

    args = parser.parse_args()

    # Load data
    print("Loading parquet files...")
    states, actions = load_parquet_files(args.data_dir)

    # Calculate statistics
    print("Calculating statistics...")
    state_stats = calculate_statistics(states)
    action_stats = calculate_statistics(actions)

    # Create output structure matching the expected format
    output = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats
        }
    }

    # Save to JSON file
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Statistics saved to {args.output}")

    # Print summary
    print("\nSummary:")
    print(f"State dimensions: {len(state_stats['mean'])}")
    print(f"Action dimensions: {len(action_stats['mean'])}")
    print(f"State mean range: [{min(state_stats['mean']):.6f}, {max(state_stats['mean']):.6f}]")
    print(f"Action mean range: [{min(action_stats['mean']):.6f}, {max(action_stats['mean']):.6f}]")


if __name__ == "__main__":
    main()