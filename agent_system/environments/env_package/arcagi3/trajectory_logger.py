"""Trajectory logger for ARC-AGI-3 experiments.

Records every step of every episode for post-hoc analysis:
- VLM output (think, memory, action)
- Frame changes
- Reward
- Game state

Saves to JSONL files, one per game.
"""

import json
import os
import time
from typing import Optional


class TrajectoryLogger:
    """Logs trajectories to JSONL files for analysis."""

    def __init__(self, log_dir: str = "trajectories"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episodes = {}  # env_idx -> current episode data

    def start_episode(self, env_idx: int, game_stem: str):
        """Called on env reset."""
        self.episodes[env_idx] = {
            "game_stem": game_stem,
            "start_time": time.time(),
            "steps": [],
        }

    def log_step(self, env_idx: int, step_data: dict):
        """Log one step of an episode.

        step_data should contain:
            action: int (1-7)
            action_text: str (raw VLM output)
            think: str (extracted think content)
            memory: str (extracted memory content)
            reward: float
            done: bool
            won: bool
            valid: bool
            pixels_changed: int (optional)
        """
        if env_idx not in self.episodes:
            return
        self.episodes[env_idx]["steps"].append(step_data)

    def end_episode(self, env_idx: int):
        """Called when episode ends. Writes to JSONL file."""
        if env_idx not in self.episodes:
            return

        episode = self.episodes.pop(env_idx)
        episode["end_time"] = time.time()
        episode["duration_s"] = episode["end_time"] - episode["start_time"]
        episode["total_steps"] = len(episode["steps"])
        episode["total_reward"] = sum(s.get("reward", 0) for s in episode["steps"])
        episode["any_won"] = any(s.get("won", False) for s in episode["steps"])

        game_stem = episode["game_stem"]
        log_path = os.path.join(self.log_dir, f"{game_stem}.jsonl")

        with open(log_path, "a") as f:
            f.write(json.dumps(episode, default=str) + "\n")

    def get_summary(self) -> dict:
        """Return summary stats across all logged episodes."""
        stats = {
            "active_episodes": len(self.episodes),
        }

        # Count episodes per game from log files
        for fname in os.listdir(self.log_dir):
            if fname.endswith(".jsonl"):
                stem = fname.replace(".jsonl", "")
                path = os.path.join(self.log_dir, fname)
                with open(path) as f:
                    lines = f.readlines()
                stats[f"logged_episodes_{stem}"] = len(lines)

        return stats
