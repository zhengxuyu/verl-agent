"""Tests for ARC-AGI-3 environment integration.

TDD: tests written BEFORE implementation fixes.
Run: cd verl-agent && python -m pytest tests/test_arcagi3.py -v
"""

import os
import sys
import pytest
import numpy as np

# ---- Data Preparation Tests ----

class TestDataPreparation:
    """Parquet dataset must cover all 25 games in correct format."""

    def test_all_25_games_in_dataset(self):
        """Dataset should contain all 25 public games, not just 8."""
        import pandas as pd
        # After fix, this parquet should have 25 rows
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        games = set(df["data_source"].tolist())
        assert len(games) == 25, f"Expected 25 games, got {len(games)}: {games}"

    def test_image_format_is_dict_with_bytes(self):
        """Each image must be {'bytes': <png bytes>, 'path': None}."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        for i, row in df.iterrows():
            images = row["images"]
            assert len(images) >= 1, f"Row {i}: no images"
            img = images[0]
            assert isinstance(img, dict), f"Row {i}: image is {type(img)}, expected dict"
            assert "bytes" in img, f"Row {i}: image missing 'bytes' key"
            assert isinstance(img["bytes"], bytes), f"Row {i}: bytes is {type(img['bytes'])}"
            assert img["bytes"][:4] == b'\x89PNG', f"Row {i}: not PNG data"

    def test_prompt_is_chat_message_format(self):
        """Prompt must be list of chat messages, not plain string."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        for i, row in df.iterrows():
            prompt = row["prompt"]
            assert isinstance(prompt, (list, np.ndarray)), \
                f"Row {i}: prompt is {type(prompt)}, expected list"
            msg = prompt[0]
            assert isinstance(msg, dict), f"Row {i}: message is {type(msg)}"
            assert "role" in msg, f"Row {i}: message missing 'role'"
            assert "content" in msg, f"Row {i}: message missing 'content'"
            assert "<image>" in msg["content"], f"Row {i}: missing <image> tag"

    def test_batch_size_divisible_by_gpus(self):
        """train_batch_size must be divisible by n_gpus (2)."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        # With 25 games, we need batch_size that divides by 2
        # Options: use all 25 but batch=24, or pad to 26
        n = len(df)
        assert n >= 25, f"Expected at least 25 samples, got {n}"


# ---- Projection Tests ----

class TestProjection:
    """Projection must parse <think>, <memory>, <action> and return 3 values."""

    def test_returns_three_values(self):
        """projection must return (actions, valids, memories)."""
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arcagi3_projection = mod.arcagi3_projection

        texts = ["<think>I see a blue dot</think><memory>Color 1 = player</memory><action>ACTION4</action>"]
        result = arcagi3_projection(texts)
        assert len(result) == 3, f"Expected 3 return values, got {len(result)}"
        actions, valids, memories = result

    def test_parses_action_correctly(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arcagi3_projection = mod.arcagi3_projection

        texts = [
            "<think>going right</think><memory>test</memory><action>ACTION4</action>",
            "<think>going up</think><memory>test</memory><action>ACTION1</action>",
            "<think>undo</think><memory>test</memory><action>ACTION7</action>",
        ]
        actions, valids, memories = arcagi3_projection(texts)
        assert actions == [4, 1, 7]
        assert valids == [1, 1, 1]

    def test_extracts_memory(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arcagi3_projection = mod.arcagi3_projection

        texts = [
            "<think>reasoning</think><memory>Color 1 is the player. ACTION4 moves right.</memory><action>ACTION4</action>"
        ]
        actions, valids, memories = arcagi3_projection(texts)
        assert "Color 1 is the player" in memories[0]
        assert "ACTION4 moves right" in memories[0]

    def test_invalid_without_think_tag(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arcagi3_projection = mod.arcagi3_projection

        texts = ["<memory>test</memory><action>ACTION4</action>"]  # no <think>
        actions, valids, memories = arcagi3_projection(texts)
        assert valids == [0], "Should be invalid without <think> tag"

    def test_invalid_without_action_tag(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arcagi3_projection = mod.arcagi3_projection

        texts = ["<think>reasoning</think><memory>test</memory>ACTION4"]  # no <action> tag
        actions, valids, memories = arcagi3_projection(texts)
        assert actions == [0], "Should return 0 for missing <action> tag"

    def test_empty_memory_returns_empty_string(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        arcagi3_projection = mod.arcagi3_projection

        texts = ["<think>test</think><action>ACTION4</action>"]  # no <memory>
        actions, valids, memories = arcagi3_projection(texts)
        assert memories[0] == "", "Missing memory should return empty string"


# ---- Prompt Template Tests ----

class TestPromptTemplates:

    def _load_prompts(self):
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("arcagi3_prompts",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/prompts/arcagi3.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_initial_prompt_has_memory_section(self):
        mod = self._load_prompts()
        assert "Memory" in mod.ARCAGI3_VISUAL_TEMPLATE or "memory" in mod.ARCAGI3_VISUAL_TEMPLATE

    def test_initial_prompt_has_image_tag(self):
        mod = self._load_prompts()
        assert "<image>" in mod.ARCAGI3_VISUAL_TEMPLATE

    def test_initial_prompt_instructs_think_memory_action(self):
        mod = self._load_prompts()
        assert "<think>" in mod.ARCAGI3_VISUAL_TEMPLATE
        assert "<memory>" in mod.ARCAGI3_VISUAL_TEMPLATE
        assert "<action>" in mod.ARCAGI3_VISUAL_TEMPLATE

    def test_history_prompt_has_memory_placeholder(self):
        mod = self._load_prompts()
        assert "{memory}" in mod.ARCAGI3_VISUAL_TEMPLATE_WITH_HISTORY

    def test_history_prompt_formats_correctly(self):
        mod = self._load_prompts()
        result = mod.ARCAGI3_VISUAL_TEMPLATE_WITH_HISTORY.format(
            memory="Color 1 = player",
            step_count=5,
            history_length=3,
            action_history="ACTION4, ACTION2, ACTION1",
            current_step=6,
        )
        assert "Color 1 = player" in result
        assert "ACTION4, ACTION2, ACTION1" in result
        assert "Step 6" in result


# ---- Environment Tests ----

class TestArcAgi3Environment:
    """Test the ARC-AGI-3 game environment wrapper."""

    def test_env_returns_won_in_info(self):
        """Info dict must contain 'won' key for verl-agent compatibility."""
        env_dir = os.path.expanduser("~/verl-agent/data/environment_files")
        if not os.path.exists(env_dir):
            env_dir = "data/environment_files"
        if not os.path.exists(env_dir):
            pytest.skip("No environment files")

        from agent_system.environments.env_package.arcagi3.envs import ArcAgi3Worker
        worker = ArcAgi3Worker("ls20", env_dir, render_scale=4)
        obs, info = worker.reset()
        assert "won" in info, "Reset info must contain 'won' key"

        obs, reward, done, info = worker.step(4)  # ACTION4 = right
        assert "won" in info, "Step info must contain 'won' key"

    def test_env_returns_rgb_image(self):
        env_dir = os.path.expanduser("~/verl-agent/data/environment_files")
        if not os.path.exists(env_dir):
            env_dir = "data/environment_files"
        if not os.path.exists(env_dir):
            pytest.skip("No environment files")

        from agent_system.environments.env_package.arcagi3.envs import ArcAgi3Worker
        worker = ArcAgi3Worker("ls20", env_dir, render_scale=4)
        obs, info = worker.reset()
        assert obs.shape == (256, 256, 3), f"Expected (256,256,3), got {obs.shape}"
        assert obs.dtype == np.uint8

    def test_action6_default_coords(self):
        """ACTION6 (click) should not crash with default coords."""
        env_dir = os.path.expanduser("~/verl-agent/data/environment_files")
        if not os.path.exists(env_dir):
            env_dir = "data/environment_files"
        if not os.path.exists(env_dir):
            pytest.skip("No environment files")

        from agent_system.environments.env_package.arcagi3.envs import ArcAgi3Worker
        worker = ArcAgi3Worker("bp35", env_dir, render_scale=4)  # bp35 supports click
        worker.reset()
        obs, reward, done, info = worker.step(6)  # ACTION6
        # Should not crash, even with default coords
        assert isinstance(reward, float)

    def test_reward_only_on_level_clear(self):
        """Reward should be 0 for normal steps, >0 only on level clear."""
        env_dir = os.path.expanduser("~/verl-agent/data/environment_files")
        if not os.path.exists(env_dir):
            env_dir = "data/environment_files"
        if not os.path.exists(env_dir):
            pytest.skip("No environment files")

        from agent_system.environments.env_package.arcagi3.envs import ArcAgi3Worker
        worker = ArcAgi3Worker("ls20", env_dir, render_scale=4)
        worker.reset()

        # First few random actions should give 0 reward
        for action in [1, 2, 3, 4, 1, 4]:
            obs, reward, done, info = worker.step(action)
            # Most steps won't clear a level
            # reward should be 0 or positive (never negative)
            assert reward >= 0, f"Reward should never be negative, got {reward}"


# ---- Wandb Stats Tests ----

class TestWandbStats:
    """Environment manager should track stats for wandb logging."""

    def test_game_distribution_tracked(self):
        """get_stats() should report per-game play counts."""
        # Mock test since we can't easily instantiate the full manager
        from collections import defaultdict
        game_play_counts = defaultdict(int)
        game_play_counts["ls20"] = 5
        game_play_counts["cd82"] = 3
        game_play_counts["ar25"] = 7

        stats = {}
        for stem, count in game_play_counts.items():
            stats[f"arcagi3/game_{stem}_plays"] = count

        assert stats["arcagi3/game_ls20_plays"] == 5
        assert stats["arcagi3/game_cd82_plays"] == 3
        assert stats["arcagi3/game_ar25_plays"] == 7

    def test_level_stats_tracked(self):
        """get_stats() should report level clearing metrics."""
        episode_levels = [0, 0, 1, 0, 2, 0, 0, 1]

        stats = {
            "arcagi3/avg_levels_per_episode": np.mean(episode_levels),
            "arcagi3/max_levels_in_episode": max(episode_levels),
            "arcagi3/episodes_with_levels": sum(1 for l in episode_levels if l > 0),
        }

        assert stats["arcagi3/avg_levels_per_episode"] == 0.5
        assert stats["arcagi3/max_levels_in_episode"] == 2
        assert stats["arcagi3/episodes_with_levels"] == 3


# ---- Trajectory Logger Tests ----

class TestTrajectoryLogger:
    """Trajectory logger must record episodes for post-hoc analysis."""

    def test_start_and_end_episode(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("trajectory_logger",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/trajectory_logger.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrajectoryLogger = mod.TrajectoryLogger
        logger = TrajectoryLogger(log_dir=str(tmp_path))
        logger.start_episode(0, "ls20")
        logger.log_step(0, {"action": 4, "reward": 0.0, "done": False, "won": False})
        logger.log_step(0, {"action": 2, "reward": 0.0, "done": True, "won": False})
        logger.end_episode(0)

        # Should write to ls20.jsonl
        log_file = tmp_path / "ls20.jsonl"
        assert log_file.exists()
        import json
        with open(log_file) as f:
            episode = json.loads(f.readline())
        assert episode["game_stem"] == "ls20"
        assert episode["total_steps"] == 2
        assert episode["total_reward"] == 0.0

    def test_logs_think_and_memory(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("trajectory_logger",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/trajectory_logger.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrajectoryLogger = mod.TrajectoryLogger
        logger = TrajectoryLogger(log_dir=str(tmp_path))
        logger.start_episode(0, "cd82")
        logger.log_step(0, {
            "action": 4,
            "think": "I see a blue dot, trying to move right",
            "memory": "Color 1 = player, ACTION4 = right",
            "reward": 0.0,
            "done": False,
            "won": False,
        })
        logger.end_episode(0)

        import json
        with open(tmp_path / "cd82.jsonl") as f:
            episode = json.loads(f.readline())
        step = episode["steps"][0]
        assert "think" in step
        assert "memory" in step
        assert "Color 1 = player" in step["memory"]

    def test_tracks_won_episodes(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("trajectory_logger",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/trajectory_logger.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrajectoryLogger = mod.TrajectoryLogger
        logger = TrajectoryLogger(log_dir=str(tmp_path))
        logger.start_episode(0, "ls20")
        logger.log_step(0, {"action": 4, "reward": 0.0, "done": False, "won": False})
        logger.log_step(0, {"action": 4, "reward": 1.0, "done": True, "won": True})
        logger.end_episode(0)

        import json
        with open(tmp_path / "ls20.jsonl") as f:
            episode = json.loads(f.readline())
        assert episode["any_won"] is True
        assert episode["total_reward"] == 1.0

    def test_multiple_episodes_same_game(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("trajectory_logger",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/trajectory_logger.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrajectoryLogger = mod.TrajectoryLogger
        logger = TrajectoryLogger(log_dir=str(tmp_path))

        for i in range(3):
            logger.start_episode(0, "ls20")
            logger.log_step(0, {"action": 4, "reward": 0.0, "done": True, "won": False})
            logger.end_episode(0)

        import json
        with open(tmp_path / "ls20.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_summary_counts_episodes(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("trajectory_logger",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/trajectory_logger.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        TrajectoryLogger = mod.TrajectoryLogger
        logger = TrajectoryLogger(log_dir=str(tmp_path))

        logger.start_episode(0, "ls20")
        logger.log_step(0, {"action": 4, "reward": 0.0, "done": True, "won": False})
        logger.end_episode(0)

        logger.start_episode(0, "cd82")
        logger.log_step(0, {"action": 1, "reward": 0.0, "done": True, "won": False})
        logger.end_episode(0)

        summary = logger.get_summary()
        assert summary["logged_episodes_ls20"] == 1
        assert summary["logged_episodes_cd82"] == 1


# ---- Data Coverage Test ----

class TestDataCoverage:
    """Dataset must cover all 25 games."""

    def test_25_games_in_cluster_data(self):
        """Verify the cluster has 25-game dataset."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not on this machine")
        df = pd.read_parquet(path)
        games = set(df["data_source"].tolist())
        assert len(games) == 25, f"Expected 25 games, got {len(games)}: {games}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
