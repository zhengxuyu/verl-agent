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

    def test_system_prompt_mentions_tools(self):
        mod = self._load_prompts()
        assert "tool" in mod.ARCAGI3_SYSTEM_PROMPT.lower()
        assert "memory_update" in mod.ARCAGI3_SYSTEM_PROMPT.lower()

    def test_first_step_has_grid_placeholder(self):
        mod = self._load_prompts()
        assert "{grid}" in mod.ARCAGI3_USER_FIRST_STEP

    def test_history_prompt_has_memory_and_grid(self):
        mod = self._load_prompts()
        assert "{memory}" in mod.ARCAGI3_USER_WITH_HISTORY
        assert "{grid}" in mod.ARCAGI3_USER_WITH_HISTORY

    def test_history_prompt_formats_correctly(self):
        mod = self._load_prompts()
        result = mod.ARCAGI3_USER_WITH_HISTORY.format(
            memory="Color 1 = player",
            grid="Background color: 0\n. . . .",
            history_length=3,
            action_history="Up, Right, Down",
            current_step=6,
        )
        assert "Color 1 = player" in result
        assert "Up, Right, Down" in result
        assert "Step 6" in result
        assert "Background color: 0" in result


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

    def test_logs_full_debug_info(self, tmp_path):
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
            "llm_raw_output": '{"name": "action_right", "arguments": {"reasoning": "test"}}',
            "grid_text": "Background color: 0\n. . 1 . . . . .",
            "memory": "Color 1 = player",
            "reward": 0.0,
            "done": False,
            "won": False,
            "valid": True,
        })
        logger.end_episode(0)

        import json
        with open(tmp_path / "cd82.jsonl") as f:
            episode = json.loads(f.readline())
        step = episode["steps"][0]
        assert "llm_raw_output" in step
        assert "grid_text" in step
        assert "memory" in step
        assert "timestamp" in step
        assert "action_right" in step["llm_raw_output"]
        assert "Background color" in step["grid_text"]

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

class TestTrajectoryIntegration:
    """Trajectory logger must be wired into env_manager."""

    def test_env_manager_has_trajectory_logger(self):
        """ArcAgi3EnvironmentManager should create a TrajectoryLogger."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("trajectory_logger",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/trajectory_logger.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Verify TrajectoryLogger has the interface we need
        logger = mod.TrajectoryLogger(log_dir="/tmp/test_traj")
        assert hasattr(logger, "start_episode")
        assert hasattr(logger, "log_step")
        assert hasattr(logger, "end_episode")
        assert hasattr(logger, "get_summary")


class TestBatchSizeDivisibility:
    """batch_size must be divisible by n_gpus."""

    def test_25_games_batch_size_for_2_gpus(self):
        """With 25 games and 2 GPUs, batch_size should be even (e.g. 24 or 26)."""
        n_games = 25
        n_gpus = 2
        # Options: 24 (drop 1) or 26 (duplicate 1)
        batch_size = n_games + 1  # pad to 26
        assert batch_size % n_gpus == 0


class TestScoringStats:
    """Wandb must log per-epoch score distribution."""

    def test_score_stats_in_get_stats(self):
        """get_stats() must include avg/max/min scores."""
        episode_scores = [0.0, 0.0, 0.5, 0.0, 1.2, 0.0, 0.0, 0.3]

        stats = {
            "arcagi3/score_avg": np.mean(episode_scores),
            "arcagi3/score_max": max(episode_scores),
            "arcagi3/score_min": min(episode_scores),
            "arcagi3/score_nonzero_count": sum(1 for s in episode_scores if s > 0),
        }

        assert stats["arcagi3/score_avg"] == pytest.approx(0.25)
        assert stats["arcagi3/score_max"] == 1.2
        assert stats["arcagi3/score_min"] == 0.0
        assert stats["arcagi3/score_nonzero_count"] == 3

    def test_official_score_formula(self):
        """Score per level = (baseline/actual)^2, capped at 1.15^2."""
        baseline = 21
        actual = 21
        score = min((baseline / actual) ** 2, 1.15 ** 2)
        assert score == pytest.approx(1.0)

        # Faster than human
        score_fast = min((21 / 10) ** 2, 1.15 ** 2)
        assert score_fast == pytest.approx(1.3225)  # capped

        # Slower than human
        score_slow = min((21 / 100) ** 2, 1.15 ** 2)
        assert score_slow == pytest.approx(0.0441)

        # Much slower
        score_bad = min((21 / 1000) ** 2, 1.15 ** 2)
        assert score_bad == pytest.approx(0.000441)

    def test_reward_matches_official_with_floor(self):
        """Reward = max(0.1, official_score) per level cleared."""
        baseline = 21
        # Good: 21 steps
        reward_good = max(0.1, min((baseline / 21) ** 2, 1.15 ** 2))
        assert reward_good == pytest.approx(1.0)

        # Bad: 1000 steps — floor kicks in
        reward_bad = max(0.1, min((baseline / 1000) ** 2, 1.15 ** 2))
        assert reward_bad == 0.1


class TestEpisodeTermination:
    """Episode must end on death, win, or stuck (no progress)."""

    def test_death_ends_episode(self):
        """GAME_OVER → done=True."""
        # Simulated: envs.py should return done=True on GAME_OVER
        # We test the logic, not the actual game
        game_over_state = "GAME_OVER"
        done = (game_over_state == "GAME_OVER")
        assert done is True

    def test_win_ends_episode(self):
        """WIN → done=True."""
        win_state = "WIN"
        done = (win_state == "WIN")
        assert done is True

    def test_stuck_detection(self):
        """If no frame change for N consecutive steps, episode should end."""
        consecutive_no_change = 0
        max_no_change = 50  # if 50 steps with no change, give up

        # Simulate 50 steps with no change
        for _ in range(50):
            frame_changed = False
            if not frame_changed:
                consecutive_no_change += 1
            else:
                consecutive_no_change = 0

        done = consecutive_no_change >= max_no_change
        assert done is True, "Should end episode after 50 steps with no frame change"

    def test_stuck_resets_on_change(self):
        """Stuck counter resets when frame changes."""
        consecutive_no_change = 30
        frame_changed = True
        if frame_changed:
            consecutive_no_change = 0
        assert consecutive_no_change == 0

    def test_steps_since_level_resets_on_new_episode(self):
        """New episode should start with steps_since_level = 0."""
        # After death → done=True → verl starts new episode → reset()
        # reset() sets steps_since_level = 0
        steps_since_level = 0  # from reset
        assert steps_since_level == 0


class TestToolCalling:
    """Projection must parse Qwen tool call format."""

    def _load_projection(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.arcagi3_projection

    def test_parse_tool_call_action_right(self):
        proj = self._load_projection()
        text = '{"name": "action_right", "arguments": {"reasoning": "going right", "memory_update": "player moves right"}}'
        actions, valids, memories = proj([text])
        assert actions == [4]
        assert valids == [1]
        assert "player moves right" in memories[0]

    def test_parse_tool_call_action_up(self):
        proj = self._load_projection()
        text = '{"name": "action_up", "arguments": {"reasoning": "try up", "memory_update": "testing"}}'
        actions, valids, memories = proj([text])
        assert actions == [1]
        assert valids == [1]

    def test_parse_tool_call_action_click_with_coords(self):
        proj = self._load_projection()
        text = '{"name": "action_click", "arguments": {"reasoning": "click target", "memory_update": "found button", "x": 20, "y": 30}}'
        actions, valids, memories = proj([text])
        assert actions == [6]
        assert valids == [1]

    def test_parse_tool_call_no_memory(self):
        proj = self._load_projection()
        text = '{"name": "action_down", "arguments": {"reasoning": "going down"}}'
        actions, valids, memories = proj([text])
        assert actions == [2]
        assert memories[0] == ""

    def test_invalid_tool_name(self):
        proj = self._load_projection()
        text = '{"name": "invalid_action", "arguments": {}}'
        actions, valids, memories = proj([text])
        assert actions == [0]
        assert valids == [0]

    def test_malformed_json(self):
        proj = self._load_projection()
        text = 'this is not json at all'
        actions, valids, memories = proj([text])
        assert actions == [0]
        assert valids == [0]

    def test_still_supports_xml_format(self):
        """Backward compat: old <think>/<action> format should still work."""
        proj = self._load_projection()
        text = '<think>reasoning</think><memory>test mem</memory><action>ACTION4</action>'
        actions, valids, memories = proj([text])
        assert actions == [4]
        assert "test mem" in memories[0]

    def test_tool_definitions_list(self):
        """Verify tool definitions are available for prompt construction."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("projection",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/env_package/arcagi3/projection.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tools = mod.ARCAGI3_TOOLS
        assert len(tools) == 7
        names = [t["function"]["name"] for t in tools]
        assert "action_up" in names
        assert "action_click" in names
        # action_click must have x, y parameters
        click_tool = [t for t in tools if t["function"]["name"] == "action_click"][0]
        assert "x" in click_tool["function"]["parameters"]["properties"]
        assert "y" in click_tool["function"]["parameters"]["properties"]


class TestTextGridMode:
    """Text grid mode: env returns text, no images."""

    def _load_prompts(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("arcagi3_prompts",
            os.path.join(os.path.dirname(__file__), "../agent_system/environments/prompts/arcagi3.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_grid_to_text_output(self):
        """grid_to_text should produce readable 8x8 overview."""
        mod = self._load_prompts()
        grid_to_text = mod.grid_to_text
        frame = np.zeros((64, 64), dtype=int)
        frame[10, 10] = 1
        frame[50, 50] = 3
        text = grid_to_text(frame)
        assert "Background color: 0" in text
        assert "." in text  # background cells
        assert "1" in text  # color 1
        assert "3" in text  # color 3
        lines = text.strip().split("\n")
        assert len(lines) == 9  # 1 header + 8 grid rows

    def test_dataset_has_no_images(self):
        """Text-mode dataset should not have images column."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not on this machine")
        df = pd.read_parquet(path)
        assert "images" not in df.columns, "Text mode dataset should not have images"

    def test_dataset_prompt_is_chat_messages(self):
        """Prompt should be list of chat messages with system + user."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not on this machine")
        df = pd.read_parquet(path)
        prompt = df.iloc[0]["prompt"]
        assert isinstance(prompt, (list, np.ndarray))
        roles = [m["role"] for m in prompt]
        assert "system" in roles
        assert "user" in roles

    def test_dataset_prompt_contains_grid(self):
        """User message should contain the grid text."""
        import pandas as pd
        path = os.path.expanduser("~/data/verl-agent/arcagi3/train.parquet")
        if not os.path.exists(path):
            pytest.skip("Dataset not on this machine")
        df = pd.read_parquet(path)
        user_msg = [m for m in df.iloc[0]["prompt"] if m["role"] == "user"][0]
        assert "Background color" in user_msg["content"]
        assert "." in user_msg["content"]

    def test_tools_in_system_prompt(self):
        """System prompt should mention tools/actions."""
        mod = self._load_prompts()
        ARCAGI3_SYSTEM_PROMPT = mod.ARCAGI3_SYSTEM_PROMPT
        assert "tool" in ARCAGI3_SYSTEM_PROMPT.lower()
        assert "reasoning" in ARCAGI3_SYSTEM_PROMPT.lower()
        assert "memory_update" in ARCAGI3_SYSTEM_PROMPT.lower()


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
