"""ARC-AGI-3 environment for verl-agent.

Each worker holds one game instance. Games are 64x64 pixel grids with
7 possible actions. Observation is an RGB rendering of the frame.

Reward: only on level clear, efficiency-based (fewer steps = higher reward).
"""

import ray
import gym
import glob
import json
import importlib.util
import numpy as np
from PIL import Image


# ARC-AGI-3 16-color palette → RGB
ARC_PALETTE = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),     # 1: blue
    (255, 65, 54),     # 2: red
    (46, 204, 64),     # 3: green
    (255, 220, 0),     # 4: yellow
    (170, 170, 170),   # 5: gray
    (240, 18, 190),    # 6: magenta
    (255, 133, 27),    # 7: orange
    (127, 219, 255),   # 8: light blue
    (135, 12, 37),     # 9: maroon
    (0, 0, 0),        # 10: black2
    (128, 0, 128),     # 11: purple
    (0, 128, 128),     # 12: teal
    (128, 128, 0),     # 13: olive
    (255, 192, 203),   # 14: pink
    (255, 255, 255),   # 15: white
]


def frame_to_rgb(frame, scale=4):
    """Convert 64x64 color grid to RGB image.

    Args:
        frame: 64x64 numpy array of ints 0-15
        scale: upscale factor (4 → 256x256 output)
    Returns:
        numpy array (H, W, 3) uint8
    """
    h, w = frame.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in enumerate(ARC_PALETTE):
        mask = frame == c
        img[mask] = rgb

    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return img


class ArcAgi3Worker:
    """Ray remote actor wrapping one ARC-AGI-3 game instance."""

    def __init__(self, game_stem, env_dir, render_scale=4):
        self.game_stem = game_stem
        self.render_scale = render_scale
        self.game = None
        self.baseline_actions = []
        self.levels_cleared = 0
        self.steps_since_level = 0
        self.total_steps = 0

        # Load game class
        from arcengine import ARCBaseGame, GameAction, ActionInput, GameState
        self._GameAction = GameAction
        self._ActionInput = ActionInput
        self._GameState = GameState

        py_files = glob.glob(f"{env_dir}/{game_stem}/*/{game_stem}.py")
        if not py_files:
            raise FileNotFoundError(f"Game {game_stem} not found in {env_dir}")

        spec = importlib.util.spec_from_file_location(game_stem, py_files[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._game_class = None
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame:
                self._game_class = v
                break

        # Load baseline
        meta_files = glob.glob(f"{env_dir}/{game_stem}/*/metadata.json")
        if meta_files:
            with open(meta_files[0]) as f:
                meta = json.load(f)
            self.baseline_actions = meta.get("baseline_actions", [])

    def reset(self, seed_for_reset=None):
        """Reset game and return initial RGB observation."""
        self.game = self._game_class()
        result = self.game.perform_action(
            self._ActionInput(id=self._GameAction.RESET))

        self.levels_cleared = 0
        self.steps_since_level = 0
        self.total_steps = 0

        frame = np.array(result.frame[0])
        obs = frame_to_rgb(frame, self.render_scale)
        info = {
            "available_actions": result.available_actions,
            "game_stem": self.game_stem,
        }
        return obs, info

    def step(self, action):
        """Execute one action.

        Args:
            action: int 0-7 mapping:
                0 = no-op/invalid
                1-7 = ACTION1-ACTION7

        Returns:
            obs: RGB image
            reward: float (only on level clear)
            done: bool
            info: dict
        """
        if action == 0 or self.game is None:
            # Invalid action
            frame = np.zeros((64, 64), dtype=int)
            if self.game:
                result = self.game.perform_action(
                    self._ActionInput(id=self._GameAction.RESET))
                frame = np.array(result.frame[0])
            return frame_to_rgb(frame, self.render_scale), 0.0, False, {"valid": False}

        ga = getattr(self._GameAction, f"ACTION{action}", None)
        if ga is None:
            return frame_to_rgb(np.zeros((64, 64), dtype=int), self.render_scale), 0.0, False, {"valid": False}

        data = {}
        # ACTION6 needs x,y — for now we don't support click from LLM
        # (will be handled via projection function with coordinates)

        result = self.game.perform_action(self._ActionInput(id=ga, data=data))
        self.steps_since_level += 1
        self.total_steps += 1

        frame = np.array(result.frame[0])
        obs = frame_to_rgb(frame, self.render_scale)
        reward = 0.0
        done = False

        # Win
        if result.state == self._GameState.WIN:
            baseline = (self.baseline_actions[self.levels_cleared]
                       if self.levels_cleared < len(self.baseline_actions) else 50)
            reward = max(0.1, baseline / max(1, self.steps_since_level))
            self.levels_cleared = result.levels_completed
            done = True

        # Level clear (not final win)
        elif result.levels_completed > self.levels_cleared:
            baseline = (self.baseline_actions[self.levels_cleared]
                       if self.levels_cleared < len(self.baseline_actions) else 50)
            reward = max(0.1, baseline / max(1, self.steps_since_level))
            self.levels_cleared = result.levels_completed
            self.steps_since_level = 0

        # Death — just reset, no penalty
        elif result.state == self._GameState.GAME_OVER:
            result = self.game.perform_action(
                self._ActionInput(id=self._GameAction.RESET))
            frame = np.array(result.frame[0])
            obs = frame_to_rgb(frame, self.render_scale)

        info = {
            "levels_cleared": self.levels_cleared,
            "total_steps": self.total_steps,
            "valid": True,
        }
        return obs, reward, done, info


class ArcAgi3MultiProcessEnv(gym.Env):
    """Ray-based parallel ARC-AGI-3 environment.

    Manages multiple game instances across Ray workers.
    Each group of `group_n` workers plays the same game (for GRPO/GiGPO).
    """

    def __init__(self, seed=0, env_num=1, group_n=1,
                 env_dir="data/environment_files",
                 render_scale=4,
                 resources_per_worker={"num_cpus": 0.1},
                 is_train=True):
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.env_dir = env_dir

        np.random.seed(seed)

        # Discover all available games
        self.game_stems = sorted([
            d for d in glob.os.listdir(env_dir)
            if glob.os.path.isdir(glob.os.path.join(env_dir, d))
        ])

        # Create workers: each env_num gets a random game, repeated group_n times
        env_worker = ray.remote(**resources_per_worker)(ArcAgi3Worker)
        self.workers = []
        self.game_assignments = []  # which game each worker plays

        for i in range(env_num):
            stem = self.game_stems[i % len(self.game_stems)]
            for g in range(group_n):
                worker = env_worker.remote(stem, env_dir, render_scale)
                self.workers.append(worker)
                self.game_assignments.append(stem)

    def step(self, actions):
        assert len(actions) == self.num_processes
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        done_list = [r[2] for r in results]
        info_list = [r[3] for r in results]
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Shuffle game assignments for training variety
        if hasattr(self, 'is_train') and self.is_train:
            np.random.shuffle(self.game_stems)

        futures = [w.reset.remote(None) for w in self.workers]
        results = ray.get(futures)
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def close(self):
        for w in self.workers:
            ray.kill(w)


def build_arcagi3_envs(seed=0, env_num=1, group_n=1,
                       env_dir="data/environment_files",
                       render_scale=4,
                       resources_per_worker={"num_cpus": 0.1},
                       is_train=True, **kwargs):
    return ArcAgi3MultiProcessEnv(
        seed, env_num, group_n, env_dir, render_scale,
        resources_per_worker, is_train)
