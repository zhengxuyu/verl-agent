"""ARC-AGI-3 environment for verl-agent.

Each worker holds one game instance. Games are 64x64 pixel grids with
7 possible actions. Observation is a text grid (8x8 overview).

Reward: only on level clear, max(0.1, (baseline/actual)^2) capped at 1.15^2.
Episode ends on: WIN, GAME_OVER (death), or stuck (50 steps no change).
"""

import ray
import gymnasium as gym
import glob
import json
import importlib.util
import numpy as np


def grid_to_text(frame, scale_to=8):
    """Convert 64x64 grid to compact 8x8 text overview."""
    arr = np.array(frame)
    bg = int(np.bincount(arr.flatten()).argmax())
    block = 64 // scale_to
    rows = []
    for by in range(scale_to):
        row = []
        for bx in range(scale_to):
            blk = arr[by*block:(by+1)*block, bx*block:(bx+1)*block]
            non_bg = blk[blk != bg]
            if len(non_bg) > 0:
                vals, cnts = np.unique(non_bg, return_counts=True)
                row.append(str(int(vals[cnts.argmax()])))
            else:
                row.append(".")
        rows.append(" ".join(row))
    return "Background color: " + str(bg) + "\n" + "\n".join(rows)


class ArcAgi3Worker:
    """Ray remote actor wrapping one ARC-AGI-3 game instance."""

    MAX_NO_CHANGE = 50  # end episode after 50 steps with no frame change

    def __init__(self, game_stem, env_dir, render_scale=4):
        self.game_stem = game_stem
        self.game = None
        self.baseline_actions = []
        self.levels_cleared = 0
        self.steps_since_level = 0
        self.total_steps = 0
        self.consecutive_no_change = 0
        self.prev_frame_hash = None

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
        """Reset game and return initial text grid observation."""
        self.game = self._game_class()
        result = self.game.perform_action(
            self._ActionInput(id=self._GameAction.RESET))

        self.levels_cleared = 0
        self.steps_since_level = 0
        self.total_steps = 0
        self.consecutive_no_change = 0

        frame = np.array(result.frame[0])
        text_obs = grid_to_text(frame)
        self.prev_frame_hash = frame.tobytes()
        info = {
            "available_actions": result.available_actions,
            "game_stem": self.game_stem,
            "won": False,
        }
        return text_obs, info

    def step(self, action):
        """Execute one action. Returns (text_obs, reward, done, info)."""
        if action == 0 or self.game is None:
            return "invalid action", 0.0, False, {"valid": False, "won": False}

        ga = getattr(self._GameAction, f"ACTION{action}", None)
        if ga is None:
            return "invalid action", 0.0, False, {"valid": False, "won": False}

        data = {}
        if action == 6:
            data = {"x": 32, "y": 32}  # default center click

        result = self.game.perform_action(self._ActionInput(id=ga, data=data))
        self.steps_since_level += 1
        self.total_steps += 1

        frame = np.array(result.frame[0])
        text_obs = grid_to_text(frame)
        reward = 0.0
        done = False

        # Win — all levels cleared
        if result.state == self._GameState.WIN:
            baseline = (self.baseline_actions[self.levels_cleared]
                       if self.levels_cleared < len(self.baseline_actions) else 50)
            ratio = baseline / max(1, self.steps_since_level)
            reward = max(0.1, min(ratio ** 2, 1.15 ** 2))
            self.levels_cleared = result.levels_completed
            done = True

        # Level clear (not final win)
        elif result.levels_completed > self.levels_cleared:
            baseline = (self.baseline_actions[self.levels_cleared]
                       if self.levels_cleared < len(self.baseline_actions) else 50)
            ratio = baseline / max(1, self.steps_since_level)
            reward = max(0.1, min(ratio ** 2, 1.15 ** 2))
            self.levels_cleared = result.levels_completed
            self.steps_since_level = 0

        # Death — episode ends
        elif result.state == self._GameState.GAME_OVER:
            done = True

        # Stuck detection
        current_hash = frame.tobytes()
        if current_hash == self.prev_frame_hash:
            self.consecutive_no_change += 1
        else:
            self.consecutive_no_change = 0
        self.prev_frame_hash = current_hash

        if self.consecutive_no_change >= self.MAX_NO_CHANGE:
            done = True

        info = {
            "levels_cleared": self.levels_cleared,
            "total_steps": self.total_steps,
            "valid": True,
            "won": done and reward > 0,
            "stuck": self.consecutive_no_change >= self.MAX_NO_CHANGE,
        }
        return text_obs, reward, done, info


class ArcAgi3MultiProcessEnv(gym.Env):
    """Ray-based parallel ARC-AGI-3 environment."""

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

        self.game_stems = sorted([
            d for d in __import__('os').listdir(env_dir)
            if __import__('os').path.isdir(__import__('os').path.join(env_dir, d))
        ])

        env_worker = ray.remote(**resources_per_worker)(ArcAgi3Worker)
        self.workers = []
        self.game_assignments = []

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
        return (
            [r[0] for r in results],
            [r[1] for r in results],
            [r[2] for r in results],
            [r[3] for r in results],
        )

    def reset(self):
        if hasattr(self, 'is_train') and self.is_train:
            np.random.shuffle(self.game_stems)

        futures = [w.reset.remote(None) for w in self.workers]
        results = ray.get(futures)
        return [r[0] for r in results], [r[1] for r in results]

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
