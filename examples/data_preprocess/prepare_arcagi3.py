"""Prepare ARC-AGI-3 training data for verl-agent.

Generates initial game states as parquet files:
- Each row = one game, with initial frame rendered as RGB image
- The dynamic trainer will rollout from these initial states

Usage:
  python -m examples.data_preprocess.prepare_arcagi3 \
    --env_dir data/environment_files \
    --train_data_size 24 \
    --val_data_size 24
"""

import argparse
import glob
import importlib.util
import os
import io
import random

import numpy as np
import pandas as pd
from PIL import Image

# ARC 16-color palette
ARC_PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (0, 0, 0), (128, 0, 128),
    (0, 128, 128), (128, 128, 0), (255, 192, 203), (255, 255, 255),
]


def frame_to_png_bytes(frame, scale=4):
    """Convert 64x64 grid to PNG bytes."""
    arr = np.array(frame)
    h, w = arr.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in enumerate(ARC_PALETTE):
        mask = arr == c
        img[mask] = rgb
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def load_game_class(game_stem, env_dir):
    from arcengine import ARCBaseGame
    py_files = glob.glob(f"{env_dir}/{game_stem}/*/{game_stem}.py")
    if not py_files:
        return None
    spec = importlib.util.spec_from_file_location(game_stem, py_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for v in vars(mod).values():
        if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame:
            return v
    return None


def generate_initial_states(env_dir, num_samples):
    """Generate initial game states with rendered frames."""
    from arcengine import GameAction, ActionInput

    game_stems = sorted([
        d for d in os.listdir(env_dir)
        if os.path.isdir(os.path.join(env_dir, d))
    ])

    samples = []
    for i in range(num_samples):
        stem = game_stems[i % len(game_stems)]
        gc = load_game_class(stem, env_dir)
        if gc is None:
            continue

        try:
            game = gc()
            result = game.perform_action(ActionInput(id=GameAction.RESET))
            frame = np.array(result.frame[0])
            avail = result.available_actions

            # Render frame to PNG
            png_bytes = frame_to_png_bytes(frame, scale=4)

            avail_str = ", ".join(f"ACTION{a}" for a in avail if a != 0)
            prompt = (
                "You are playing a puzzle game on a 64x64 pixel grid. "
                "Figure out the rules by trying actions and observing what changes. "
                "Clear all levels to win.\n\n"
                f"Available actions: {avail_str}\n\n"
                "The current game state is shown in the image. <image>\n\n"
                "Choose your action. First reason about what you see in "
                "<think></think> tags, then output your action in <action></action> tags."
            )

            samples.append({
                "images": [png_bytes],
                "prompt": prompt,
                "data_source": stem,
            })
        except Exception as e:
            print(f"  Failed {stem}: {e}")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_dir", default="data/environment_files")
    parser.add_argument("--train_data_size", type=int, default=24)
    parser.add_argument("--val_data_size", type=int, default=24)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.expanduser("~"), "data", "verl-agent", "arcagi3")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating ARC-AGI-3 training data...")
    print(f"  env_dir: {args.env_dir}")
    print(f"  train: {args.train_data_size}, val: {args.val_data_size}")

    train_samples = generate_initial_states(args.env_dir, args.train_data_size)
    val_samples = generate_initial_states(args.env_dir, args.val_data_size)

    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"  Saved train: {train_path} ({len(train_df)} samples)")
    print(f"  Saved val: {val_path} ({len(val_df)} samples)")


if __name__ == "__main__":
    main()
