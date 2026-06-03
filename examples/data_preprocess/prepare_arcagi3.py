"""Prepare ARC-AGI-3 training data — text grid, tool calling format.

Each sample = one game's initial state as text grid prompt.
No images — the 64x64 grid is represented as 8x8 text overview.

Usage:
  python -m examples.data_preprocess.prepare_arcagi3 \
    --env_dir data/environment_files \
    --output_dir ~/data/verl-agent/arcagi3
"""

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd


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
    return f"Background color: {bg}\n" + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_dir", default="data/environment_files")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    env_dir = os.path.expanduser(args.env_dir)
    output_dir = args.output_dir or os.path.join(
        os.path.expanduser("~"), "data", "verl-agent", "arcagi3")
    os.makedirs(output_dir, exist_ok=True)

    from arcengine import ARCBaseGame, GameAction, ActionInput

    game_stems = sorted([
        d for d in os.listdir(env_dir)
        if os.path.isdir(os.path.join(env_dir, d))
    ])
    print(f"Found {len(game_stems)} games")

    system_prompt = (
        "You are playing a puzzle game on a 64x64 pixel grid with 16 colors (0-15). "
        "You must figure out the rules by trying actions and observing what changes. "
        "Clear all levels to win.\n\n"
        "The grid is shown as an 8x8 overview where each cell represents an 8x8 block. "
        "\".\" means background color only.\n\n"
        "You have tools to take actions. Each tool requires a \"reasoning\" argument "
        "where you explain your thinking, and an optional \"memory_update\" argument "
        "where you record what you've learned for future steps."
    )

    samples = []
    for stem in game_stems:
        py_files = glob.glob(f"{env_dir}/{stem}/*/{stem}.py")
        if not py_files:
            continue
        spec = importlib.util.spec_from_file_location(stem, py_files[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gc = None
        for v in vars(mod).values():
            if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame:
                gc = v
                break
        if gc is None:
            continue

        try:
            game = gc()
            result = game.perform_action(ActionInput(id=GameAction.RESET))
            frame = np.array(result.frame[0])
            grid_text = grid_to_text(frame)

            user_msg = (
                f"This is a new game. You know nothing about the rules yet.\n\n"
                f"Current grid:\n{grid_text}\n\n"
                f"Explore by trying different actions to understand what they do."
            )

            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]

            from agent_system.environments.env_package.arcagi3.projection import ARCAGI3_TOOLS
            # Store extra_info as JSON string to prevent parquet/pandas
            # from converting nested dicts/lists to numpy arrays
            samples.append({
                "prompt": prompt,
                "data_source": stem,
                "extra_info": json.dumps({
                    "tools_kwargs": {"tools": ARCAGI3_TOOLS},
                    "need_tools_kwargs": True,
                }),
            })
            print(f"  OK: {stem}")
        except Exception as e:
            print(f"  FAIL: {stem} — {e}")

    df = pd.DataFrame(samples)

    # Split: last 5 games for validation, rest for training
    val_stems = sorted([s["data_source"] for s in samples])[-5:]
    train_df = df[~df["data_source"].isin(val_stems)].reset_index(drop=True)
    val_df = df[df["data_source"].isin(val_stems)].reset_index(drop=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(test_path, index=False)

    print(f"\nTrain: {len(train_df)} games — {list(train_df['data_source'])}")
    print(f"Val:   {len(val_df)} games — {list(val_df['data_source'])}")

    # Verify: extra_info is a JSON string, survives parquet round-trip
    df_check = pd.read_parquet(train_path)
    ei = df_check.iloc[0]["extra_info"]
    assert isinstance(ei, str), f"extra_info should be string, got {type(ei)}"
    parsed = json.loads(ei)
    assert "tools_kwargs" in parsed, "Missing tools_kwargs"
    json.dumps(parsed)  # verify JSON serializable
    print(f"Verified parquet round-trip OK")


if __name__ == "__main__":
    main()
