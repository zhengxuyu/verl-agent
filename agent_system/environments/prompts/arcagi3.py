"""Prompt templates for ARC-AGI-3 games — text grid + tool calling."""

import json


def grid_to_text(frame, scale_to=8):
    """Convert 64x64 grid to compact text. Each cell = most common color in block."""
    import numpy as np
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


ARCAGI3_SYSTEM_PROMPT = """You are playing a puzzle game on a 64x64 pixel grid with 16 colors (0-15). You must figure out the rules by trying actions and observing what changes. Clear all levels to win.

The grid is shown as an 8x8 overview where each cell represents an 8x8 block. "." means background color only.

You have tools to take actions. Each tool requires a "reasoning" argument where you explain your thinking, and an optional "memory_update" argument where you record what you've learned. The memory_update will be shown to you in future steps — use it to track:
- What each color represents (player, wall, goal, etc.)
- What each action does
- Rules and patterns you've discovered
- Your current plan"""


ARCAGI3_USER_FIRST_STEP = """This is a new game. You know nothing about the rules yet.

Current grid:
{grid}

Explore by trying different actions to understand what they do."""


ARCAGI3_USER_WITH_HISTORY = """Your memory from previous steps:
{memory}

Step {current_step}. Last {history_length} actions: {action_history}

Current grid:
{grid}

Based on what you've learned, choose your next action."""


def build_tools_json():
    """Return tools list for chat template."""
    from agent_system.environments.env_package.arcagi3.projection import ARCAGI3_TOOLS
    return ARCAGI3_TOOLS
