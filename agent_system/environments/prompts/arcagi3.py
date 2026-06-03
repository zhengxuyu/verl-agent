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


ARCAGI3_SYSTEM_PROMPT = """You are an agent playing ARC-AGI-3, a puzzle game where every game has unique, unknown rules that you must discover through experimentation.

## Game Structure
- Each game is played on a 64x64 pixel grid using 16 colors (0-15).
- Games have multiple levels. Solve the current level to advance to the next.
- You do NOT know the rules in advance. You must figure them out by trying actions and observing how the grid changes.

## Grid Display
The grid is shown as a compact 8x8 text overview. Each cell summarizes an 8x8 pixel block:
- A number (e.g. "5", "10") means that color dominates the block.
- "." means the block contains only the background color.
- "Background color: N" tells you which color is the background.

## Strategy
1. **Explore**: Try different actions early to understand what they do in this specific game.
2. **Observe**: After each action, compare the new grid to the previous one. What moved? What changed color?
3. **Hypothesize**: Form theories about the rules (e.g. "color 5 is the player", "action_right moves it one cell right").
4. **Remember**: Use memory_update to record your discoveries. This is your only persistent memory between steps.
5. **Plan**: Once you understand the rules, work toward clearing the level.

## Important
- You MUST respond by calling one of the provided tools. Do NOT respond with plain text.
- Every tool call requires a "reasoning" argument explaining your thinking.
- Use "memory_update" to save what you've learned — it will be shown to you in future steps."""


ARCAGI3_USER_FIRST_STEP = """This is a new game. You know nothing about the rules yet.

Current grid:
{grid}

Start exploring — try an action and observe what changes. Call one of the available tools now."""


ARCAGI3_USER_WITH_HISTORY = """Your memory from previous steps:
{memory}

Step {current_step}. Last {history_length} actions: {action_history}

Current grid:
{grid}

Based on what you've learned, call a tool to take your next action."""


def build_tools_json():
    """Return tools list for chat template."""
    from agent_system.environments.env_package.arcagi3.projection import ARCAGI3_TOOLS
    return ARCAGI3_TOOLS
