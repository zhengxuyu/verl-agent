"""Projection function: maps LLM text output to ARC-AGI-3 actions.

LLM outputs: <think>reasoning</think><action>ACTION4</action>
Maps to: integer action (1-7) for the game engine.
"""

from typing import List


def arcagi3_projection(actions: List[str]):
    """Parse LLM outputs into ARC-AGI-3 action integers.

    Expected LLM format:
        <think>some reasoning about the game state</think>
        <action>ACTION4</action>

    Or for click actions:
        <action>ACTION6 32 40</action>

    Returns:
        actions: list of int (0=invalid, 1-7=ACTION1-7)
        valids: list of int (0=invalid format, 1=valid)
    """
    action_map = {
        "action1": 1, "action2": 2, "action3": 3, "action4": 4,
        "action5": 5, "action6": 6, "action7": 7,
        "up": 1, "down": 2, "left": 3, "right": 4,
        "enter": 5, "click": 6, "undo": 7,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    }

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original = actions[i]
        text = actions[i].lower()

        # Extract <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        try:
            if start_idx == -1 or end_idx == -1:
                actions[i] = 0
                continue

            content = text[start_idx + len(start_tag):end_idx].strip()

            # Try to match action
            matched = False
            for key, val in action_map.items():
                if key in content:
                    actions[i] = val
                    valids[i] = 1
                    matched = True
                    break

            if not matched:
                actions[i] = 0

        except Exception:
            actions[i] = 0

        # Check <think>...</think> exists
        think_start = original.lower().find("<think>")
        think_end = original.lower().find("</think>")
        if think_start == -1 or think_end == -1:
            valids[i] = 0

    return actions, valids
