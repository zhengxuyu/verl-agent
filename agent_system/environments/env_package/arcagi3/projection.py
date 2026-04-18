"""Parse LLM output: extract <action>, <memory>, and validity.

Returns 3 values: (actions, valids, memories) — unlike other envs which return 2.
"""
from typing import List, Tuple


ACTION_MAP = {
    "action1": 1, "action2": 2, "action3": 3, "action4": 4,
    "action5": 5, "action6": 6, "action7": 7,
    "up": 1, "down": 2, "left": 3, "right": 4,
    "enter": 5, "click": 6, "undo": 7,
}


def extract_tag(text, tag):
    """Extract content between <tag> and </tag>, case-insensitive search."""
    lower = text.lower()
    start = lower.find(f"<{tag.lower()}>")
    end = lower.find(f"</{tag.lower()}>")
    if start == -1 or end == -1:
        return None
    start += len(f"<{tag}>")
    return text[start:end].strip()


def arcagi3_projection(actions: List[str]) -> Tuple[list, list, list]:
    """Parse LLM outputs into ARC-AGI-3 actions + memories.

    Expected format:
        <think>reasoning</think>
        <memory>learned rules</memory>
        <action>ACTION4</action>

    Returns:
        actions: list of int (0=invalid, 1-7=ACTION1-7)
        valids: list of int (0=invalid format, 1=valid)
        memories: list of str (extracted memory text)
    """
    valids = [0] * len(actions)
    memories = [""] * len(actions)

    for i in range(len(actions)):
        original = actions[i]
        text = actions[i].lower()

        # Extract memory (case-preserving from original)
        mem = extract_tag(original, "memory")
        memories[i] = mem if mem is not None else ""

        # Extract action
        start = text.find("<action>")
        end = text.find("</action>")
        try:
            if start == -1 or end == -1:
                actions[i] = 0
                continue

            content = text[start + 8:end].strip()
            matched = False
            for key, val in ACTION_MAP.items():
                if key in content:
                    actions[i] = val
                    valids[i] = 1
                    matched = True
                    break

            if not matched:
                actions[i] = 0

        except Exception:
            actions[i] = 0

        # Check <think> exists
        if "<think>" not in text or "</think>" not in text:
            valids[i] = 0

    return actions, valids, memories
