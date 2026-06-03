"""Parse LLM output: tool call format (primary) or XML tags (fallback).

Tool call format (Qwen function calling):
  {"name": "action_right", "arguments": {"reasoning": "...", "memory_update": "..."}}

XML fallback:
  <think>...</think><memory>...</memory><action>ACTION4</action>

Returns 3 values: (actions, valids, memories)
"""
import json
import re
from typing import List, Tuple


TOOL_NAME_TO_ACTION = {
    "action_up": 1,
    "action_down": 2,
    "action_left": 3,
    "action_right": 4,
    "action_enter": 5,
    "action_click": 6,
    "action_undo": 7,
}

# For XML fallback
XML_ACTION_MAP = {
    "action1": 1, "action2": 2, "action3": 3, "action4": 4,
    "action5": 5, "action6": 6, "action7": 7,
    "up": 1, "down": 2, "left": 3, "right": 4,
    "enter": 5, "click": 6, "undo": 7,
}

# Tool definitions for prompt construction (OpenAI format)
_COMMON_PARAMS = {
    "reasoning": {
        "type": "string",
        "description": "Your analysis of the current grid state, what changed since last step, and why you chose this action.",
    },
    "memory_update": {
        "type": "string",
        "description": "Record your discoveries here. This text will be shown to you in your next step. Track: what colors represent (player, wall, goal, etc.), what each action does, rules you've found, and your current plan.",
    },
}

ARCAGI3_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "action_up",
            "description": "Move up. May move the player, shift objects, scroll the view, or have other game-specific effects depending on the puzzle rules.",
            "parameters": {
                "type": "object",
                "properties": {**_COMMON_PARAMS},
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_down",
            "description": "Move down. May move the player, shift objects, scroll the view, or have other game-specific effects depending on the puzzle rules.",
            "parameters": {
                "type": "object",
                "properties": {**_COMMON_PARAMS},
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_left",
            "description": "Move left. May move the player, shift objects, scroll the view, or have other game-specific effects depending on the puzzle rules.",
            "parameters": {
                "type": "object",
                "properties": {**_COMMON_PARAMS},
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_right",
            "description": "Move right. May move the player, shift objects, scroll the view, or have other game-specific effects depending on the puzzle rules.",
            "parameters": {
                "type": "object",
                "properties": {**_COMMON_PARAMS},
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_enter",
            "description": "Confirm, select, or interact. May submit an answer, toggle a cell, activate a mechanism, or trigger game-specific interactions.",
            "parameters": {
                "type": "object",
                "properties": {**_COMMON_PARAMS},
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_click",
            "description": "Click at a specific (x, y) position on the 64x64 grid. May select a cell, place a color, toggle a pixel, or interact with an object at that location.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_COMMON_PARAMS,
                    "x": {"type": "integer", "description": "X coordinate (0-63, left to right)"},
                    "y": {"type": "integer", "description": "Y coordinate (0-63, top to bottom)"},
                },
                "required": ["reasoning", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_undo",
            "description": "Undo the last action and revert the grid to its previous state. Useful when an action had an undesired effect.",
            "parameters": {
                "type": "object",
                "properties": {**_COMMON_PARAMS},
                "required": ["reasoning"],
            },
        },
    },
]


def _try_parse_gemma4_call(text: str):
    """Try to parse Gemma 4 style: call:action_right{reasoning:...,memory_update:...}"""
    m = re.search(r'call:(\w+)\{(.+?)\}', text, re.DOTALL)
    if not m:
        return None
    name = m.group(1)
    if name not in TOOL_NAME_TO_ACTION:
        return None
    args_str = m.group(2)
    memory = ""
    mem_match = re.search(r'memory_update[:\s]*([^,}]+)', args_str)
    if mem_match:
        memory = mem_match.group(1).strip()
    return TOOL_NAME_TO_ACTION[name], memory, 1


def _try_parse_qwen35_xml(text: str):
    """Parse Qwen3.5/qwen3_coder XML format:
    <tool_call>
    <function=action_right>
    <parameter=reasoning>...</parameter>
    <parameter=memory_update>...</parameter>
    </function>
    </tool_call>
    """
    # Find <function=NAME> inside <tool_call> block
    tc_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not tc_match:
        # Also try without closing tag (truncated output)
        tc_match = re.search(r'<tool_call>(.*)', text, re.DOTALL)
    if not tc_match:
        return None

    tc_body = tc_match.group(1)
    func_match = re.search(r'<function=(\w+)>', tc_body)
    if not func_match:
        return None

    name = func_match.group(1)
    if name not in TOOL_NAME_TO_ACTION:
        return None

    # Extract parameters
    memory = ""
    x_val, y_val = None, None
    for param_match in re.finditer(
        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', tc_body, re.DOTALL
    ):
        pname = param_match.group(1)
        pval = param_match.group(2).strip()
        if pname == "memory_update":
            memory = pval
        elif pname == "x":
            try: x_val = int(pval)
            except ValueError: pass
        elif pname == "y":
            try: y_val = int(pval)
            except ValueError: pass

    return TOOL_NAME_TO_ACTION[name], memory, 1, x_val, y_val


def _try_parse_tool_call(text: str):
    """Try to parse tool call. Supports Qwen3.5 XML, Gemma4, and JSON formats.
    Returns (action_int, memory, valid) or None."""
    text = text.strip()

    # Strip <think>...</think> block if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Try Qwen3.5 XML format first: <tool_call><function=name><parameter=...>
    qwen35_result = _try_parse_qwen35_xml(text)
    if qwen35_result is not None:
        action_int, memory, valid, x_val, y_val = qwen35_result
        return action_int, memory, valid

    # Try Gemma 4 format: call:function_name{key:value,...}
    gemma_result = _try_parse_gemma4_call(text)
    if gemma_result is not None:
        return gemma_result

    # Try JSON format: <tool_call>{"name":...}</tool_call> or plain {"name":...}
    json_str = text
    for tag in ["<tool_call>", "<|tool_call|>"]:
        if tag in text.lower():
            start = text.lower().find(tag) + len(tag)
            end_tag = tag.replace("<", "</")
            end = text.lower().find(end_tag)
            if end == -1:
                end = len(text)
            json_str = text[start:end].strip()
            break

    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}")
    if brace_start == -1 or brace_end == -1:
        return None

    json_str = json_str[brace_start:brace_end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    name = data.get("name", "")
    if name not in TOOL_NAME_TO_ACTION:
        return None

    action_int = TOOL_NAME_TO_ACTION[name]
    args = data.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    memory = args.get("memory_update", "")
    return action_int, memory, 1


def _try_parse_xml(text: str):
    """Fallback: parse <think>/<memory>/<action> XML tags."""
    lower = text.lower()

    # Extract memory
    mem_start = lower.find("<memory>")
    mem_end = lower.find("</memory>")
    memory = ""
    if mem_start != -1 and mem_end != -1:
        memory = text[mem_start + 8:mem_end].strip()

    # Extract action
    act_start = lower.find("<action>")
    act_end = lower.find("</action>")
    if act_start == -1 or act_end == -1:
        return None

    content = lower[act_start + 8:act_end].strip()
    for key, val in XML_ACTION_MAP.items():
        if key in content:
            # Check <think> exists
            valid = 1 if ("<think>" in lower and "</think>" in lower) else 0
            return val, memory, valid

    return None


def arcagi3_projection(actions: List[str]) -> Tuple[list, list, list]:
    """Parse LLM outputs into ARC-AGI-3 actions + memories.

    Tries tool call JSON first, falls back to XML tags.

    Returns:
        actions: list of int (0=invalid, 1-7=ACTION1-7)
        valids: list of int (0=invalid format, 1=valid)
        memories: list of str (extracted memory text)
    """
    valids = [0] * len(actions)
    memories = [""] * len(actions)

    for i in range(len(actions)):
        original = actions[i]

        # Try tool call format first
        result = _try_parse_tool_call(original)
        if result is not None:
            actions[i], memories[i], valids[i] = result
            if i == 0:
                print(f"[PROJ] valid tool_call -> action={actions[i]}, raw={str(original)[:200]}")
            continue

        # Fall back to XML format
        result = _try_parse_xml(original)
        if result is not None:
            actions[i], memories[i], valids[i] = result
            if i == 0:
                print(f"[PROJ] valid xml -> action={actions[i]}, raw={str(original)[:200]}")
            continue

        # Nothing parsed
        actions[i] = 0
        if i == 0:
            print(f"[PROJ] INVALID -> raw={str(original)[:300]}")

    return actions, valids, memories
