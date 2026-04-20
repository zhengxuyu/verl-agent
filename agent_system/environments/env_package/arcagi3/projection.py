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
ARCAGI3_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "action_up",
            "description": "Move up",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_down",
            "description": "Move down",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_left",
            "description": "Move left",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_right",
            "description": "Move right",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_enter",
            "description": "Enter, select, or interact",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_click",
            "description": "Click at a position on the 64x64 grid",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                    "x": {"type": "integer", "description": "X coordinate (0-63)"},
                    "y": {"type": "integer", "description": "Y coordinate (0-63)"},
                },
                "required": ["reasoning", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "action_undo",
            "description": "Undo last action",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Your reasoning about the current game state"},
                    "memory_update": {"type": "string", "description": "What you have learned so far. This will be shown to you in the next step."},
                },
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


def _try_parse_tool_call(text: str):
    """Try to parse as JSON tool call. Returns (action_int, memory, valid) or None."""
    text = text.strip()

    # Try Gemma 4 format first: call:function_name{key:value,...}
    gemma_result = _try_parse_gemma4_call(text)
    if gemma_result is not None:
        return gemma_result

    # Try to find JSON object in the text
    # Qwen may output: <tool_call>{"name":...}</tool_call> or just {"name":...}
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

    # Find first { and last }
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
            continue

        # Fall back to XML format
        result = _try_parse_xml(original)
        if result is not None:
            actions[i], memories[i], valids[i] = result
            continue

        # Nothing parsed
        actions[i] = 0

    return actions, valids, memories
