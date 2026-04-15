"""Prompt templates for ARC-AGI-3 games."""

ARCAGI3_VISUAL_TEMPLATE = """
You are playing a puzzle game on a 64x64 pixel grid. You must figure out the rules by trying actions and observing what happens. Your goal is to clear all levels.

# Actions
- ACTION1: Up
- ACTION2: Down
- ACTION3: Left
- ACTION4: Right
- ACTION5: Enter/Select
- ACTION6: Click (needs coordinates)
- ACTION7: Undo

# Current Step
Your current observation is shown in the image: <image>
Your admissible actions are ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about what you see in the image — identify the player, obstacles, and possible goals. Think about what each color might represent and what action would make progress. This reasoning MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose an action and present it within <action> </action> tags.
"""

ARCAGI3_VISUAL_TEMPLATE_WITH_HISTORY = """
You are playing a puzzle game on a 64x64 pixel grid. You must figure out the rules by trying actions and observing what happens. Your goal is to clear all levels.

# Actions
- ACTION1: Up
- ACTION2: Down
- ACTION3: Left
- ACTION4: Right
- ACTION5: Enter/Select
- ACTION6: Click (needs coordinates)
- ACTION7: Undo

# Current Step
Prior to this step, you have taken {step_count} step(s). Below are your most recent {history_length} actions and what happened: {action_history}
You are now at step {current_step} and your current observation is shown in the image: <image>
Your admissible actions are ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about what you see — what changed after your last action? What pattern do you notice? What should you try next? This reasoning MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose an action and present it within <action> </action> tags.
"""
