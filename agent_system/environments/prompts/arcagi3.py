"""Prompt templates for ARC-AGI-3 games with memory support."""

ARCAGI3_VISUAL_TEMPLATE = """You are playing a puzzle game on a 64x64 pixel grid. You must figure out the rules by trying actions and observing what changes. Clear all levels to win.

# Actions
- ACTION1: Up
- ACTION2: Down
- ACTION3: Left
- ACTION4: Right
- ACTION5: Enter/Select
- ACTION7: Undo

# Your Memory
(empty - this is your first step)

# Current Observation
The current game state is shown in the image: <image>

# Instructions
1. First, reason about what you see in <think></think> tags.
2. Then, write what you have learned so far in <memory></memory> tags. This will be shown to you in future steps. Record:
   - What each color likely represents (player, wall, goal, etc.)
   - What actions do (which direction they move things)
   - Any rules or patterns you discovered
   - What you should try next and why
3. Finally, choose ONE action in <action></action> tags.
"""

ARCAGI3_VISUAL_TEMPLATE_WITH_HISTORY = """You are playing a puzzle game on a 64x64 pixel grid. You must figure out the rules by trying actions and observing what changes. Clear all levels to win.

# Actions
- ACTION1: Up
- ACTION2: Down
- ACTION3: Left
- ACTION4: Right
- ACTION5: Enter/Select
- ACTION7: Undo

# Your Memory (from previous steps)
{memory}

# History (last {history_length} actions)
Step {current_step}/{step_count} total. Recent actions: {action_history}

# Current Observation
The current game state is shown in the image: <image>

# Instructions
1. First, reason about what you see and what changed after your last action in <think></think> tags.
2. Then, UPDATE your memory in <memory></memory> tags. Keep useful info, discard wrong guesses. Record:
   - What each color likely represents
   - What actions do (movement directions, effects)
   - Any rules or patterns discovered
   - Current goal and plan
3. Finally, choose ONE action in <action></action> tags.
"""
