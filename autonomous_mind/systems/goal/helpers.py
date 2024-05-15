"""Helpers for goal system."""

from typing import Literal, Sequence
from autonomous_mind import config
from autonomous_mind.schema import Goal
from autonomous_mind.systems.helpers import save_items


def save_goals(goals: Sequence[Goal]) -> Literal[True]:
    """Save a goal object to the goals file."""
    return save_items(goals, config.GOALS_DIRECTORY)
