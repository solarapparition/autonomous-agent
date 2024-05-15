"""Helpers for goal system."""

from typing import Literal
from autonomous_mind.schema import Goal


def save_goal(goal: Goal) -> Literal[True]:
    """Save a goal object to the goals file."""

    # factor out feed functionality to feed system
    # factor out common items in save_event
    breakpoint()
