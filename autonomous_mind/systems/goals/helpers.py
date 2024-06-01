"""Helpers for goal system."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence
from autonomous_mind.systems.config import settings
from autonomous_mind.helpers import load_yaml
from autonomous_mind.schema import Goal, ItemId
from autonomous_mind.systems.helpers import read_goal, save_items


def save_goals(goals: Sequence[Goal]) -> Literal[True]:
    """Save a goal object to the goals file."""
    return save_items(goals, settings.GOALS_DIRECTORY)


def find_goal(goal_id: ItemId) -> Goal:
    """Get the filename for a goal."""
    return next(
        goal
        for goal_file in reversed(list(settings.GOALS_DIRECTORY.iterdir()))
        if (goal := read_goal(goal_file)).id == goal_id
    )


def find_last_descendant_index(new_list: list[Goal], parent_id: ItemId) -> int:
    """Find the last descendant index of a parent goal."""
    last_index = -1
    for i, goal in enumerate(new_list):
        if goal.parent_goal_id == parent_id:
            last_index = max(last_index, i)
            child_last_index = find_last_descendant_index(new_list, goal.id)
            last_index = max(last_index, child_last_index)
        elif goal.id == parent_id and last_index < i:
            last_index = i
    return last_index


def reorder_goals(old_list: list[Goal]) -> tuple[list[Goal], list[Goal]]:
    """Reorder goals to be in parent-child order."""
    new_list: list[Goal] = []
    orphaned: list[Goal] = []

    for goal in old_list:
        if goal.parent_goal_id is None:
            new_list.append(goal)
        else:
            parent_exists = any(parent.id == goal.parent_goal_id for parent in new_list)
            if parent_exists:
                parent_index = find_last_descendant_index(new_list, goal.parent_goal_id)
                new_list.insert(parent_index + 1, goal)
            else:
                orphaned.append(goal)

    return new_list, orphaned
