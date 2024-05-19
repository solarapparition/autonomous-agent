"""System functions for goal management."""

# pylint: disable=line-too-long

from autonomous_mind.helpers import get_timestamp
from autonomous_mind.schema import Goal, ItemId
from autonomous_mind.systems.config.global_state import set_global_state
from autonomous_mind.systems.goal.helpers import save_goals
from autonomous_mind.text import dedent_and_strip


def add_goal(summary: str, details: str | None, parent_goal_id: str | None = None, switch_focus: bool = True):
    """
    Add a new goal for {mind_name}.
    `summary` should be no more than a sentence.
    `details` should only be provided if the goal requires more explanation than can be given in the `summary`.
    If parent_goal_id is provided, the goal will be a subgoal of the parent goal with that id; otherwise, it will be a root-level goal.
    If `switch_focus` is True, then the new goal will automatically become the FOCUSED goal.
    """
    goal = Goal(
        timestamp=get_timestamp(),
        summary=summary,
        details=details,
        parent_goal_id=ItemId(parent_goal_id) if parent_goal_id else None,
    )
    save_goals([goal])
    set_global_state("focused_goal_id", str(goal.id))
    confirmation = f"""
    - GOAL added:
        id: {goal.id}
        summary: {goal.summary}
    - Switched focus to GOAL {goal.id} from {parent_goal_id}
    """
    return dedent_and_strip(confirmation)
