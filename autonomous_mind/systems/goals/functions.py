"""System functions for goal management."""

# pylint: disable=line-too-long

from autonomous_mind import config
from autonomous_mind.helpers import get_timestamp
from autonomous_mind.id_generation import generate_id
from autonomous_mind.schema import Goal
from autonomous_mind.systems.config.global_state import set_global_state
from autonomous_mind.systems.goals.helpers import find_goal, save_goals
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
        id=generate_id(),
        batch_number=config.action_batch_number(),
        timestamp=get_timestamp(),
        summary=summary,
        details=details,
        parent_goal_id=parent_goal_id,
    )
    save_goals([goal])
    set_global_state("focused_goal_id", str(goal.id))
    confirmation = f"""
    GOAL_SYSTEM:
    - GOAL added:
        id: {goal.id}
        summary: {goal.summary}
    - Switched focus to GOAL {goal.id} from {parent_goal_id}
    """
    return dedent_and_strip(confirmation)


async def edit_goal(goal_id: int, new_summary: str | None, new_details: str | None, new_parent_goal_id: int | None):
    """Edit a goal with the given `goal_id`. Any parameter set to None will not be changed."""

    goal = find_goal(goal_id)
    goal.summary = new_summary or goal.summary
    goal.details = new_details or goal.details
    goal.parent_goal_id = new_parent_goal_id or goal.parent_goal_id
    save_goals([goal])
    return f"GOAL_SYSTEM: Goal {goal.id} updated."
