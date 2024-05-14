"""System functions for goal management."""

from dataclasses import dataclass
from uuid import UUID


def add_goal(summary: str, details: str | None, parent_goal_id: str | None = None, switch_focus: bool = True):
    """
    Add a new goal for {mind_name}.
    `summary` should be no more than a sentence.
    `details` should only be provided if the goal requires more explanation than can be given in the `summary`.
    If parent_goal_id is provided, the goal will be a subgoal of the parent goal with that id; otherwise, it will be a root-level goal.
    If `switch_focus` is True, then the new goal will automatically become the FOCUSED goal.
    """
    if parent_goal_id:
        raise NotImplementedError("TODO: Implement subgoals.")
    
    





    # commit
    breakpoint()
    @dataclass
    class Goal:
        """A goal for the Mind."""
        id: UUID
        timestamp: str
        summary: str
        details: str | None
        parent_goal_id: UUID | None
        is_focused: bool
    # create a new goal object
    # save goal object to file
    # return confirmation message
    # > add request_system_upgrade