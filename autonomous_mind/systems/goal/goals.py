from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from uuid import UUID

from autonomous_mind import config
from autonomous_mind.helpers import as_yaml_str
from autonomous_mind.printout import full_itemized_repr, short_itemized_repr
from autonomous_mind.systems.goal.helpers import read_goal, reorder_goals


@dataclass
class Goals:
    """Goals for the Mind."""

    goals_directory: Path

    @cached_property
    def goals_files(self) -> list[Path]:
        """Get the timestamps of all goals."""
        return sorted(list(self.goals_directory.iterdir()))

    @property
    def focused(self) -> UUID | None:
        """Get the focused goal."""
        focused_id = config.GLOBAL_STATE.get("focused_goal_id")
        return UUID(focused_id) if focused_id else None

    def format(self) -> str:
        """Get a printable representation of the goals."""
        if not self.goals_files:
            return "None"
        if len(self.goals_files) == 1:
            goal = read_goal(self.goals_files[0])
            if goal.id == self.focused:
                return (
                    f"{as_yaml_str([goal.serialize()])}"
                )
        goals = [read_goal(goal_file) for goal_file in self.goals_files]
        ordered_goals, orphaned_goals = reorder_goals(goals)
        if orphaned_goals:
            raise NotImplementedError("TODO: Implement handling for orphaned goals.")
        if len(goals) == 3:
            breakpoint()  # examine ordered_goals to see if it's correct
        if len(ordered_goals) > 10:
            raise NotImplementedError("TODO: Implement handling for more than 10 goals.")
        return "\n".join(
            [
                (
                    full_itemized_repr(goal)
                    if goal.id == self.focused
                    else short_itemized_repr(goal)
                )
                for goal in ordered_goals
            ]
        )
