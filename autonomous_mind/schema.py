"""Schema definitions for system data structures."""

from dataclasses import dataclass
from uuid import UUID

from autonomous_mind.helpers import Timestamp


@dataclass
class Goal:
    """A goal for the Mind."""
    id: UUID
    timestamp: Timestamp
    summary: str
    details: str | None
    parent_goal_id: UUID | None
    is_focused: bool
