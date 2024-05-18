"""Schema definitions for system data structures."""

from dataclasses import dataclass, field
from textwrap import indent
from typing import Any, Literal, Mapping, Self
from uuid import UUID, uuid4

from autonomous_mind.helpers import (
    Timestamp,
    as_yaml_str,
    format_timestamp,
    from_yaml_str,
    get_timestamp,
)
from autonomous_mind.text import dedent_and_strip


@dataclass
class FunctionCallEvent:
    """An action event."""

    goal_id: UUID | None
    "Id of the goal this action is related to."
    summary: str
    content: str
    id: UUID = field(default_factory=uuid4)
    timestamp: Timestamp = field(default_factory=get_timestamp)
    success: Literal[-1, 0, 1] = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create event from a mapping."""
        return cls(
            id=UUID(mapping["id"]),
            goal_id=UUID(mapping["goal_id"]) if mapping["goal_id"] else None,
            timestamp=format_timestamp(mapping["timestamp"]),
            summary=mapping["summary"],
            content=mapping["content"],
        )

    def __repr__(self) -> str:
        """Get the string representation of the event."""
        template = """
        id: {id}
        type: function_call
        timestamp: {timestamp}
        goal_id: {goal_id}
        summary: |-
        {summary}
        content: |-
        {content}
        success: {success}
        """
        summary = indent(self.summary, "  ")
        content = indent(self.content, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            goal_id=self.goal_id or "!!null",
            summary=summary,
            timestamp=self.timestamp,
            content=content,
            success=self.success,
        )

    def __str__(self) -> str:
        """Printout of event."""
        template = """
        id: {id}
        type: function_call
        timestamp: {timestamp}
        goal_id: {goal_id}
        summary: |-
        {summary}
        content: |-
          [Collapsed]
        success: {success}
        """
        summary = indent(self.summary, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            goal_id=self.goal_id or "!!null",
            timestamp=self.timestamp,
            summary=summary,
            success=self.success,
        )


@dataclass
class CallResultEvent:
    """Event for result of calls."""

    goal_id: UUID | None
    "Id of the goal this call result is related to."
    function_call_id: UUID
    "Id of the function call this result is for."
    content: str
    id: UUID = field(default_factory=uuid4)
    timestamp: Timestamp = field(default_factory=get_timestamp)
    summary: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create an event from a mapping."""
        return cls(
            id=UUID(mapping["id"]),
            goal_id=UUID(mapping["goal_id"]) if mapping["goal_id"] else None,
            timestamp=format_timestamp(mapping["timestamp"]),
            function_call_id=UUID(mapping["function_call_id"]),
            content=mapping["content"],
        )

    def __repr__(self) -> str:
        """Get the string representation of the event."""
        template = """
        id: {id}
        type: call_result
        timestamp: {timestamp}
        goal_id: {goal_id}
        function_call_id: {function_call_id}
        summary: |-
        {summary}
        content: |-
        {content}
        """
        content = indent(self.content, "  ")
        summary = indent(self.summary, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            goal_id=self.goal_id or "!!null",
            timestamp=self.timestamp,
            function_call_id=self.function_call_id,
            content=content,
            summary=summary,
        )

    def __str__(self) -> str:
        """Printout of event."""
        template = """
        id: {id}
        type: call_result
        timestamp: {timestamp}
        goal_id: {goal_id}
        function_call_id: {function_call_id}
        summary: |-
        {summary}
        content: |-
          [Collapsed]
        """
        summary = indent(self.summary, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            goal_id=self.goal_id or "!!null",
            timestamp=self.timestamp,
            function_call_id=self.function_call_id,
            summary=summary,
        )


Event = FunctionCallEvent | CallResultEvent


@dataclass
class Goal:
    """A goal for the Mind."""

    summary: str
    details: str | None
    parent_goal_id: UUID | None
    id: UUID = field(default_factory=uuid4)
    timestamp: Timestamp = field(default_factory=get_timestamp)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create a goal from a mapping."""
        return cls(
            id=UUID(mapping["id"]),
            timestamp=format_timestamp(mapping["timestamp"]),
            summary=mapping["summary"],
            details=mapping["details"],
            parent_goal_id=(
                UUID(mapping["parent_goal_id"]) if mapping["parent_goal_id"] else None
            ),
        )

    def __repr__(self) -> str:
        """Get the string representation of the goal."""
        template = """
        id: {id}
        parent_goal_id: {parent_goal_id}
        timestamp: {timestamp}
        summary: |-
        {summary}
        details: |-
        {details}
        """
        summary = indent(self.summary, "  ")
        details = indent(self.details or "", "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            timestamp=self.timestamp,
            summary=summary,
            details=details,
            parent_goal_id=self.parent_goal_id or "!!null",
        )

    def serialize(self) -> Mapping[str, Any]:
        """Serialize the goal to a mapping."""
        return from_yaml_str(repr(self))

    def __str__(self) -> str:
        """Printout of goal."""
        template = """
        id: {id}
        parent_goal_id: {parent_goal_id}
        timestamp: {timestamp}
        summary: |-
        {summary}
        details: |-
          [Collapsed]
        """
        summary = indent(self.summary, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            timestamp=self.timestamp,
            summary=summary,
            parent_goal_id=self.parent_goal_id or "!!null",
        )


Item = Goal | Event
