"""Schema definitions for system data structures."""

from dataclasses import dataclass, field
from textwrap import indent
from typing import Any, Literal, Mapping, MutableMapping, Self

from autonomous_mind.helpers import (
    Timestamp,
    format_timestamp,
    from_yaml_str,
    get_timestamp,
)
from autonomous_mind.text import dedent_and_strip

ItemId = int | str


@dataclass
class FunctionCallEvent:
    """An action event."""

    goal_id: ItemId | None
    "Id of the goal this action is related to."
    batch_number: int
    summary: str
    content: str
    id: ItemId
    type: Literal["function_call"] = "function_call"
    timestamp: Timestamp = field(default_factory=get_timestamp)
    success: Literal[-1, 0, 1] = 0

    @classmethod
    def from_mapping(cls, mapping: MutableMapping[str, Any]) -> Self:
        """Create event from a mapping."""
        mapping["goal_id"] = mapping["goal_id"] or None
        mapping["timestamp"] = format_timestamp(mapping["timestamp"])
        return cls(**mapping)

    def __repr__(self) -> str:
        """Get the string representation of the event."""
        template = """
        id: {id}
        type: function_call
        batch_number: {batch_number}
        goal_id: {goal_id}
        timestamp: {timestamp}
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
            batch_number=self.batch_number,
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
        batch_number: {batch_number}
        goal_id: {goal_id}
        timestamp: {timestamp}
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
            batch_number=self.batch_number,
            summary=summary,
            success=self.success,
        )


@dataclass
class CallResultEvent:
    """Event for result of calls."""

    goal_id: ItemId | None
    "Id of the goal this call result is related to."
    function_call_id: ItemId
    "Id of the function call this result is for."
    batch_number: int
    content: str
    id: ItemId
    type: Literal["call_result"] = "call_result"
    timestamp: Timestamp = field(default_factory=get_timestamp)
    summary: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create an event from a mapping."""
        mapping = dict(mapping)
        mapping["goal_id"] = mapping["goal_id"] or None
        mapping["timestamp"] = format_timestamp(mapping["timestamp"])
        return cls(**mapping)

    def __repr__(self) -> str:
        """Get the string representation of the event."""
        template = """
        id: {id}
        type: call_result
        batch_number: {batch_number}
        function_call_id: {function_call_id}
        goal_id: {goal_id}
        timestamp: {timestamp}
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
            batch_number=self.batch_number,
            content=content,
            summary=summary,
        )

    def __str__(self) -> str:
        """Printout of event."""
        template = """
        id: {id}
        type: call_result
        batch_number: {batch_number}
        function_call_id: {function_call_id}
        goal_id: {goal_id}
        timestamp: {timestamp}
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
            batch_number=self.batch_number,
            function_call_id=self.function_call_id,
            summary=summary,
        )


@dataclass
class NotificationEvent:
    """Notification event."""

    content: str
    batch_number: int
    id: ItemId
    type: Literal["notification"] = "notification"
    timestamp: Timestamp = field(default_factory=get_timestamp)
    summary: str = ""

    @property
    def goal_id(self) -> None:
        """Get the goal id."""
        return None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create an event from a mapping."""
        mapping = dict(mapping)
        mapping["timestamp"] = format_timestamp(mapping["timestamp"])
        return cls(**mapping)

    def __repr__(self) -> str:
        """Get the string representation of the event."""
        template = """
        id: {id}
        type: notification
        batch_number: {batch_number}
        timestamp: {timestamp}
        summary: |-
        {summary}
        content: |-
        {content}
        """
        summary = indent(self.summary, "  ")
        content = indent(self.content, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            timestamp=self.timestamp,
            batch_number=self.batch_number,
            summary=summary,
            content=content,
        )

    def __str__(self) -> str:
        """Printout of event."""
        return repr(self)


Event = FunctionCallEvent | CallResultEvent | NotificationEvent


@dataclass
class Goal:
    """A goal for the Mind."""

    summary: str
    details: str | None
    batch_number: int
    parent_goal_id: ItemId | None
    id: ItemId
    timestamp: Timestamp = field(default_factory=get_timestamp)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create a goal from a mapping."""
        mapping = dict(mapping)
        mapping["timestamp"] = format_timestamp(mapping["timestamp"])
        mapping["parent_goal_id"] = mapping["parent_goal_id"] or None
        return cls(**mapping)

    def __repr__(self) -> str:
        """Get the string representation of the goal."""
        template = """
        id: {id}
        batch_number: {batch_number}
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
            batch_number=self.batch_number,
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
        batch_number: {batch_number}
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
            batch_number=self.batch_number,
            summary=summary,
            parent_goal_id=self.parent_goal_id or "!!null",
        )


@dataclass
class Note:
    """A note."""

    content: str
    context: str
    batch_number: int
    id: ItemId
    summary: str = ""
    goal_id: ItemId | None = None
    timestamp: Timestamp = field(default_factory=get_timestamp)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create a memory node from a mapping."""
        mapping = dict(mapping)
        mapping["timestamp"] = format_timestamp(mapping["timestamp"])
        return cls(**mapping)

    def __repr__(self) -> str:
        """Get the string representation of the memory node."""
        template = """
        id: {id}
        batch_number: {batch_number}
        goal_id: {goal_id}
        timestamp: {timestamp}
        summary: |-
        {summary}
        context: |-
        {context}
        content: |-
        {content}
        """
        summary = indent(self.summary, "  ")
        context = indent(self.context, "  ")
        content = indent(self.content, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            timestamp=self.timestamp,
            batch_number=self.batch_number,
            summary=summary,
            context=context,
            content=content,
            goal_id=self.goal_id or "!!null",
        )

    def __str__(self) -> str:
        """Printout of memory node."""
        template = """
        id: {id}
        batch_number: {batch_number}
        goal_id: {goal_id}
        timestamp: {timestamp}
        summary: |-
        {summary}
        context: |-
        {context}
        content: |-
          [Collapsed]
        """
        summary = indent(self.summary, "  ")
        context = indent(self.context, "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            timestamp=self.timestamp,
            batch_number=self.batch_number,
            summary=summary,
            context=context,
            goal_id=self.goal_id or "!!null",
        )


Item = Goal | Event | Note
