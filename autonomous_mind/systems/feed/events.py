"""Event classes for the feed system."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from textwrap import indent
from typing import Any, Literal, Mapping, Self
from uuid import UUID

from autonomous_mind import config
from autonomous_mind.helpers import (
    Timestamp,
    as_yaml_str,
    count_tokens,
    format_timestamp,
    from_yaml_str,
    load_yaml,
)
from autonomous_mind.systems.helpers import save_items
from autonomous_mind.text import dedent_and_strip


@dataclass
class FunctionCallEvent:
    """An action event."""

    id: UUID
    goal_id: UUID | None
    "Id of the goal this action is related to."
    timestamp: Timestamp
    content: Mapping[str, Any]
    success: Literal[-1, 0, 1] = 0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """Create an action event from a mapping."""
        return cls(
            id=UUID(mapping["id"]),
            goal_id=UUID(mapping["goal_id"]) if mapping["goal_id"] else None,
            timestamp=format_timestamp(mapping["timestamp"]),
            content=mapping["content"],
        )

    @property
    def summary(self) -> str:
        """Get a summary of the action event."""
        return self.content["action_intention"]

    def __repr__(self) -> str:
        """Get the string representation of the event."""
        template = """
        id: {id}
        type: function_call
        timestamp: {timestamp}
        goal_id: {goal_id}
        content:
        {content}
        success: {success}
        """
        content = indent(as_yaml_str(self.content), "  ")
        return dedent_and_strip(template).format(
            id=self.id,
            goal_id=self.goal_id or "!!null",
            timestamp=self.timestamp,
            content=content,
            success=self.success,
        )

    def __str__(self) -> str:
        """Printout of action event."""
        return self.__repr__()


@dataclass
class CallResultEvent:
    """Event for result of calls."""

    id: UUID
    timestamp: Timestamp
    goal_id: UUID | None
    "Id of the goal this call result is related to."
    function_call_id: UUID
    "Id of the function call this result is for."
    content: str
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
        """Get the string representation of the action event."""
        template = """
        id: {id}
        type: call_result
        timestamp: {timestamp}
        goal_id: {goal_id}
        function_call_id: {function_call_id}
        content: |-
        {content}
        summary: |-
        {summary}
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
        """Printout of action event."""
        return self.__repr__()


Event = FunctionCallEvent | CallResultEvent

def save_events(events: list[Event]) -> Literal[True]:
    """Save the events to disk."""
    return save_items(events, config.EVENTS_DIRECTORY)


@lru_cache(maxsize=None)
def read_event(event_file: Path) -> Event:
    """Read an event from disk."""
    event_dict = load_yaml(event_file)
    type_mapping = {
        "function_call": FunctionCallEvent,
        "call_result": CallResultEvent,
    }
    return type_mapping[event_dict["type"]].from_mapping(event_dict)


@dataclass
class Feed:
    """Feed of events and actions."""

    events_directory: Path

    @cached_property
    def event_files(self) -> list[Path]:
        """Get the timestamps of all events."""
        return sorted(list(self.events_directory.iterdir()))

    def call_event_batch(self, action_number: int = 1) -> list[Event]:
        """New events since a certain number of actions ago."""
        events: list[Event] = []
        action_count = 0
        for event_file in reversed(self.event_files):
            event = read_event(event_file)
            events.insert(0, event)
            if isinstance(event, FunctionCallEvent):
                action_count += 1
            if action_count == action_number:
                break
        return events

    @cached_property
    def recent_events(self) -> list[Event]:
        """Get all recent events."""
        return self.call_event_batch(3)

    def format(self, focused_goal: UUID | None) -> str:
        """Get a printable representation of the feed."""
        # cycle back from the most recent event until we get to ~2000 tokens
        # max_semi_recent_tokens = 1000
        recent_events_text = ""
        current_action_text = ""
        for file in reversed(self.event_files):
            event = read_event(file)
            if focused_goal:
                raise NotImplementedError(
                    "TODO: Implement filtering of events."
                )  # > make sure to add contentless versions of recent async events (within last 3 actions)
            event_repr = as_yaml_str([from_yaml_str(repr(event))])
            current_action_text = "\n".join([event_repr, current_action_text])
            if not isinstance(event, FunctionCallEvent):
                continue
            proposed_recent_events_text = "\n".join(
                [current_action_text, recent_events_text]
            )
            if (
                count_tokens(proposed_recent_events_text)
                > config.MAX_RECENT_FEED_TOKENS
            ):
                raise NotImplementedError("TODO: Rewind back to `recent_events_text`.")
            recent_events_text = proposed_recent_events_text
            current_action_text = ""
        return recent_events_text.strip()
