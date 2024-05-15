"""Event classes for the feed system."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal, Sequence
from uuid import UUID

from autonomous_mind import config
from autonomous_mind.helpers import (
    as_yaml_str,
    count_tokens,
    from_yaml_str,
    load_yaml,
)
from autonomous_mind.schema import CallResultEvent, Event, FunctionCallEvent
from autonomous_mind.systems.helpers import save_items


def save_events(events: Sequence[Event]) -> Literal[True]:
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
                breakpoint()
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
