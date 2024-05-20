"""Event classes for the feed system."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal, Sequence

from autonomous_mind import config
from autonomous_mind.helpers import count_tokens, load_yaml
from autonomous_mind.printout import full_itemized_repr, short_itemized_repr
from autonomous_mind.schema import CallResultEvent, Event, FunctionCallEvent, ItemId
from autonomous_mind.systems.goal.helpers import read_goal
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

    def format(self, focused_goal: ItemId | None, parent_goal_id: ItemId | None) -> str:
        """Get a printable representation of the feed."""
        # max_semi_recent_tokens = 1000
        recent_events_text = ""
        current_action_text = ""
        action_number = 1
        for file in reversed(self.event_files):
            event = read_event(file)
            # we represent the event differently depending on various conditions
            goal_unrelated = (
                event.goal_id != focused_goal
                or (parent_goal_id and event.goal_id != parent_goal_id)
                or (not parent_goal_id and not event.goal_id)
            )
            if focused_goal and action_number > 3 and goal_unrelated:
                # hide older events not related to the focused goal or its parent
                continue
            if action_number == 1 or (not focused_goal and action_number <= 3):
                # always show last events in full; also show recent events in full if not focused on specific goal
                event_repr = full_itemized_repr(event)
            else:
                # otherwise show abbreviated version of event
                event_repr = short_itemized_repr(event)
            current_action_text = "\n".join([event_repr, current_action_text])
            if not isinstance(event, FunctionCallEvent):
                continue
            action_number += 1
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
