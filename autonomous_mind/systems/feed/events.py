"""Event classes for the feed system."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from autonomous_mind.systems.config import settings
from autonomous_mind.systems.config.global_state import global_state
from autonomous_mind.helpers import count_tokens
from autonomous_mind.printout import full_itemized_repr, short_itemized_repr
from autonomous_mind.schema import (
    Event,
    ItemId,
    FunctionCallEvent,
)
from autonomous_mind.systems.helpers import read_event


@dataclass
class Feed:
    """Feed of events and actions."""

    events_directory: Path

    @cached_property
    def event_files(self) -> list[Path]:
        """Get the timestamps of all events."""
        return sorted(list(self.events_directory.iterdir()))

    def call_event_batch(self, action_batch_limit: int = 1) -> list[Event]:
        """New events since a certain number of actions ago."""
        events: list[Event] = []
        # action_count = 0
        for event_file in reversed(self.event_files):
            event = read_event(event_file)
            if global_state.action_batch_number - event.batch_number > action_batch_limit:
                break
            events.insert(0, event)
            # if isinstance(event, FunctionCallEvent):
            #     action_count += 1
            # if action_count == action_batch_limit:
            #     break
        return events

    @cached_property
    def recent_events(self) -> list[Event]:
        """Get all recent events."""
        return self.call_event_batch(3)

    def format(
        self, focused_goal: ItemId | None, parent_goal_id: ItemId | None
    ) -> str | None:
        """Get a printable representation of the feed."""
        # max_semi_recent_tokens = 1000
        recent_events_text = ""
        current_action_text = ""
        for file in reversed(self.event_files):
            event = read_event(file)
            # we represent the event differently depending on various conditions
            batch_recency = global_state.action_batch_number - event.batch_number
            goal_unrelated = (
                event.goal_id != focused_goal
                or (parent_goal_id and event.goal_id != parent_goal_id)
                or (not parent_goal_id and not event.goal_id)
            )
            if focused_goal and batch_recency > 3 and goal_unrelated:
                # hide older events not related to the focused goal or its parent
                continue
            if (not focused_goal and batch_recency <= 3) or (
                event.goal_id == focused_goal and batch_recency == 1
            ):
                # always show last event related to current goal, or not focused on specific goal
                event_repr = full_itemized_repr(event)
            else:
                # otherwise show abbreviated version of event
                event_repr = short_itemized_repr(event)
            current_action_text = "\n".join([event_repr, current_action_text])
            if not isinstance(event, FunctionCallEvent):
                continue
            batch_recency += 1
            proposed_recent_events_text = "\n".join(
                [current_action_text, recent_events_text]
            )
            if (
                count_tokens(proposed_recent_events_text)
                > settings.MAX_RECENT_FEED_TOKENS
            ):
                breakpoint()
                raise NotImplementedError("TODO: Rewind back to `recent_events_text`.")
            recent_events_text = proposed_recent_events_text
            current_action_text = ""
        return recent_events_text.strip() or None
