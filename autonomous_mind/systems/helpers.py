"""Helpers for systems."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

from autonomous_mind.helpers import load_yaml, timestamp_to_filename
from autonomous_mind.schema import (
    CallResultEvent,
    Event,
    FunctionCallEvent,
    Goal,
    Item,
    ItemId,
    Note,
    NotificationEvent,
)
from autonomous_mind.systems.config import settings


def save_items(items: Sequence[Item], directory: Path) -> Literal[True]:
    """Save an event to disk."""
    for item in items:
        event_str = repr(item)
        event_file = directory / f"{timestamp_to_filename(item.timestamp)}.yaml"
        event_file.write_text(event_str, encoding="utf-8")
    return True


def save_events(events: Sequence[Event]) -> Literal[True]:
    """Save the events to disk."""
    return save_items(events, settings.EVENTS_DIRECTORY)


@lru_cache(maxsize=None)
def read_event(event_file: Path) -> Event:
    """Read an event from disk."""
    event_dict = load_yaml(event_file)
    type_mapping = {
        "function_call": FunctionCallEvent,
        "call_result": CallResultEvent,
        "notification": NotificationEvent,
    }
    return type_mapping[event_dict["type"]].from_mapping(event_dict)


@lru_cache(maxsize=None)
def read_goal(goal_file: Path) -> Goal:
    """Read a goal from disk."""
    return Goal.from_mapping(load_yaml(goal_file))


@lru_cache(maxsize=None)
def read_note(note_file: Path) -> Note:
    """Read a note from disk."""
    return Note.from_mapping(load_yaml(note_file))


@lru_cache(maxsize=None)
def load_item_by_id(item_id: ItemId) -> Item:
    """Load item by id."""
    for path in settings.NOTES_DIRECTORY.iterdir():
        if (mapping := load_yaml(path))["id"] == item_id:
            return Note.from_mapping(mapping)

    raise NotImplementedError("TODO: implement loading of items besides notes")
