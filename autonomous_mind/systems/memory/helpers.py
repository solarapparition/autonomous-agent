"""
Helper functions for the memory system.
"""

from typing import Sequence
from autonomous_mind.systems.config import settings

from autonomous_mind.schema import ItemId, Note
from autonomous_mind.systems.config.global_state import global_state
from autonomous_mind.systems.helpers import save_items


def save_notes(notes: Sequence[Note]) -> None:
    """Save notes to the NOTES directory."""
    save_items(notes, directory=settings.NOTES_DIRECTORY)


def load_note_to_memory(note_id: ItemId) -> None:
    """Load a note to memory."""
    global_state.loaded_memories = [note_id, *global_state.loaded_memories]
