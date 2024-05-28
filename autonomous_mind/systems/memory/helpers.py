"""
Helper functions for the memory system.
"""

from typing import Sequence
from autonomous_mind.systems.config import settings

from autonomous_mind.schema import ItemId, Note
from autonomous_mind.systems.config import global_state
from autonomous_mind.systems.helpers import save_items


def save_notes(notes: Sequence[Note]) -> None:
    """Save notes to the NOTES directory."""
    save_items(notes, directory=settings.NOTES_DIRECTORY)

def load_note_to_memory(note_id: ItemId) -> None:
    """Load a note to memory."""
    loaded_notes = settings.loaded_notes()
    global_state.set_global_state("loaded_notes", [note_id, *loaded_notes])
