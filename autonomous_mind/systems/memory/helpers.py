"""
Helper functions for the memory system.
"""

from typing import Sequence
from autonomous_mind import config

from autonomous_mind.schema import ItemId, Note
from autonomous_mind.systems.helpers import save_items


def save_notes(notes: Sequence[Note]) -> None:
    """Save notes to the NOTES directory."""
    save_items(notes, directory=config.NOTES_DIRECTORY)

def load_note_to_memory(note_id: ItemId) -> None:
    """Load a note to memory."""
    loaded_notes = config.loaded_notes()
    config.set_global_state("loaded_notes", [note_id, *loaded_notes])
