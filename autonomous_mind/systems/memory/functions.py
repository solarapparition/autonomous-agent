"""Memory system functions."""

from typing import Literal
from autonomous_mind.systems.config import settings
from autonomous_mind.id_generation import generate_id
from autonomous_mind.schema import ItemId, Note
from autonomous_mind.systems.memory.helpers import load_note_to_memory, save_notes


def create_note(content: str, context: str, summary: str, goal_id: int | None = None, load_to_memory: bool = True):
    """
    Create a new NOTE with the given `content`.
    `context` adds context that might not be obvious from just the `content`.
    `summary` should be no more than a sentence.
    `goal_id` is the id of the goal that this note is related to. If None, the note is just a general note.
    `load_to_memory` determines whether the note should be immediately loaded into the MEMORY section or not.
    """
    note = Note(
        id=generate_id(),
        content=content,
        context=context,
        summary=summary,
        goal_id=goal_id,
        batch_number=settings.action_batch_number(),
    )
    save_notes([note])
    confirmation = f"MEMORY_SYSTEM: Note {note.id} created."
    if load_to_memory:
        load_note_to_memory(note.id)
        confirmation += f"MEMORY_SYSTEM: Note {note.id} loaded to active memory."
    return confirmation



async def search_memories(by: Literal["id", "keywords", "semantic_embedding"], query: str):
    """
    Search for an item (GOAL, EVENT, AGENT, etc.) by various means. Can be used to find items that are hidden, or view the contents of collapsed items.
    `query`'s meaning will change depending on the `by` parameter.
    """
    return "MEMORY_SYSTEM: memory search is not yet implemented."


def save_item_as_memory_node(item_id: str, context: str, summary: str, load_to_memory: bool = True):
    '''Save any item (except a MEMORY_NODE) that has an id as a MEMORY_NODE, copying its contents into the node. Can be used to view the contents of collapsed items in the FEED.'''
    raise NotImplementedError
