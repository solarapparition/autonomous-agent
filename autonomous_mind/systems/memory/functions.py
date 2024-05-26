"""Memory system functions."""

from autonomous_mind import config
from autonomous_mind.schema import ItemId, Note


def save_note(content: str, context: str, summary: str, goal_id: str | None = None, load_to_memory: bool = True):
    """
    Save a new NOTE with the given `content`.
    `context` adds context that might not be obvious from just the `content`.
    `summary` should be no more than a sentence.
    `goal_id` is the id of the goal that this note is related to. If None, the note is just a general note.
    `load_to_memory` determines whether the note should be immediately loaded into the MEMORY section or not.
    """

    note = Note(
        content=content,
        context=context,
        summary=summary,
        goal_id=ItemId(goal_id) if goal_id else None,
        batch_number=config.action_batch_number(),
    )


    # > add a note object
    # > if load_to_memory is True, add global state value for note being loaded
    breakpoint()


def save_item_as_memory_node(item_id: str, context: str, summary: str, load_to_memory: bool = True):
    '''Save any item (except a MEMORY_NODE) that has an id as a MEMORY_NODE, copying its contents into the node. Can be used to view the contents of collapsed items in the FEED.'''
    raise NotImplementedError
