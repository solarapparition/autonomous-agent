"""Memory system functions."""

def save_note_memory_node(content: str, context: str, summary: str, load_to_memory: bool = True):
    '''
    Save a new NOTE MEMORY_NODE with the given `content`.
    `context` adds context that might not be obvious from just the `content`.
    `summary` should be no more than a sentence.
    `load_to_memory` determines whether the node should be immediately loaded into the MEMORY section or not.
    '''


    # > add a note object
    # > if load_to_memory is True, add global state value for note being loaded
    breakpoint()


def save_item_as_memory_node(item_id: str, context: str, summary: str, load_to_memory: bool = True):
    '''Save any item (except a MEMORY_NODE) that has an id as a MEMORY_NODE, copying its contents into the node. Can be used to view the contents of collapsed items in the FEED.'''
    raise NotImplementedError
