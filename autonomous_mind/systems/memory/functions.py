"""Memory system functions."""

def save_memory_node(content: str, context: str, summary: str, load_to_memory: bool = True):
    """
    Save a new MEMORY_NODE with the given `content`.
    `context` adds context that might not be obvious from just the `content`.
    `summary` should be no more than a sentence.
    `load_to_memory` determines whether the node should be immediately loaded into the MEMORY or not.
    """
    raise NotImplementedError
