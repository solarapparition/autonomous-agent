"""Knowledge system functions."""

def save_knowledge_node(content: str, context: str, summary: str, load_to_knowledge: bool = True):
    """
    Save a new KNOWLEDGE_NODE with the given `content`.
    `context` adds context that might not be obvious from just the `content`.
    `summary` should be no more than a sentence.
    `load_to_knowledge` determines whether the node should be immediately loaded into the KNOWLEDGE or not.
    """
    raise NotImplementedError
