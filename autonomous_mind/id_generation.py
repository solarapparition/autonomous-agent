"""Functions for generating unique identifiers."""

from autonomous_mind import config
from autonomous_mind.schema import ItemId


def generate_id() -> ItemId:
    """Generate a unique identifier."""
    last_id = config.last_id()
    generated_id = last_id + 1
    config.update_last_id(generated_id)
    return ItemId(generated_id)
