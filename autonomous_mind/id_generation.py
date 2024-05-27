"""Functions for generating unique identifiers."""

from autonomous_mind import config


def generate_id() -> int:
    """Generate a unique identifier."""
    last_id = config.last_id()
    generated_id = last_id + 1
    config.update_last_id(generated_id)
    return generated_id
