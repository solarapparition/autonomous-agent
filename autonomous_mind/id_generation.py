"""Functions for generating unique identifiers."""

from autonomous_mind.systems.config import settings


def generate_id() -> int:
    """Generate a unique identifier."""
    last_id = settings.last_id()
    generated_id = last_id + 1
    settings.update_last_id(generated_id)
    return generated_id
