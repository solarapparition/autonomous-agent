"""Functions for generating unique identifiers."""

from autonomous_mind.systems.config.global_state import global_state


def generate_id() -> int:
    """Generate a unique identifier."""
    last_id = global_state.last_id
    generated_id = last_id + 1
    global_state.last_id = generated_id
    return generated_id
