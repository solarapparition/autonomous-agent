"""Global state management."""

from autonomous_mind.systems.config.settings import GLOBAL_STATE, GLOBAL_STATE_FILE
from autonomous_mind.helpers import as_yaml_str


def set_global_state(key: str, value: str) -> None:
    """Set a global state variable."""
    GLOBAL_STATE[key] = value
    GLOBAL_STATE_FILE.write_text(as_yaml_str(GLOBAL_STATE), encoding="utf-8")
