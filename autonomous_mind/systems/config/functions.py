"""Callable functions for config system."""

from typing import Literal

from autonomous_mind.systems.config import settings
from autonomous_mind.helpers import save_yaml


def update_self_description(mode: Literal["replace", "append", "prepend"], new_description: str):
    """Update {mind_name}'s self-description. By default replaces the current description; set `mode` parameter to change how the new description is added."""
    if mode in ["append", "prepend"]:
        raise NotImplementedError("TODO: Implement append and prepend modes.")
    settings.CONFIG_DATA["self_description"] = settings.SELF_DESCRIPTION = new_description
    save_yaml(settings.CONFIG_DATA, settings.CONFIG_FILE)
    return "Self-description successfully updated."
