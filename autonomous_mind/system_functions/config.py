"""Callable functions for config system."""

from typing import Literal

from autonomous_mind import config
from autonomous_mind.helpers import save_yaml


async def update_self_description(
    mode: Literal["replace", "append", "prepend"], new_description: str
):
    """Update {agent_name}'s self-description. By default replaces the current description; set `mode` parameter to change how the new description is added."""
    if mode in ["append", "prepend"]:
        raise NotImplementedError("TODO: Implement append and prepend modes.")
    config.CONFIG_DATA["self_description"] = config.SELF_DESCRIPTION = new_description
    save_yaml(config.CONFIG_DATA, config.CONFIG_FILE)
    return "Self-description successfully updated."
