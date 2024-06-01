"""Environment system functions."""

from typing import Literal

from autonomous_mind.systems.environment import shell


def sleep(mode: Literal["until", "hour", "minute", "second"], time: str | int):
    """Put {mind_name} to sleep until a specific UTC timestamp or for a specific duration."""
    if mode == "until":
        return f"Sleeping until {time}."
    return f"ENVIRONMENT_SYSTEM: Sleeping for {time} {mode}(s)."


def send_shell_command(command: str) -> str:
    """Send a shell command to the environment."""
    shell.send_command(command)
    abbreviated_command = f"{command[:25]} [...]"
    return f"Sent shell command:\n```\n{abbreviated_command}\n```"
