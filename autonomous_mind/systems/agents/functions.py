"""Functions for interacting with AGENTS."""

from pathlib import Path
import importlib.util
from textwrap import indent

from autonomous_mind import config
from autonomous_mind.config import AGENTS_DIRECTORY


class MessageSendingError(Exception):
    """Raised when a message failed to be sent to an AGENT."""


def record_message(agent_id: str, message: str) -> None:
    """Record a message in the message log for the agent."""
    record_file = Path(f"{AGENTS_DIRECTORY}/{agent_id}/messages.yaml")
    message = indent(message, "    ")
    with record_file.open("a", encoding="utf-8") as f:
        f.write(f"- sender: {config.NAME}\n  content: |-\n{message}\n")


async def message_agent(agent_id: str, message: str):
    """Send a message to an AGENT with the given `id`."""
    agent_package_path = Path(f"{AGENTS_DIRECTORY}/{agent_id}/__init__.py")
    spec = importlib.util.spec_from_file_location(agent_id, agent_package_path)
    assert spec
    assert spec.loader
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    send_message = agent_module.send_message
    try:
        send_message(message)
    except MessageSendingError as e:
        return f"Failed to send message to agent {agent_id}. Error: {str(e)}"
    record_message(agent_id, message)
    return f"SYSTEM: Message sent to agent {agent_id}. You will receive a notification in your FEED if the agent responds."


def list_agents():
    """List all known AGENTS with their ids, names, and short summaries."""
    raise NotImplementedError
