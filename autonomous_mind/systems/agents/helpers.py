from pathlib import Path
import importlib.util
from textwrap import indent
from types import ModuleType
from typing import Mapping

from autonomous_mind import config
from autonomous_mind.config import AGENTS_DIRECTORY
from autonomous_mind.helpers import get_timestamp, load_yaml
from autonomous_mind.schema import ItemId, NotificationEvent


class MessageSendingError(Exception):
    """Raised when a message failed to be sent to an AGENT."""


def post_message(record_file: Path, message: str, sender: str, unread: bool) -> None:
    """Post a message to a record file."""
    message = indent(message, "    ")
    with record_file.open("a", encoding="utf-8") as f:
        f.write(
            f"- sender: {sender}\n  timestamp: {get_timestamp()}\n  content: |-\n{message}\n"
        )
        if unread:
            f.write("  new: true\n")


def get_record_file(agent_id: str) -> Path:
    """Get the record file for an agent."""
    return Path(f"{AGENTS_DIRECTORY}/{agent_id}/messages.yaml")


def record_message_from_self(agent_id: str, message: str) -> None:
    """Record a message in the message log for the agent."""
    post_message(get_record_file(agent_id), message, config.NAME, unread=False)


def load_agent_module(agent_id: str) -> ModuleType:
    """Load the agent module from the given path."""
    agent_package_path = Path(f"{AGENTS_DIRECTORY}/{agent_id}/__init__.py")
    spec = importlib.util.spec_from_file_location(agent_id, agent_package_path)
    assert spec
    assert spec.loader
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    return agent_module


def count_new_messages(agent_id: ItemId) -> int:
    """Count the number of new messages for an agent."""
    record_file = Path(f"{AGENTS_DIRECTORY}/{agent_id}/messages.yaml")
    if not record_file.exists():
        return 0
    records = load_yaml(record_file)
    return (
        sum(
            bool(record)
            for record in records
            if record["sender"] != config.NAME and record.get("new")
        )
        if records
        else 0
    )


def download_new_messages() -> dict[ItemId, int]:
    """Download messages from all agents."""
    agent_ids = [
        ItemId(str(agent_id.name))
        for agent_id in AGENTS_DIRECTORY.iterdir()
        if agent_id.is_dir()
    ]
    message_counts: dict[ItemId, int] = {}
    for agent_id in agent_ids:
        agent_module = load_agent_module(agent_id)
        if num_new_messages := agent_module.download_new_messages():
            message_counts[agent_id] = num_new_messages
    return message_counts


def sender_name(sender_id: ItemId) -> str:
    """Get the name of the sender."""
    if sender_id == ItemId("25b9a536-54d0-4162-bae9-ec81dba993e9"):  # developer
        return config.DEVELOPER

    raise NotImplementedError(
        "TODO: Implement sender_name (i.e. contact list) for other agents."
    )


def new_messages_notification(
    new_message_counts: Mapping[ItemId, int]
) -> NotificationEvent:
    """Generate a notification for new messages."""
    senders = {
        sender_id: sender_name(sender_id) for sender_id in new_message_counts.keys()
    }
    sender_names = ", ".join(
        f"{name} ({sender_id})" for sender_id, name in senders.items()
    )
    return NotificationEvent(
        content=f"New message(s) from: {sender_names}. Open the conversation with the agent to view.",
        batch_number=config.ACTION_BATCH_NUMBER - 1,
    )
