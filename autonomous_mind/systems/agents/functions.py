"""Functions for interacting with AGENTS."""

from autonomous_mind import config
from autonomous_mind.helpers import load_yaml
from autonomous_mind.systems.agents.helpers import (
    MessageSendingError,
    get_record_file,
    load_agent_module,
    record_message_from_self,
)


async def message_agent(agent_id: str, message: str):
    """Send a message to an AGENT with the given `agent_id`."""
    agent_module = load_agent_module(agent_id)
    send_message = agent_module.send_message
    try:
        send_message(message)
    except MessageSendingError as e:
        return f"Failed to send message to agent {agent_id}. Error: {str(e)}"
    record_message_from_self(agent_id, message)
    return f"SYSTEM: Message sent to agent {agent_id}. You will receive a notification in your FEED if the agent responds."


def list_agents():
    """List all known AGENTS with their ids, names, and short summaries."""
    raise NotImplementedError


async def open_conversation(agent_id: str):
    """Switch the OPENED_AGENT_CONVERSATION to the AGENT with the given `agent_id`. The currently open conversation will be closed."""
    config.set_global_state("opened_agent_conversation", agent_id)
    # record_file = get_record_file(agent_id)
    # messages = load_yaml(record_file)
    # if len(messages) <= 5:
    #     return (
    #         "\n\n".join(
    #             [
    #                 f"[{message['timestamp']}] {message['sender']}: {message['content']}"
    #                 for message in messages
    #             ]
    #         )
    #         + "\n\n (Page 1 of 1)"
    #     )
    # raise NotImplementedError("TODO: Implement handling for more than 5 messages.")
