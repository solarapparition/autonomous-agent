"""Functions for interacting with AGENTS."""

from autonomous_mind.systems.agents.helpers import (
    MessageSendingError,
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
