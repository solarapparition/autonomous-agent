"""Functions for interacting with AGENTS."""

from autonomous_mind import config
from autonomous_mind.systems.agents.helpers import (
    MessageSendingError,
    load_agent_module,
    record_message_from_self,
)


async def message_agent(message: str):
    """Send a message to the agent that is currently open in the OPENED_AGENT_CONVERSATION. If no conversation is open, return an error message."""
    if not (agent_id := config.opened_agent_conversation()):
        return "AGENT_SYSTEM: No conversation with an agent is currently open. Please open a conversation with an agent before sending a message."
    agent_module = load_agent_module(agent_id)
    send_message = agent_module.send_message
    try:
        send_message(message)
    except MessageSendingError as e:
        return f"Failed to send message to agent {agent_id}. Error: {str(e)}"
    record_message_from_self(agent_id, message)
    return f"AGENT_SYSTEM: Message sent to agent {agent_id}. You will receive a notification in your FEED if the agent responds."


def list_agents():
    """List all known AGENTS with their ids, names, and short summaries."""
    raise NotImplementedError


async def open_conversation(agent_id: str):
    """Switch the OPENED_AGENT_CONVERSATION to the AGENT with the given `agent_id`. The currently open conversation will be closed."""
    previous_agent_id = config.opened_agent_conversation()
    config.set_global_state("opened_agent_conversation", agent_id)
    return f"AGENT_SYSTEM: Opened conversation with agent {agent_id}. Closed conversation with agent {previous_agent_id}."
