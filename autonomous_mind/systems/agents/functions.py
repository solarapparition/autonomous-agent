"""Functions for interacting with AGENTS."""

from pathlib import Path
import importlib.util

from autonomous_mind.config import AGENTS_DIRECTORY

class MessageSendingError(Exception):
    """Raised when a message failed to be sent to an AGENT."""
    pass


async def message_agent(id: str, message: str):
    """Send a message to an AGENT with the given `id`."""
    
    agent_package_path = Path(f"{AGENTS_DIRECTORY}/{id}/__init__.py")
    spec = importlib.util.spec_from_file_location(id, agent_package_path)
    assert spec
    assert spec.loader
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    send_message = agent_module.send_message

    try:
        send_message(message)
        breakpoint()
        return f"Message sent to agent {id}. You will receive a notification in your FEED if the agent responds."
    except MessageSendingError as e:
        return f"Failed to send message to agent {id}. Error: {str(e)}"


    # > chatty
    # read in message sending function from agent's module
    breakpoint()
    # send message
    # record message in log for agent
    raise NotImplementedError

    return f"Message sent to agent {id}. You will receive a notification in your FEED if the agent responds."



def test_successful_message():
    result = message_agent('successful_agent', 'Hello, World!')
    assert result == "Message sent to agent successful_agent. You will receive a notification in your FEED if the agent responds."

def test_failing_message():
    result = message_agent('failing_agent', 'Hello, World!')
    assert "Failed to send message to agent failing_agent. Error: Simulated failure" in result


# implement contacts list
# implement receiving message

def list_agents():
    """List all known AGENTS with their ids, names, and short summaries."""
    raise NotImplementedError
