"""Configuration loader."""

from pathlib import Path
from typing import Any, MutableMapping

from langchain_anthropic import ChatAnthropic

from autonomous_mind.helpers import as_yaml_str, load_yaml

CONFIG_FILE = Path("data/config.yaml")
CONFIG_DATA = load_yaml(CONFIG_FILE)
GLOBAL_STATE_FILE = Path("data/global_state.yaml")
GLOBAL_STATE: MutableMapping[str, Any] = load_yaml(GLOBAL_STATE_FILE)
RUN_STATE_DIRECTORY = Path("data/run_state")
RUN_STATE_FILE = RUN_STATE_DIRECTORY / "current.yaml"
RUN_STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
SOURCE_DIRECTORY = Path(__file__).parent.parent
BUILD_CONFIG_FILE = Path("pyproject.toml")
EVENTS_DIRECTORY = Path("data/events")
EVENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
GOALS_DIRECTORY = Path("data/goals")
GOALS_DIRECTORY.mkdir(parents=True, exist_ok=True)
AGENTS_DIRECTORY = Path("data/agents")
AGENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
NOTES_DIRECTORY = Path("data/notes")
NOTES_DIRECTORY.mkdir(parents=True, exist_ok=True)

NAME = CONFIG_DATA["name"]
ID = CONFIG_DATA["id"]
LLM_BACKEND = CONFIG_DATA["llm_backend"]
CORE_MODEL = ChatAnthropic(  # type: ignore
    temperature=0.8, model=LLM_BACKEND, verbose=False, max_tokens_to_sample=4096  # type: ignore
)
DEVELOPER = CONFIG_DATA["developer"]
SELF_DESCRIPTION = as_yaml_str(CONFIG_DATA["self_description"])
COMPUTE_RATE = str(CONFIG_DATA["compute_rate"]).format(agent_name=NAME)
MAX_RECENT_FEED_TOKENS = CONFIG_DATA["feed"]["max_recent_tokens"]
OPENED_AGENT_CONVERSATION = GLOBAL_STATE.get("opened_agent_conversation")


def set_global_state(key: str, value: Any) -> None:
    """Set a global state."""
    GLOBAL_STATE[key] = value
    GLOBAL_STATE_FILE.write_text(as_yaml_str(GLOBAL_STATE), encoding="utf-8")


def action_batch_number() -> int:
    """Get the current action batch number."""
    return GLOBAL_STATE["action_batch_number"]


def opened_agent_conversation() -> int | None:
    """Get the current opened agent conversation."""
    return GLOBAL_STATE.get("opened_agent_conversation")


def loaded_notes() -> list[str]:
    """Get the current loaded notes."""
    return GLOBAL_STATE.get("loaded_notes", [])


def last_id() -> int:
    """Get the last id."""
    return GLOBAL_STATE.get("last_id", 0)


def update_last_id(new_id: int) -> None:
    """Update the last id."""
    set_global_state("last_id", new_id)
