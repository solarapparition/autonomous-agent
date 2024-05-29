"""Configuration loader."""

from pathlib import Path

from langchain_anthropic import ChatAnthropic

from autonomous_mind.helpers import as_yaml_str, load_yaml

CONFIG_FILE = Path("data/config.yaml")
CONFIG_DATA = load_yaml(CONFIG_FILE)
GLOBAL_STATE_FILE = Path("data/global_state.yaml")
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
DEVELOPER_ID = CONFIG_DATA["developer_id"]
DEVELOPER_NAME = CONFIG_DATA["developer_name"]
SELF_DESCRIPTION = as_yaml_str(CONFIG_DATA["self_description"])
COMPUTE_RATE = str(CONFIG_DATA["compute_rate"]).format(agent_name=NAME)
MAX_RECENT_FEED_TOKENS = CONFIG_DATA["feed"]["max_recent_tokens"]
