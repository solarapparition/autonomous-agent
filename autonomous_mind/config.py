"""Configuration loader."""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI

from autonomous_mind.helpers import load_yaml

CONFIG_FILE = Path("data/config.yaml")
CONFIG_DATA = load_yaml(CONFIG_FILE)
GLOBAL_STATE_FILE = Path("data/global_state.yaml")
GLOBAL_STATE = load_yaml(GLOBAL_STATE_FILE)
RUN_STATE_DIRECTORY = Path("data/run_state")
RUN_STATE_FILE = RUN_STATE_DIRECTORY / "current.yaml"
RUN_STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
SOURCE_DIRECTORY = Path(__file__).parent.parent
BUILD_CONFIG_FILE = Path("pyproject.toml")
EVENTS_DIRECTORY = Path("data/events")
EVENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

NAME = CONFIG_DATA["name"]
ID = CONFIG_DATA["id"]
LLM_BACKEND = CONFIG_DATA["llm_backend"]
CORE_MODEL = ChatAnthropic(  # type: ignore
    temperature=0.8, model=LLM_BACKEND, verbose=False, max_tokens_to_sample=4096  # type: ignore
)
# CORE_MODEL = ChatOpenAI(
#     temperature=0.8, model=LLM_BACKEND, verbose=False
# )
DEVELOPER = CONFIG_DATA["developer"]
SELF_DESCRIPTION = str(CONFIG_DATA["self_description"]).format(agent_name=NAME)
COMPUTE_RATE = str(CONFIG_DATA["compute_rate"]).format(agent_name=NAME)
MAX_RECENT_FEED_TOKENS = CONFIG_DATA["feed"]["max_recent_tokens"]
