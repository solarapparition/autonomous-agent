"""Configuration loader."""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from autonomous_agent.helpers import load_yaml

CONFIG_FILE = Path("data/config.yaml")
CONFIG_DATA = load_yaml(CONFIG_FILE)
STATE_FILE = Path("data/run_state.yaml")
SOURCE_DIRECTORY = Path(__file__).parent.parent
BUILD_CONFIG_FILE = Path("pyproject.toml")

NAME = CONFIG_DATA["name"]
ID = CONFIG_DATA["id"]
LLM_BACKEND = CONFIG_DATA["llm_backend"]
CORE_MODEL = ChatAnthropic(  # type: ignore
    temperature=0.8, model=LLM_BACKEND, verbose=False, max_tokens_to_sample=4096  # type: ignore
)
DEVELOPER = CONFIG_DATA["developer"]
SELF_DESCRIPTION = str(CONFIG_DATA["self_description"]).format(agent_name=NAME)
COMPUTE_RATE = str(CONFIG_DATA["compute_rate"]).format(agent_name=NAME)
