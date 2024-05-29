"""Global state management."""

from dataclasses import dataclass, field
from typing import Any, MutableMapping
from autonomous_mind.schema import ItemId
from autonomous_mind.systems.config.settings import GLOBAL_STATE_FILE
from autonomous_mind.helpers import as_yaml_str, load_yaml


@dataclass
class GlobalState:
    """Global state management."""

    mapping: MutableMapping[str, Any] = field(
        default_factory=lambda: load_yaml(GLOBAL_STATE_FILE),
        init=False,
    )

    def set_value(self, key: str, value: Any) -> None:
        """Set a global state variable."""
        self.mapping[key] = value
        GLOBAL_STATE_FILE.write_text(as_yaml_str(self.mapping), encoding="utf-8")

    @property
    def action_batch_number(self) -> int:
        """Get the current action batch number."""
        return self.mapping["action_batch_number"]

    @action_batch_number.setter
    def action_batch_number(self, value: int) -> None:
        """Set the current action batch number."""
        self.set_value("action_batch_number", value)

    @property
    def focused_goal_id(self) -> ItemId | None:
        """Get the current focused goal id."""
        return self.mapping.get("focused_goal_id")

    @focused_goal_id.setter
    def focused_goal_id(self, value: ItemId | None) -> None:
        """Set the current focused goal id."""
        self.set_value("focused_goal_id", value)

    @property
    def opened_agent_id(self) -> ItemId | None:
        """Get the current opened agent conversation."""
        return self.mapping["opened_agent_conversation"]

    @opened_agent_id.setter
    def opened_agent_id(self, value: ItemId | None) -> None:
        """Set the current opened agent conversation."""
        self.set_value("opened_agent_conversation", value)

    @property
    def loaded_notes(self) -> list[ItemId]:
        """Get the current loaded notes."""
        return self.mapping.get("loaded_notes", [])

    @loaded_notes.setter
    def loaded_notes(self, value: list[ItemId]) -> None:
        """Set the current loaded notes."""
        self.set_value("loaded_notes", value)

    @property
    def last_id(self) -> int:
        """Get the last id."""
        return self.mapping.get("last_id", 0)

    @last_id.setter
    def last_id(self, value: int) -> None:
        """Set the last id."""
        self.set_value("last_id", value)


global_state = GlobalState()
