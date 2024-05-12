"""Helpers for autonomous agent systems."""

import datetime
import os
from pathlib import Path
import time
from typing import Mapping, Any, MutableMapping, Sequence, NewType

from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

Timestamp = NewType("Timestamp", str)
TimestampFilename = NewType("TimestampFilename", str)

YAML_SAFE = YAML(typ="safe")
YAML_SAFE.default_flow_style = False
YAML_SAFE.default_style = "|"  # type: ignore
YAML_SAFE.allow_unicode = True
DEFAULT_YAML = YAML()
DEFAULT_YAML.default_flow_style = False
DEFAULT_YAML.default_style = "|"  # type: ignore
DEFAULT_YAML.allow_unicode = True


def save_yaml(
    data: Mapping[str, Any], location: Path, yaml: YAML = DEFAULT_YAML
) -> None:
    """Save YAML to a file, making sure the directory exists."""
    if not location.exists():
        os.makedirs(location.parent, exist_ok=True)
    yaml.dump(data, location)


def load_yaml(location: Path, yaml: YAML = DEFAULT_YAML) -> MutableMapping[str, Any]:
    """Load YAML from a file."""
    with location.open("r", encoding="utf-8") as file:
        return yaml.load(file)


def from_yaml_str(yaml_str: str, yaml: YAML = DEFAULT_YAML) -> MutableMapping[str, Any]:
    """Load yaml from a string."""
    return yaml.load(yaml_str)


def as_yaml_str(
    data: Mapping[str, Any] | Sequence[Any], yaml: YAML = DEFAULT_YAML
) -> str:
    """Dump yaml as a string."""
    yaml.dump(data, stream := StringIO())
    return stream.getvalue().strip()


# def get_timestamp() -> Timestamp:
#     """Get the current timestamp in UTC."""
#     return Timestamp(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "Z")


# def timestamp_to_filename(timestamp: Timestamp) -> TimestampFilename:
#     """Convert a timestamp to a filename."""
#     return TimestampFilename(timestamp.replace(":", "-").replace(" ", "_"))


# def filename_to_timestamp_final(filename: TimestampFilename) -> Timestamp:
#     """Convert a filename to a timestamp."""
#     date_part, time_part = filename.split("_")
#     time_part = time_part.replace("-", ":")
#     return Timestamp(f"{date_part} {time_part}")


def get_timestamp() -> str:
    """Get the current timestamp in UTC with microseconds."""
    return f"{datetime.datetime.now(tz=None).isoformat()}Z"


def timestamp_to_filename(timestamp: str) -> str:
    """Convert a timestamp to a filename."""
    return timestamp.replace(":", "-").replace(".", "-").replace("T", "_T")


def filename_to_timestamp(filename: str) -> str:
    """Convert a filename to a timestamp."""
    date_part, time_part = filename.split("_T")
    time_part = time_part.replace("-", ":", 2)  # Restore colons for the hour and minute
    time_part = time_part.replace("-", ".")  # Restore the period for microseconds
    return f"{date_part}T{time_part}"
