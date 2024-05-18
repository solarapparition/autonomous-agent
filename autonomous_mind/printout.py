"""Printout functions for various objects."""

from autonomous_mind.helpers import as_yaml_str, from_yaml_str
from autonomous_mind.schema import Item


def full_itemized_repr(item: Item) -> str:
    """Give full representation of event as an item."""
    return as_yaml_str([from_yaml_str(repr(item))])


def short_itemized_repr(item: Item) -> str:
    """Give short representation of an item."""
    return as_yaml_str([from_yaml_str(str(item))])
