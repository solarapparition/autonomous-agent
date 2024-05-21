"""Environment system functions."""


from typing import Literal


def sleep(mode: Literal["until", "hour", "minute", "second"], time: str | int):
    """Put {mind_name} to sleep until a specific UTC timestamp or for a specific duration."""
    if mode == "until":
        return f"Sleeping until {time}."
    return f"SYSTEM: Sleeping for {time} {mode}(s)."
