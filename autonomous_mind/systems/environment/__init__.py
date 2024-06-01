"""This module is responsible for managing the environment in which the systems operate."""

from autonomous_mind.systems.config import settings
from autonomous_mind.systems.environment.shell import PersistentShell


shell = PersistentShell(settings.SHELL_NAME)
