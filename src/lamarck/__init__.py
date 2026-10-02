"""Lamarck: evolve a coding agent's harness against verified tasks, with held-out proof."""

from .harness import Edit, Harness
from .tasks import Task, TaskSuite

__version__ = "0.1.0"
__all__ = ["Edit", "Harness", "Task", "TaskSuite", "__version__"]
