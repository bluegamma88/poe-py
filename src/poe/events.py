"""Small UI-independent messages shared by the provider and agent."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    kind: str
    text: str = ""
    data: Any = None


Emit = Callable[[Event], Awaitable[None]]
