from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"{self.name} '{name}' is not registered. Available: {available}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
