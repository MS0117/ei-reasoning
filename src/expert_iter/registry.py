"""Name -> class registries for the pluggable research components.

Adding a new anchor policy / operator / gate / adapter / verifier is:
one new class + one @register decorator; config then selects it by name.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

ADAPTERS: dict[str, type] = {}
VERIFIERS: dict[str, type] = {}
ANCHOR_POLICIES: dict[str, type] = {}
OPERATORS: dict[str, type] = {}
GATES: dict[str, type] = {}


def register(registry: dict[str, type], name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        if name in registry:
            raise ValueError(f"duplicate registration for {name!r}")
        registry[name] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return deco


def build(registry: dict[str, type], name: str, *args, **kwargs):
    if name not in registry:
        raise KeyError(
            f"unknown component {name!r}; registered: {sorted(registry)}"
        )
    return registry[name](*args, **kwargs)
