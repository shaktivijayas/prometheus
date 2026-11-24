"""Creators: the intelligences that design, teach and evolve children."""
from __future__ import annotations

from typing import Callable

from .base import ChildSpec, Creator, GenerationReport
from .clone import CloneCreator
from .heuristic import HeuristicCreator
from .random_mutation import RandomMutationCreator

_FACTORIES: dict[str, Callable[[str], Creator]] = {
    "random": lambda arg: RandomMutationCreator(),
    "heuristic": lambda arg: HeuristicCreator(),
    "clone": lambda arg: CloneCreator(),
}
CREATOR_NAMES = _FACTORIES  # live view: the neural Creator registers itself here


def register_creator(name: str, factory: Callable[[str], Creator]) -> None:
    _FACTORIES[name] = factory


def get_creator(name: str) -> Creator:
    """`name` may carry an argument after a colon, e.g. "neural:v003"."""
    base, _, arg = name.partition(":")
    if base not in _FACTORIES:
        raise ValueError(f"unknown creator {name!r}; valid: {sorted(_FACTORIES)}")
    return _FACTORIES[base](arg)


__all__ = ["ChildSpec", "Creator", "GenerationReport", "CREATOR_NAMES", "get_creator", "register_creator"]
