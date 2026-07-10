from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .affinity_catalog import CardSpec


@dataclass(frozen=True)
class Permanent:
    spec: CardSpec
    tapped: bool = False
    entered_this_turn: bool = False


@dataclass
class Deck:
    cards: List[CardSpec]

    def count(self) -> int:
        return len(self.cards)
