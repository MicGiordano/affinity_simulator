from __future__ import annotations

from typing import Sequence

from .affinity_catalog import CardSpec


def keepable_opening_hand(hand: Sequence[CardSpec]) -> bool:
    lands = [c for c in hand if c.is_land]
    nonlands = [c for c in hand if not c.is_land]
    if not (2 <= len(lands) <= 4):
        return False
    if not nonlands:
        return False
    if any(card.name in {'Thoughtcast', 'Ichor Wellspring', "Reckoner's Bargain", 'Fanatical Offering'} for card in nonlands):
        return True
    return True


def sort_for_bottom(hand: Sequence[CardSpec]) -> list[CardSpec]:
    def priority(card: CardSpec):
        if card.name == 'Makeshift Munitions':
            return (0, card.name)
        if card.name == 'Galvanic Blast':
            return (1, card.name)
        if card.is_land:
            return (4, card.name)
        return (10, card.name)

    return sorted(hand, key=priority)
