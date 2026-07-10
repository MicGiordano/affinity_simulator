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
    land_count = sum(card.is_land for card in hand)

    def priority(card: CardSpec):
        # If we already have 3+ lands, lands become better candidates to bottom
        if land_count >= 3 and card.is_land:
            return (0, card.name)

        # Keep important spells if we don't have enough lands
        if card.name == 'Makeshift Munitions':
            return (1, card.name)
        if card.name == 'Galvanic Blast':
            return (2, card.name)
        if card.name == 'Toxin Analysis':
            return (3, card.name)

        # Otherwise, non-essential cards are higher priority to bottom
        if not card.is_land:
            return (5, card.name)

        # Lands are protected when we have fewer than 3
        return (20, card.name)

    return sorted(hand, key=priority)