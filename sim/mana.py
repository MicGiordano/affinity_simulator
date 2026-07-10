from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Optional, Tuple

from .affinity_catalog import CardSpec
from .models import Permanent


def artifact_count(permanents: Iterable[Permanent]) -> int:
    return sum(1 for p in permanents if p.spec.is_artifact)


def effective_total_cost(card: CardSpec, permanents: Iterable[Permanent]) -> int:
    """
    Affinity reduces only the generic (colorless) portion of a spell's cost,
    never colored pips.
    """
    total = card.mana_value

    # Colored mana cannot be reduced
    colored = sum(card.color_pips.values())

    # Generic portion can be reduced
    generic = max(0, total - colored)

    if card.affinity_for_artifacts:
        generic = max(0, generic - artifact_count(permanents))

    return colored + generic


def untapped_lands(permanents: Iterable[Permanent]) -> Tuple[Permanent, ...]:
    return tuple(p for p in permanents if p.spec.is_land and not p.tapped)


def _can_cover(chosen_specs, pips: Counter) -> bool:
    """
    Check whether the chosen lands can satisfy the colored pip requirements.
    """
    remaining = Counter(pips)

    for spec in chosen_specs:
        for color in spec.land_colors:
            if remaining[color] > 0:
                remaining[color] -= 1
                break

    return all(v == 0 for v in remaining.values())


def _land_flexibility_penalty(spec: CardSpec) -> int:
    """
    Lower penalty is better.

    Single-color lands are preferred over flexible lands.
    Example:
      Vault of Whispers -> 0
      Drossforge Bridge -> 1
    """
    return max(0, len(spec.land_colors) - 1)


def _payment_choice_key(
    chosen_indexes: Tuple[int, ...],
    battlefield: Tuple[Permanent, ...],
    pips: Counter,
):
    """
    Choose the most conservative payment among all valid payments:
    1) prefer lands with fewer colors (mono-color first)
    2) prefer using fewer flexible/artifact bridge lands
    3) keep it deterministic by land names
    """
    specs = [battlefield[i].spec for i in chosen_indexes]

    return (
        sum(_land_flexibility_penalty(spec) for spec in specs),
        sum(1 for spec in specs if spec.is_artifact and len(spec.land_colors) > 1),
        tuple(spec.name for spec in specs),
    )


def choose_lands_for_payment(
    total: int,
    pips: Counter,
    permanents: Iterable[Permanent],
) -> Optional[Tuple[int, ...]]:
    battlefield = tuple(permanents)
    idx = tuple(i for i, p in enumerate(battlefield) if p.spec.is_land and not p.tapped)

    if total == 0:
        return ()

    if len(idx) < total:
        return None

    valid_choices = []

    for chosen in combinations(idx, total):
        chosen_specs = [battlefield[i].spec for i in chosen]
        if not pips or _can_cover(chosen_specs, pips):
            valid_choices.append(tuple(chosen))

    if not valid_choices:
        return None

    valid_choices.sort(key=lambda chosen: _payment_choice_key(chosen, battlefield, pips))
    return valid_choices[0]


def choose_lands_for_card(
    card: CardSpec,
    permanents: Iterable[Permanent],
) -> Optional[Tuple[int, ...]]:
    return choose_lands_for_payment(
        effective_total_cost(card, permanents),
        card.color_pips,
        permanents,
    )


def can_pay_with_lands(card: CardSpec, permanents: Iterable[Permanent]) -> bool:
    return choose_lands_for_card(card, permanents) is not None


def can_pay_ability(total: int, pips: Counter, permanents: Iterable[Permanent]) -> bool:
    return choose_lands_for_payment(total, pips, permanents) is not None


def tap_payment_lands(
    permanents: Iterable[Permanent],
    indexes: Iterable[int],
) -> Tuple[Permanent, ...]:
    indexes = set(indexes)
    out = []
    for i, permanent in enumerate(tuple(permanents)):
        if i in indexes:
            out.append(Permanent(permanent.spec, True, permanent.entered_this_turn))
        else:
            out.append(permanent)
    return tuple(out)


def projected_colors_after_land_drops(
    current: Iterable[Permanent],
    hand_lands: Iterable[CardSpec],
    turns_ahead: int,
) -> set[str]:
    colors = set()

    for permanent in current:
        if permanent.spec.is_land:
            colors.update(permanent.spec.land_colors)

    for land in list(hand_lands)[:turns_ahead]:
        colors.update(land.land_colors)

    return colors