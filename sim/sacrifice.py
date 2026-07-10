from __future__ import annotations

from typing import Iterable, Optional

from .models import Permanent

TOKEN_NAMES = {'Blood Token', 'Clue Token', 'Map Token'}
STRICTLY_PROTECTED = {'Myr Enforcer', 'Utrom Monitor', 'Refurbished Familiar'}


def choose_sacrifice(
    permanents: Iterable[Permanent],
    *,
    allow_land_if_low_mana: bool = False,
) -> Optional[Permanent]:
    permanents = tuple(permanents)
    if not permanents:
        return None

    lands = [p for p in permanents if p.spec.is_land]
    non_land_candidates = [p for p in permanents if not p.spec.is_land]

    def band(p: Permanent) -> tuple[int, int, str]:
        if p.spec.ltb_draw_cards > 0:
            return (0, 0, p.spec.name)
        if p.spec.name in TOKEN_NAMES:
            return (1, 0, p.spec.name)
        if p.spec.is_artifact and not p.spec.is_creature and not p.spec.is_land:
            if p.spec.has_delayed_draw_value:
                return (3, 0, p.spec.name)
            return (2, 0, p.spec.name)
        if p.spec.name == 'Krark-Clan Shaman':
            return (4, 0, p.spec.name)
        if p.spec.is_creature:
            protected_bias = 1 if p.spec.name in STRICTLY_PROTECTED else 0
            return (5, protected_bias, p.spec.name)
        if p.spec.is_land:
            land_penalty = 0 if len(lands) >= 4 else 10
            return (6, land_penalty, p.spec.name)
        return (7, 0, p.spec.name)

    ordered = sorted(permanents, key=band)
    if non_land_candidates:
        non_land_only = [p for p in ordered if not p.spec.is_land]
        return non_land_only[0] if non_land_only else None
    if allow_land_if_low_mana and len(lands) >= 4:
        return ordered[0]
    return None
