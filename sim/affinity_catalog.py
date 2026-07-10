from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ActivatedAbility:
    total_mana_cost: int = 0
    color_pips: Counter = field(default_factory=Counter)
    sacrifice_self: bool = False
    draw_cards: int = 0
    discard_cards: int = 0


@dataclass(frozen=True)
class CardSpec:
    name: str
    mana_cost: str = ""
    mana_value: int = 0
    color_pips: Counter = field(default_factory=Counter)
    categories: frozenset = frozenset()

    # Activated / triggered behavior
    activated_ability: Optional[ActivatedAbility] = None
    creates_token_on_cast: Optional[str] = None
    creates_token_on_resolve: Optional[str] = None
    resolve_draw_cards: int = 0
    etb_draw_cards: int = 0
    ltb_draw_cards: int = 0

    # Permanent / typing flags
    is_creature_flag: bool = False
    land_colors: Tuple[str, ...] = ()
    enters_tapped: bool = False

    # Gameplay / planner flags
    affinity_for_artifacts: bool = False
    sac_fodder: bool = False
    has_delayed_draw_value: bool = False
    requires_creature: bool = False
    requires_sacrifice: bool = False

    @property
    def is_artifact(self) -> bool:
        return "artifact" in self.categories

    @property
    def is_land(self) -> bool:
        return len(self.land_colors) > 0

    @property
    def is_creature(self) -> bool:
        return self.is_creature_flag

    @property
    def is_permanent(self) -> bool:
        return (
            self.is_land
            or self.is_artifact
            or self.is_creature
            or "enchantment" in self.categories
        )


def cs(*x):
    return frozenset(x)


def mc(cost: str) -> Counter:
    out = Counter()
    for ch in cost:
        if ch in {"B", "U", "R", "G", "W"}:
            out[ch] += 1
    return out


def token_spec(name: str) -> CardSpec:
    if name == "Blood Token":
        return CardSpec(
            name="Blood Token",
            categories=cs("artifact", "token"),
            activated_ability=ActivatedAbility(
                total_mana_cost=1,
                color_pips=Counter(),
                sacrifice_self=True,
                draw_cards=1,
                discard_cards=1,
            ),
            sac_fodder=True,
        )

    if name == "Clue Token":
        return CardSpec(
            name="Clue Token",
            categories=cs("artifact", "token"),
            activated_ability=ActivatedAbility(
                total_mana_cost=2,
                color_pips=Counter(),
                sacrifice_self=True,
                draw_cards=1,
                discard_cards=0,
            ),
            sac_fodder=True,
        )

    if name == "Map Token":
        return CardSpec(
            name="Map Token",
            categories=cs("artifact", "token"),
            sac_fodder=True,
        )

    if name == "Hero Token":
        return CardSpec(
            name="Hero Token",
            categories=cs("creature", "token"),
            is_creature_flag=True,
            sac_fodder=True,
        )

    raise KeyError(name)


CARDS = {
    # -------------------------
    # Lands
    # -------------------------
    "Drossforge Bridge": CardSpec(
        name="Drossforge Bridge",
        categories=cs("artifact", "land", "permanent"),
        land_colors=("B", "R"),
        enters_tapped=True,
    ),
    "Mistvault Bridge": CardSpec(
        name="Mistvault Bridge",
        categories=cs("artifact", "land", "permanent"),
        land_colors=("U", "B"),
        enters_tapped=True,
    ),
    "Silverbluff Bridge": CardSpec(
        name="Silverbluff Bridge",
        categories=cs("artifact", "land", "permanent"),
        land_colors=("U", "R"),
        enters_tapped=True,
    ),
    "Vault of Whispers": CardSpec(
        name="Vault of Whispers",
        categories=cs("artifact", "land", "permanent"),
        land_colors=("B",),
    ),
    "Seat of the Synod": CardSpec(
        name="Seat of the Synod",
        categories=cs("artifact", "land", "permanent"),
        land_colors=("U",),
    ),
    "Great Furnace": CardSpec(
        name="Great Furnace",
        categories=cs("artifact", "land", "permanent"),
        land_colors=("R",),
    ),
    "Swamp": CardSpec(
        name="Swamp",
        categories=cs("land", "permanent", "permanent"),
        land_colors=("B",),
    ),
    "Island": CardSpec(
        name="Island",
        categories=cs("land", "permanent", "permanent"),
        land_colors=("U",),
    ),
    "Mountain": CardSpec(
        name="Mountain",
        categories=cs("land", "permanent", "permanent"),
        land_colors=("R",),
    ),

    # -------------------------
    # Main deck cards
    # -------------------------
    "Black Mage's Rod": CardSpec(
        name="Black Mage's Rod",
        mana_cost="B",
        mana_value=1,
        color_pips=mc("B"),
        categories=cs("artifact", "permanent"),
        sac_fodder=True,
        creates_token_on_cast="Hero Token",
    ),
    "Blood Fountain": CardSpec(
        name="Blood Fountain",
        mana_cost="B",
        mana_value=1,
        color_pips=mc("B"),
        categories=cs("artifact", "permanent"),
        creates_token_on_cast="Blood Token",
        sac_fodder=True,
    ),
    "Cryogen Relic": CardSpec(
        name="Cryogen Relic",
        mana_cost="1U",
        mana_value=2,
        color_pips=mc("U"),
        categories=cs("artifact", "permanent"),
        activated_ability=ActivatedAbility(
            total_mana_cost=2,
            color_pips=mc("U"),
            sacrifice_self=True,
            draw_cards=0,
            discard_cards=0,
        ),
        etb_draw_cards=1,
        ltb_draw_cards=1,
        sac_fodder=True,
        has_delayed_draw_value=True,
    ),
    "Fanatical Offering": CardSpec(
        name="Fanatical Offering",
        mana_cost="1B",
        mana_value=2,
        color_pips=mc("B"),
        categories=cs(),
        resolve_draw_cards=2,
        creates_token_on_resolve="Map Token",
        requires_sacrifice=True,
    ),
    "Galvanic Blast": CardSpec(
        name="Galvanic Blast",
        mana_cost="R",
        mana_value=1,
        color_pips=mc("R"),
        categories=cs(),
    ),
    "Ichor Wellspring": CardSpec(
        name="Ichor Wellspring",
        mana_cost="2",
        mana_value=2,
        categories=cs("artifact", "permanent"),
        etb_draw_cards=1,
        ltb_draw_cards=1,
        sac_fodder=True,
        has_delayed_draw_value=True,
    ),
    "Krark-Clan Shaman": CardSpec(
        name="Krark-Clan Shaman",
        mana_cost="R",
        mana_value=1,
        color_pips=mc("R"),
        categories=cs("creature", "permanent"),
        is_creature_flag=True,
        sac_fodder=True,
    ),
    "Makeshift Munitions": CardSpec(
        name="Makeshift Munitions",
        mana_cost="1R",
        mana_value=2,
        color_pips=mc("R"),
        categories=cs("enchantment", "permanent"),
    ),
    "Myr Enforcer": CardSpec(
        name="Myr Enforcer",
        mana_cost="7",
        mana_value=7,
        categories=cs("artifact", "creature", "permanent"),
        is_creature_flag=True,
        affinity_for_artifacts=True,
    ),
    "Nihil Spellbomb": CardSpec(
        name="Nihil Spellbomb",
        mana_cost="1",
        mana_value=1,
        categories=cs("artifact", "permanent"),
        activated_ability=ActivatedAbility(
            total_mana_cost=1,
            color_pips=mc("B"),
            sacrifice_self=True,
            draw_cards=1,
            discard_cards=0,
        ),
        sac_fodder=True,
    ),
    "Reckoner's Bargain": CardSpec(
        name="Reckoner's Bargain",
        mana_cost="1B",
        mana_value=2,
        color_pips=mc("B"),
        categories=cs(),
        resolve_draw_cards=2,
        requires_sacrifice=True,
    ),
    "Refurbished Familiar": CardSpec(
        name="Refurbished Familiar",
        mana_cost="2B",
        mana_value=3,
        color_pips=mc("B"),
        categories=cs("artifact", "creature", "permanent"),
        is_creature_flag=True,
    ),
    "Sewer-veillance Cam": CardSpec(
        name="Sewer-veillance Cam",
        mana_cost="1",
        mana_value=1,
        categories=cs("artifact", "permanent"),
        sac_fodder=True,
    ),
    "Thoughtcast": CardSpec(
        name="Thoughtcast",
        mana_cost="4U",
        mana_value=5,
        color_pips=mc("U"),
        categories=cs(),
        affinity_for_artifacts=True,
        resolve_draw_cards=2,
    ),
    "Toxin Analysis": CardSpec(
        name="Toxin Analysis",
        mana_cost="B",
        mana_value=1,
        color_pips=mc("B"),
        categories=cs(),
        creates_token_on_resolve="Clue Token",
        requires_creature=True,
    ),
    "Utrom Monitor": CardSpec(
        name="Utrom Monitor",
        mana_cost="7",
        mana_value=7,
        categories=cs("artifact", "creature", "permanent"),
        is_creature_flag=True,
        affinity_for_artifacts=True,
    ),
}


def get_card_spec(name: str) -> CardSpec:
    if name in CARDS:
        return CARDS[name]
    raise KeyError(f"Unknown card: {name}")