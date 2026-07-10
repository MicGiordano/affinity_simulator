from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

from .affinity_catalog import CardSpec, get_card_spec, token_spec
from .mana import (
    can_pay_ability,
    can_pay_with_lands,
    choose_lands_for_card,
    choose_lands_for_payment,
    effective_total_cost,
    tap_payment_lands,
)
from .models import Permanent
from .sacrifice import choose_sacrifice


@dataclass(frozen=True)
class Action:
    kind: str
    card_name: Optional[str] = None

    def label(self) -> str:
        if self.kind == "play_land":
            return f"play {self.card_name}"
        if self.kind == "cast":
            return f"cast {self.card_name}"
        if self.kind == "activate":
            return f"activate {self.card_name}"
        return self.kind


@dataclass(frozen=True)
class PlanningState:
    hand: Tuple[CardSpec, ...]
    battlefield: Tuple[Permanent, ...]
    graveyard: Tuple[CardSpec, ...]
    history: Tuple[dict, ...] = ()
    mana_spent_this_turn: int = 0
    tempo_loss_this_turn: int = 0
    land_played_this_turn: bool = False
    debug_last_event: Optional[dict] = field(default=None, compare=False)


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def _artifact_count(state: PlanningState) -> int:
    return sum(1 for p in state.battlefield if p.spec.is_artifact)


def _creature_count(state: PlanningState) -> int:
    return sum(
        1
        for p in state.battlefield
        if p.spec.is_creature and p.spec.name != "Krark-Clan Shaman"
    )


def _land_count(state: PlanningState) -> int:
    return sum(1 for p in state.battlefield if p.spec.is_land)


def _shaman_count(state: PlanningState) -> int:
    return sum(1 for p in state.battlefield if p.spec.name == "Krark-Clan Shaman")


def _is_bridge(card: CardSpec) -> bool:
    return card.is_land and card.is_artifact and getattr(card, "enters_tapped", False)


def _estimated_turn_number(state: PlanningState) -> int:
    return _land_count(state) + 1


def _payment_details(indexes, battlefield: Tuple[Permanent, ...]) -> dict:
    idx = tuple(indexes)
    specs = [battlefield[i].spec for i in idx]
    return {
        "mana_payment": [spec.name for spec in specs],
        "colors_produced": [
            "/".join(spec.land_colors) if spec.land_colors else "C" for spec in specs
        ],
    }


def _remove_first_card(cards, name: str):
    cards = list(cards)
    for idx, c in enumerate(cards):
        if c.name == name:
            removed = cards.pop(idx)
            return removed, tuple(cards)
    raise ValueError(f"Could not find card {name}")


def _remove_first_permanent(permanents, name: str):
    permanents = list(permanents)
    for idx, p in enumerate(permanents):
        if p.spec.name == name:
            removed = permanents.pop(idx)
            return removed, tuple(permanents)
    raise ValueError(f"Could not find permanent {name}")


def _draw_priority_bonus(card: CardSpec) -> float:
    bonus = 0.0
    if card.name == "Thoughtcast":
        bonus += 12.0
    if card.name == "Ichor Wellspring":
        bonus += 10.0
    if card.name in {"Reckoner's Bargain", "Fanatical Offering"}:
        bonus += 9.0
    if card.name in {"Blood Token", "Clue Token", "Nihil Spellbomb"}:
        bonus += 6.0

    if getattr(card, "resolve_draw_cards", 0):
        bonus += 3.0 * getattr(card, "resolve_draw_cards", 0)
    if card.etb_draw_cards:
        bonus += 2.5 * card.etb_draw_cards
    if card.ltb_draw_cards:
        bonus += 1.5 * card.ltb_draw_cards

    return bonus


def _is_priority_draw_card(card: CardSpec) -> bool:
    return (
        card.name in {"Thoughtcast", "Ichor Wellspring", "Reckoner's Bargain", "Fanatical Offering"}
        or getattr(card, "resolve_draw_cards", 0) > 0
        or card.etb_draw_cards > 0
    )


def _choose_discard(state: PlanningState) -> Optional[CardSpec]:
    if not state.hand:
        return None

    lands_in_hand = [c for c in state.hand if c.is_land]
    nonlands_in_hand = [c for c in state.hand if not c.is_land]
    lands_in_play = _land_count(state)

    desired_total_lands = min(4, lands_in_play + 2)
    excess_lands = len(lands_in_hand) - max(0, desired_total_lands - lands_in_play)
    if excess_lands > 0:
        bridge_lands = [c for c in lands_in_hand if _is_bridge(c)]
        if bridge_lands:
            return bridge_lands[-1]
        return lands_in_hand[-1]

    for name in ("Galvanic Blast", "Makeshift Munitions"):
        for card in state.hand:
            if card.name == name:
                return card

    if nonlands_in_hand:
        expensive_candidates = [
            c
            for c in nonlands_in_hand
            if effective_total_cost(c, state.battlefield) > lands_in_play + 1
        ]
        if expensive_candidates:
            return max(expensive_candidates, key=lambda c: c.mana_value)

    protected = {"Thoughtcast", "Ichor Wellspring", "Reckoner's Bargain", "Fanatical Offering"}
    disposable = [c for c in nonlands_in_hand if c.name not in protected]
    if disposable:
        return max(disposable, key=lambda c: (c.mana_value, c.name))

    if nonlands_in_hand:
        return max(nonlands_in_hand, key=lambda c: (c.mana_value, c.name))

    return state.hand[0]


def _should_use_blood(state: PlanningState) -> bool:
    if not state.hand:
        return False

    lands_in_hand = sum(1 for c in state.hand if c.is_land)
    lands_in_play = _land_count(state)
    turn_estimate = _estimated_turn_number(state)

    discard = _choose_discard(state)
    if discard is None:
        return False

    if turn_estimate <= 4 and lands_in_hand == 0 and lands_in_play < turn_estimate:
        return True
    if lands_in_hand >= 4:
        return True

    playable_nonlands = [
        c for c in state.hand if not c.is_land and can_pay_with_lands(c, state.battlefield)
    ]
    if not playable_nonlands and lands_in_hand <= 1:
        return True

    if discard.name in {"Galvanic Blast", "Makeshift Munitions"}:
        return True

    return False


def _play_land_priority_key(card: CardSpec, state: PlanningState):
    turn_estimate = _estimated_turn_number(state)
    if turn_estimate <= 2:
        return (
            0 if _is_bridge(card) else 1,
            0 if getattr(card, "enters_tapped", False) else 1,
            card.name,
        )
    return (
        1 if getattr(card, "enters_tapped", False) else 0,
        0 if _is_bridge(card) else 1,
        card.name,
    )


def _battlefield_with_played_land(state: PlanningState, card: CardSpec) -> Tuple[Permanent, ...]:
    return state.battlefield + (
        Permanent(card, tapped=getattr(card, "enters_tapped", False), entered_this_turn=False),
    )


def _generate_nonland_actions(state: PlanningState) -> Tuple[Action, ...]:
    actions = []

    for card in state.hand:
        if card.is_land:
            continue
        if card.name == "Galvanic Blast":
            continue
        if not can_pay_with_lands(card, state.battlefield):
            continue
        if getattr(card, "requires_creature", False) and _creature_count(state) <= 0:
            continue
        if getattr(card, "requires_sacrifice", False) and choose_sacrifice(state.battlefield) is None:
            continue
        actions.append(Action("cast", card.name))

    for permanent in state.battlefield:
        ability = permanent.spec.activated_ability
        if ability is None:
            continue
        if not can_pay_ability(ability.total_mana_cost, ability.color_pips, state.battlefield):
            continue
        if permanent.spec.name == "Blood Token":
            if getattr(ability, "discard_cards", 0) > 0 and len(state.hand) == 0:
                continue
            if not _should_use_blood(state):
                continue
        if getattr(ability, "discard_cards", 0) > 0 and len(state.hand) == 0:
            continue
        actions.append(Action("activate", permanent.spec.name))

    return tuple(actions)


def _has_priority_draw_action(actions: Tuple[Action, ...]) -> bool:
    for action in actions:
        if action.kind == "cast" and action.card_name:
            card = get_card_spec(action.card_name)
            if _is_priority_draw_card(card):
                return True
        if action.kind == "activate" and action.card_name in {"Blood Token", "Clue Token", "Nihil Spellbomb", "Cryogen Relic"}:
            return True
    return False


def _land_enables_priority_action(state: PlanningState, land: CardSpec) -> bool:
    before = _generate_nonland_actions(state)
    before_draw = _has_priority_draw_action(before)
    after_state = replace(state, battlefield=_battlefield_with_played_land(state, land))
    after = _generate_nonland_actions(after_state)
    after_draw = _has_priority_draw_action(after)
    return after_draw and not before_draw


def _land_enables_any_action(state: PlanningState, land: CardSpec) -> bool:
    before = _generate_nonland_actions(state)
    after_state = replace(state, battlefield=_battlefield_with_played_land(state, land))
    after = _generate_nonland_actions(after_state)
    return len(after) > len(before) or (not before and bool(after))


# ----------------------------------------------------------
# ACTION GENERATION
# ----------------------------------------------------------

def generate_actions(state: PlanningState) -> Tuple[Action, ...]:
    nonland_actions = list(_generate_nonland_actions(state))
    actions = list(nonland_actions)

    if not state.land_played_this_turn:
        lands = [c for c in state.hand if c.is_land]
        lands.sort(key=lambda c: _play_land_priority_key(c, state))

        immediate_draw_available = _has_priority_draw_action(tuple(nonland_actions))
        immediate_nonland_available = bool(nonland_actions)

        land_actions: list[Action] = []
        for card in lands:
            if immediate_draw_available:
                if _is_bridge(card):
                    land_actions.append(Action("play_land", card.name))
                continue

            if immediate_nonland_available:
                if _land_enables_priority_action(state, card):
                    land_actions.append(Action("play_land", card.name))
                    continue
                if _estimated_turn_number(state) <= 2 and _is_bridge(card):
                    land_actions.append(Action("play_land", card.name))
                    continue
                continue

            if _land_enables_any_action(state, card) or not nonland_actions:
                land_actions.append(Action("play_land", card.name))

        actions.extend(land_actions)

    return tuple(actions)


# ----------------------------------------------------------
# APPLY ACTION
# ----------------------------------------------------------

def apply_action(state: PlanningState, action: Action) -> PlanningState:
    hand = list(state.hand)
    battlefield = tuple(state.battlefield)
    graveyard = list(state.graveyard)
    history = list(state.history)

    if action.kind == "play_land":
        card, remaining_hand = _remove_first_card(hand, action.card_name)

        before_nonland = _generate_nonland_actions(state)
        before_draw = _has_priority_draw_action(before_nonland)
        new_battlefield = _battlefield_with_played_land(state, card)
        after_state = replace(state, battlefield=new_battlefield)
        after_nonland = _generate_nonland_actions(after_state)
        land_needed_now = bool(after_nonland) and not bool(before_nonland)
        land_enabled_priority = _has_priority_draw_action(after_nonland) and not before_draw

        event = {
            "kind": "play_land",
            "name": card.name,
            "mana_payment": [],
            "colors_produced": [],
            "total_mana_spent": 0,
            "effective_cost": 0,
            "sacrificed": None,
            "discarded": None,
            "draw_count": 0,
            "created_token": None,
            "playable_nonland_before_land": [a.label() for a in before_nonland],
            "playable_nonland_after_land": [a.label() for a in after_nonland],
            "priority_draw_available_before_land": before_draw,
            "land_was_needed_immediately": land_needed_now,
            "land_enabled_priority_action": land_enabled_priority,
            "land_entered_tapped": bool(getattr(card, "enters_tapped", False)),
        }

        history.append(event)
        return replace(
            state,
            hand=remaining_hand,
            battlefield=new_battlefield,
            history=tuple(history),
            land_played_this_turn=True,
            debug_last_event=event,
        )

    if action.kind == "cast":
        card = get_card_spec(action.card_name)
        indexes = choose_lands_for_card(card, battlefield)
        if indexes is None:
            return state

        payment = _payment_details(indexes, battlefield)
        tapped_battlefield = tap_payment_lands(battlefield, indexes)

        sacrificed_name = None
        temp_battlefield = list(tapped_battlefield)

        if getattr(card, "requires_sacrifice", False):
            sacrifice = choose_sacrifice(temp_battlefield)
            if sacrifice is None:
                return state
            sacrificed_name = sacrifice.spec.name
            removed = False
            kept = []
            for perm in temp_battlefield:
                if not removed and perm.spec.name == sacrificed_name:
                    graveyard.append(perm.spec)
                    removed = True
                else:
                    kept.append(perm)
            temp_battlefield = kept

        _, remaining_hand = _remove_first_card(hand, card.name)

        shamans_before = _shaman_count(state)
        if card.is_permanent:
            temp_battlefield.append(Permanent(card, tapped=False, entered_this_turn=True))
        else:
            graveyard.append(card)

        created_token = (
            getattr(card, "creates_token_on_cast", None)
            or getattr(card, "creates_token_on_resolve", None)
        )

        if getattr(card, "creates_token_on_cast", None):
            temp_battlefield.append(
                Permanent(token_spec(card.creates_token_on_cast), tapped=False, entered_this_turn=True)
            )

        event = {
            "kind": "cast",
            "name": card.name,
            **payment,
            "total_mana_spent": len(indexes),
            "effective_cost": effective_total_cost(card, battlefield),
            "sacrificed": sacrificed_name,
            "discarded": None,
            "draw_count": (
                getattr(card, "resolve_draw_cards", 0)
                + card.etb_draw_cards
                + (1 if sacrificed_name == "Ichor Wellspring" else 0)
            ),
            "created_token": created_token,
            "shaman_redundant_cast": card.name == "Krark-Clan Shaman" and shamans_before >= 1,
        }

        history.append(event)
        return replace(
            state,
            hand=remaining_hand,
            battlefield=tuple(temp_battlefield),
            graveyard=tuple(graveyard),
            history=tuple(history),
            mana_spent_this_turn=state.mana_spent_this_turn + len(indexes),
            tempo_loss_this_turn=state.tempo_loss_this_turn + (
                0 if getattr(card, "resolve_draw_cards", 0) or card.etb_draw_cards else 1
            ),
            debug_last_event=event,
        )

    if action.kind == "activate":
        permanent, _ = _remove_first_permanent(list(battlefield), action.card_name)

        ability = permanent.spec.activated_ability
        if ability is None:
            return state

        indexes = choose_lands_for_payment(
            ability.total_mana_cost,
            ability.color_pips,
            battlefield,
        )
        if indexes is None:
            return state

        payment = _payment_details(indexes, battlefield)
        tapped_battlefield = list(tap_payment_lands(battlefield, indexes))

        discarded_name = None
        new_hand = list(state.hand)
        if getattr(ability, "discard_cards", 0) > 0:
            discard = _choose_discard(state)
            if discard is None:
                return state
            removed = False
            kept_hand = []
            for card in new_hand:
                if not removed and card.name == discard.name:
                    graveyard.append(card)
                    discarded_name = card.name
                    removed = True
                else:
                    kept_hand.append(card)
            if not removed:
                return state
            new_hand = kept_hand

        if ability.sacrifice_self:
            removed = False
            kept = []
            for perm in tapped_battlefield:
                if not removed and perm.spec.name == permanent.spec.name:
                    graveyard.append(perm.spec)
                    removed = True
                else:
                    kept.append(perm)
            tapped_battlefield = kept

        event = {
            "kind": "activate",
            "name": permanent.spec.name,
            **payment,
            "total_mana_spent": len(indexes),
            "effective_cost": ability.total_mana_cost,
            "sacrificed": permanent.spec.name if ability.sacrifice_self else None,
            "discarded": discarded_name,
            "draw_count": ability.draw_cards + permanent.spec.ltb_draw_cards,
            "created_token": None,
        }

        history.append(event)
        return replace(
            state,
            hand=tuple(new_hand),
            battlefield=tuple(tapped_battlefield),
            graveyard=tuple(graveyard),
            history=tuple(history),
            mana_spent_this_turn=state.mana_spent_this_turn + len(indexes),
            debug_last_event=event,
        )

    return state


# ----------------------------------------------------------
# SCORING
# ----------------------------------------------------------

def score_state(state: PlanningState) -> float:
    score = 0.0

    lands = _land_count(state)
    lands_in_hand = sum(1 for c in state.hand if c.is_land)
    artifacts = sum(1 for p in state.battlefield if p.spec.is_artifact)
    nonland_fodder = sum(1 for p in state.battlefield if (not p.spec.is_land and p.spec.sac_fodder))

    score += min(lands, 3) * 7.0
    if lands >= 4:
        score += 4.0 + max(0, lands - 4) * 1.0
    score += min(lands_in_hand, max(0, 4 - lands)) * 3.5

    score += artifacts * 10.0
    score += nonland_fodder * 1.5

    bridge_count = sum(1 for p in state.battlefield if _is_bridge(p.spec))

    if lands == 1:
        score += bridge_count * 50.0
    elif lands == 2:
        score += bridge_count * 40.0

    # Small penalty if we are still in the first 2 land drops and no bridge is present
    if lands <= 2 and bridge_count == 0:
        score -= 5.0

    for card in state.hand:
        if card.is_land:
            continue
        if card.name == "Galvanic Blast":
            score -= 50.0
            continue
        if card.is_creature:
            eff = effective_total_cost(card, state.battlefield)
            if eff > 2:
                score -= 2.0
        if card.name == "Krark-Clan Shaman" and _shaman_count(state) >= 1:
            score -= 6.0
        score += _draw_priority_bonus(card)

    for permanent in state.battlefield:
        if permanent.spec.name in {"Blood Token", "Clue Token", "Hero Token"}:
            score += 15.0
        if permanent.spec.name in {"Ichor Wellspring", "Cryogen Relic"}:
            score += 4.0
        if permanent.spec.name == "Krark-Clan Shaman":
            score += 1.0
        elif permanent.spec.is_creature:
            score += 1.5

    shaman_count = _shaman_count(state)
    if shaman_count >= 2:
        score -= 14.0 * (shaman_count - 1)

    available_untapped = sum(1 for p in state.battlefield if p.spec.is_land and not p.tapped)
    score -= max(0, available_untapped - 1) * 0.75

    if state.history:
        last = state.history[-1]

        if last.get("kind") == "play_land":
            if last.get("priority_draw_available_before_land"):
                if last.get("land_entered_tapped"):
                    score += 3.0
                else:
                    score -= 8.0
            elif last.get("land_enabled_priority_action"):
                score += 5.0
            elif last.get("land_was_needed_immediately"):
                score += 2.0

        if last.get("sacrificed") in {"Clue Token", "Map Token", "Ichor Wellspring", "Cryogen Relic"}:
            score += 3.0

        if last.get("sacrificed") == "Blood Token":
            score += 3.0
            if lands <= 3:
                score -= 80.0

        if last.get("sacrificed") and (
            "Bridge" in last["sacrificed"]
            or last["sacrificed"] in {
                "Swamp",
                "Island",
                "Mountain",
                "Vault of Whispers",
                "Seat of the Synod",
                "Great Furnace",
            }
        ):
            score -= 25.0

        if last.get("discarded"):
            discarded = last["discarded"]
            if discarded in {
                "Swamp",
                "Island",
                "Mountain",
                "Vault of Whispers",
                "Seat of the Synod",
                "Great Furnace",
            } or "Bridge" in discarded:
                if lands_in_hand <= 1:
                    score -= 15.0
                else:
                    score += 1.0
            if discarded in {"Galvanic Blast", "Makeshift Munitions"}:
                score -= 50.0

        # Search may use an expected draw value instead of revealing exact future cards.
        if "expected_draw_value" in last:
            score += float(last.get("expected_draw_value", 0.0))
            score += last.get("draw_count", 0) * 1.25
        else:
            score += last.get("draw_count", 0) * 4.0

        if last.get("created_token") in {"Blood Token", "Clue Token", "Map Token", "Hero Token"}:
            score += 2.0

        if last.get("shaman_redundant_cast"):
            score -= 10.0

    if lands >= 2 and lands_in_hand > 0:
        score += min(lands_in_hand, 2) * 1.5

    return score
