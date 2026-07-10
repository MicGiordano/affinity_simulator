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


def _is_bridge(card: CardSpec) -> bool:
    # Bridge lands are the artifact lands that enter tapped.
    return card.is_land and card.is_artifact and getattr(card, "enters_tapped", False)


def _estimated_turn_number(state: PlanningState) -> int:
    # Lightweight turn estimate from current board state.
    return _land_count(state) + 1


def _payment_details(indexes, battlefield: Tuple[Permanent, ...]) -> dict:
    idx = tuple(indexes)
    specs = [battlefield[i].spec for i in idx]
    return {
        "mana_payment": [spec.name for spec in specs],
        "colors_produced": [
            "/".join(spec.land_colors) if spec.land_colors else "C"
            for spec in specs
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

    # Explicit draw priorities
    if card.name == "Thoughtcast":
        bonus += 10.0
    if card.name == "Ichor Wellspring":
        bonus += 9.5
    if card.name in {"Reckoner's Bargain", "Fanatical Offering"}:
        bonus += 9.0
    if card.name in {"Blood Token", "Clue Token"}:
        bonus += 6.5

    # Generic draw weighting
    if getattr(card, "resolve_draw_cards", 0):
        bonus += 3.0 * getattr(card, "resolve_draw_cards", 0)
    if card.etb_draw_cards:
        bonus += 2.5 * card.etb_draw_cards
    if card.ltb_draw_cards:
        bonus += 1.5 * card.ltb_draw_cards

    return bonus


def _choose_discard(state: PlanningState) -> Optional[CardSpec]:
    """
    Blood-token discard heuristic:
    1) excess lands beyond expected land development
    2) weak / low-priority cards
    3) expensive cards that are not close to castable
    4) fallback least useful card
    """
    if not state.hand:
        return None

    lands_in_hand = [c for c in state.hand if c.is_land]
    nonlands_in_hand = [c for c in state.hand if not c.is_land]
    lands_in_play = _land_count(state)

    # Preserve near-future land drops; discard only truly excess lands
    desired_total_lands = min(4, lands_in_play + 2)
    excess_lands = len(lands_in_hand) - max(0, desired_total_lands - lands_in_play)
    if excess_lands > 0:
        # Prefer discarding a tapped bridge first if truly excess, otherwise any land
        bridge_lands = [c for c in lands_in_hand if _is_bridge(c)]
        if bridge_lands:
            return bridge_lands[-1]
        return lands_in_hand[-1]

    # Very low-priority spells first
    for name in ("Galvanic Blast", "Makeshift Munitions"):
        for card in state.hand:
            if card.name == name:
                return card

    # Expensive spells that are not close to being cast
    if nonlands_in_hand:
        expensive_candidates = [
            c
            for c in nonlands_in_hand
            if effective_total_cost(c, state.battlefield) > lands_in_play + 1
        ]
        if expensive_candidates:
            return max(expensive_candidates, key=lambda c: c.mana_value)

    # Keep your best card-draw / card-flow spells when possible
    protected = {"Thoughtcast", "Ichor Wellspring", "Reckoner's Bargain", "Fanatical Offering"}
    disposable = [c for c in nonlands_in_hand if c.name not in protected]
    if disposable:
        return max(disposable, key=lambda c: (c.mana_value, c.name))

    if nonlands_in_hand:
        return max(nonlands_in_hand, key=lambda c: (c.mana_value, c.name))

    return state.hand[0]


def _should_use_blood(state: PlanningState) -> bool:
    """
    Blood should mostly be used to:
    - hit land drops
    - smooth flood
    - fix very clunky hands
    not merely because 1 mana is available.
    """
    if not state.hand:
        return False

    lands_in_hand = sum(1 for c in state.hand if c.is_land)
    lands_in_play = _land_count(state)
    turn_estimate = _estimated_turn_number(state)

    discard = _choose_discard(state)
    if discard is None:
        return False

    # Strongest case: risk of missing early land drops
    if turn_estimate <= 4 and lands_in_hand == 0 and lands_in_play < turn_estimate:
        return True

    # Flood smoothing
    if lands_in_hand >= 4:
        return True

    # Clunky hand with no good plays
    playable_nonlands = [
        c for c in state.hand
        if not c.is_land and can_pay_with_lands(c, state.battlefield)
    ]
    if not playable_nonlands and lands_in_hand <= 1:
        return True

    # If the discard is a clearly low-value card, allow smoothing
    if discard.name in {"Galvanic Blast", "Makeshift Munitions"}:
        return True

    return False


def _play_land_priority_key(card: CardSpec, state: PlanningState):
    """
    Lower key = higher priority.
    Early turns: prefer bridges when legal/possible.
    Later turns: untapped single-color lands can be more flexible.
    """
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


# ----------------------------------------------------------
# ACTION GENERATION
# ----------------------------------------------------------

def generate_actions(state: PlanningState) -> Tuple[Action, ...]:
    actions = []

    # 1) Land play
    if not state.land_played_this_turn:
        lands = [c for c in state.hand if c.is_land]
        lands.sort(key=lambda c: _play_land_priority_key(c, state))
        for card in lands:
            actions.append(Action("play_land", card.name))

    # 2) Cast spells
    for card in state.hand:
        if card.is_land:
            continue

        # Explicitly keep Galvanic Blast disabled
        if card.name == "Galvanic Blast":
            continue

        if not can_pay_with_lands(card, state.battlefield):
            continue

        if getattr(card, "requires_creature", False) and _creature_count(state) <= 0:
            continue

        if getattr(card, "requires_sacrifice", False) and choose_sacrifice(state.battlefield) is None:
            continue

        actions.append(Action("cast", card.name))

    # 3) Activated abilities
    for permanent in state.battlefield:
        ability = permanent.spec.activated_ability
        if ability is None:
            continue

        if not can_pay_ability(ability.total_mana_cost, ability.color_pips, state.battlefield):
            continue

        # Blood token: only use if it improves the hand / land development
        if permanent.spec.name == "Blood Token":
            if getattr(ability, "discard_cards", 0) > 0 and len(state.hand) == 0:
                continue
            if not _should_use_blood(state):
                continue

        # Any ability requiring discard needs something to discard
        if getattr(ability, "discard_cards", 0) > 0 and len(state.hand) == 0:
            continue

        actions.append(Action("activate", permanent.spec.name))

    return tuple(actions)


# ----------------------------------------------------------
# APPLY ACTION
# ----------------------------------------------------------

def apply_action(state: PlanningState, action: Action) -> PlanningState:
    hand = list(state.hand)
    battlefield = tuple(state.battlefield)
    graveyard = list(state.graveyard)
    history = list(state.history)

    # -----------------------
    # PLAY LAND
    # -----------------------
    if action.kind == "play_land":
        card, remaining_hand = _remove_first_card(hand, action.card_name)

        # IMPORTANT: honor enters_tapped for bridges
        new_battlefield = battlefield + (
            Permanent(
                card,
                tapped=getattr(card, "enters_tapped", False),
                entered_this_turn=False,
            ),
        )

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

    # -----------------------
    # CAST SPELL
    # -----------------------
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

        if card.is_permanent:
            temp_battlefield.append(
                Permanent(card, tapped=False, entered_this_turn=True)
            )
        else:
            graveyard.append(card)

        created_token = (
            getattr(card, "creates_token_on_cast", None)
            or getattr(card, "creates_token_on_resolve", None)
        )

        if getattr(card, "creates_token_on_cast", None):
            temp_battlefield.append(
                Permanent(
                    token_spec(card.creates_token_on_cast),
                    tapped=False,
                    entered_this_turn=True,
                )
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

    # -----------------------
    # ACTIVATE ABILITY
    # -----------------------
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

        # Tap lands first
        tapped_battlefield = list(tap_payment_lands(battlefield, indexes))

        # Then discard cost (Blood token etc.)
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

        # Then sacrifice the activating permanent if required
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

    lands = sum(1 for p in state.battlefield if p.spec.is_land)
    artifacts = sum(1 for p in state.battlefield if p.spec.is_artifact)
    nonland_fodder = sum(
        1 for p in state.battlefield
        if (not p.spec.is_land and p.spec.sac_fodder)
    )

    # ------------------------------------------------------------------
    # Core resource development
    # ------------------------------------------------------------------
    score += lands * 8.0

    # Increased artifact weight to de-incentivize Blood-token churn
    score += artifacts * 15.0

    score += nonland_fodder * 5.0

    # ------------------------------------------------------------------
    # Strong early incentive for bridges (tapped artifact lands)
    # ------------------------------------------------------------------
    #bridge_count = sum(1 for p in state.battlefield if _is_bridge(p.spec))

    #if lands == 1:
    #    score += bridge_count * 50.0
    #elif lands == 2:
    #    score += bridge_count * 40.0

    # Small penalty if we are still in the first 2 land drops and no bridge is present
    #if lands <= 2 and bridge_count == 0:
    #    score -= 5.0

    # ------------------------------------------------------------------
    # Hand priorities
    # ------------------------------------------------------------------
    for card in state.hand:
        if card.is_land:
            continue

        # Galvanic Blast intentionally heavily de-prioritized
        if card.name == "Galvanic Blast":
            score -= 50.0
            continue

        # Delay expensive creatures until they are actually cheap
        if card.is_creature:
            eff = effective_total_cost(card, state.battlefield)
            if eff > 2:
                score -= 2.0

        score += _draw_priority_bonus(card)

    # ------------------------------------------------------------------
    # Battlefield value
    # ------------------------------------------------------------------
    for permanent in state.battlefield:
        # Keeping artifact tokens on board is valuable
        if permanent.spec.name in {"Blood Token", "Clue Token", "Hero Token"}:
            score += 15.0

        if permanent.spec.name in {"Ichor Wellspring", "Cryogen Relic"}:
            score += 8.0

        if permanent.spec.is_creature and permanent.spec.name != "Krark-Clan Shaman":
            score += 1.5

    # ------------------------------------------------------------------
    # History-aware scoring
    # ------------------------------------------------------------------
    if state.history:
        last = state.history[-1]

        # Reward sacrificing the "correct" disposable / delayed-value permanents
        if last.get("sacrificed") in {"Clue Token", "Map Token", "Ichor Wellspring", "Cryogen Relic"}:
            score += 3.0

        # Blood token: strongly discourage sacrificing it, especially early
        if last.get("sacrificed") == "Blood Token":
            score += 3.0

            # Sacrificing Blood early is EVEN WORSE because early turns should be about
            # board/artifact development, not hand cycling.
            if lands <= 3:
                score -= 60.0

        # Never sacrifice lands unless absolutely unavoidable
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
            score -= 20.0

        # Discard handling
        if last.get("discarded"):
            discarded = last["discarded"]

            # Discarding lands is okay only if they are truly excess
            if discarded in {
                "Swamp",
                "Island",
                "Mountain",
                "Vault of Whispers",
                "Seat of the Synod",
                "Great Furnace",
            } or "Bridge" in discarded:
                lands_in_hand = sum(1 for c in state.hand if c.is_land)
                if lands_in_hand <= 1:
                    score -= 15.0
                else:
                    score += 1.0

            # Discarding weak cards is slightly rewarded
            if discarded in {"Galvanic Blast", "Makeshift Munitions"}:
                score -= 50

        # Card draw still matters, but should not dominate board development
        score += last.get("draw_count", 0) * 4.0

        # Token creation remains useful
        if last.get("created_token") in {"Blood Token", "Clue Token", "Map Token", "Hero Token"}:
            score += 2.0

    return score