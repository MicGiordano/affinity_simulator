from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .affinity_catalog import get_card_spec, token_spec
from .metrics import GameMetrics, evaluate_turn
from .mulligan import keepable_opening_hand, sort_for_bottom
from .mana import effective_total_cost
from .planner_hidden_info import PlanningState, apply_action, generate_actions, score_state
from .models import Deck, Permanent


@dataclass
class GameResult:
    metrics: GameMetrics
    log: dict
    mulligans_taken: int


def _initial_state(hand: Sequence):
    return PlanningState(tuple(hand), tuple(), tuple(), (), 0, 0, False)


def _lookup_spec(name: str):
    try:
        return get_card_spec(name)
    except KeyError:
        return token_spec(name)


def _draw_cards(library: list, amount: int) -> list[str]:
    drawn = []
    for _ in range(amount):
        if library:
            drawn.append(library.pop(0).name)
    return drawn


def _bf_snapshot(perms: Iterable[Permanent]) -> list[dict]:
    return [
        {
            "name": p.spec.name,
            "tapped": p.tapped,
            "entered_this_turn": p.entered_this_turn,
        }
        for p in perms
    ]


def _clear_summoning_flags(state: PlanningState) -> PlanningState:
    return PlanningState(
        hand=state.hand,
        battlefield=tuple(Permanent(p.spec, p.tapped, False) for p in state.battlefield),
        graveyard=state.graveyard,
        history=state.history,
        mana_spent_this_turn=state.mana_spent_this_turn,
        tempo_loss_this_turn=state.tempo_loss_this_turn,
        land_played_this_turn=state.land_played_this_turn,
        debug_last_event=state.debug_last_event,
    )


def _untap_lands_only(state: PlanningState) -> PlanningState:
    return PlanningState(
        hand=state.hand,
        battlefield=tuple(
            Permanent(p.spec, False, p.entered_this_turn)
            if p.spec.is_land
            else Permanent(p.spec, p.tapped, p.entered_this_turn)
            for p in state.battlefield
        ),
        graveyard=state.graveyard,
        history=state.history,
        mana_spent_this_turn=0,
        tempo_loss_this_turn=0,
        land_played_this_turn=False,
        debug_last_event=state.debug_last_event,
    )


def _with_new_hand(state: PlanningState, drawn_names: list[str]) -> PlanningState:
    if not drawn_names:
        return state
    return PlanningState(
        hand=state.hand + tuple(get_card_spec(name) for name in drawn_names),
        battlefield=state.battlefield,
        graveyard=state.graveyard,
        history=state.history,
        mana_spent_this_turn=state.mana_spent_this_turn,
        tempo_loss_this_turn=state.tempo_loss_this_turn,
        land_played_this_turn=state.land_played_this_turn,
        debug_last_event=state.debug_last_event,
    )


def _with_added_token(state: PlanningState, token_name: str) -> PlanningState:
    return PlanningState(
        hand=state.hand,
        battlefield=state.battlefield
        + (Permanent(token_spec(token_name), tapped=False, entered_this_turn=True),),
        graveyard=state.graveyard,
        history=state.history,
        mana_spent_this_turn=state.mana_spent_this_turn,
        tempo_loss_this_turn=state.tempo_loss_this_turn,
        land_played_this_turn=state.land_played_this_turn,
        debug_last_event=state.debug_last_event,
    )


def _resolve_post_action_effects(
    state: PlanningState,
    library: list,
    event: dict,
) -> tuple[PlanningState, dict]:
    """
    Real effects for the ACTUAL chosen line.
    This is the only place where concrete cards are drawn.
    """
    changed_state = state
    cards_drawn: list[str] = []
    created_tokens_resolved: list[str] = []

    kind = event.get("kind")
    name = event.get("name")

    if kind == "cast":
        card = get_card_spec(name)

        token_on_resolve = getattr(card, "creates_token_on_resolve", None)
        if token_on_resolve:
            changed_state = _with_added_token(changed_state, token_on_resolve)
            created_tokens_resolved.append(token_on_resolve)

        resolve_draws = getattr(card, "resolve_draw_cards", 0)
        if resolve_draws > 0:
            drawn = _draw_cards(library, resolve_draws)
            cards_drawn.extend(drawn)
            changed_state = _with_new_hand(changed_state, drawn)

        if card.etb_draw_cards > 0:
            drawn = _draw_cards(library, card.etb_draw_cards)
            cards_drawn.extend(drawn)
            changed_state = _with_new_hand(changed_state, drawn)

        sacrificed_name = event.get("sacrificed")
        if sacrificed_name:
            sacrificed_spec = _lookup_spec(sacrificed_name)
            if sacrificed_spec.ltb_draw_cards > 0:
                drawn = _draw_cards(library, sacrificed_spec.ltb_draw_cards)
                cards_drawn.extend(drawn)
                changed_state = _with_new_hand(changed_state, drawn)

    elif kind == "activate":
        spec = _lookup_spec(name)
        ability = spec.activated_ability

        if ability and ability.draw_cards > 0:
            drawn = _draw_cards(library, ability.draw_cards)
            cards_drawn.extend(drawn)
            changed_state = _with_new_hand(changed_state, drawn)

        sacrificed_name = event.get("sacrificed")
        if sacrificed_name:
            sacrificed_spec = _lookup_spec(sacrificed_name)
            if sacrificed_spec.ltb_draw_cards > 0:
                drawn = _draw_cards(library, sacrificed_spec.ltb_draw_cards)
                cards_drawn.extend(drawn)
                changed_state = _with_new_hand(changed_state, drawn)

    resolved_event = dict(event)
    resolved_event["cards_drawn"] = cards_drawn
    resolved_event["created_tokens_resolved"] = created_tokens_resolved
    return changed_state, resolved_event


def _card_expected_value(card, state_after_action: PlanningState) -> float:
    """
    Expected marginal value of drawing CARD.
    Uses remaining-library composition and visible state only; it does NOT depend on draw order.
    """
    lands_on_board = sum(1 for p in state_after_action.battlefield if p.spec.is_land)
    lands_in_hand = sum(1 for c in state_after_action.hand if c.is_land)
    artifacts_on_board = sum(1 for p in state_after_action.battlefield if p.spec.is_artifact)
    shaman_on_board = sum(1 for p in state_after_action.battlefield if p.spec.name == "Krark-Clan Shaman")

    value = 0.0

    if card.is_land:
        if lands_on_board + lands_in_hand < 4:
            value += 4.5
        else:
            value += 1.0
        if getattr(card, "enters_tapped", False) and lands_on_board >= 3:
            value -= 0.5
        if card.is_artifact:
            value += 1.0
        return value

    value += getattr(card, "resolve_draw_cards", 0) * 4.0
    value += getattr(card, "etb_draw_cards", 0) * 3.0
    value += getattr(card, "ltb_draw_cards", 0) * 1.5

    if card.is_artifact:
        value += 3.5
        if artifacts_on_board < 7:
            value += 1.0

    if getattr(card, "creates_token_on_cast", None) or getattr(card, "creates_token_on_resolve", None):
        value += 2.0

    if card.name == "Thoughtcast":
        value += 8.0
    if card.name == "Ichor Wellspring":
        value += 6.5
    if card.name in {"Reckoner's Bargain", "Fanatical Offering"}:
        value += 6.5
    if card.name == "Nihil Spellbomb":
        value += 2.0
    if card.name == "Galvanic Blast":
        value -= 6.0

    if card.is_creature:
        eff = effective_total_cost(card, state_after_action.battlefield)
        if eff <= max(1, lands_on_board):
            value += 2.0
        else:
            value -= 1.5

    if card.name == "Krark-Clan Shaman":
        value += 0.5
        if shaman_on_board >= 1:
            value -= 6.5

    return value


def _estimate_draw_value(library: list, draw_count: int, state_after_action: PlanningState) -> float:
    if draw_count <= 0 or not library:
        return 0.0
    values = [_card_expected_value(card, state_after_action) for card in library]
    return draw_count * (sum(values) / len(values))


def _resolve_post_action_effects_for_search(
    state: PlanningState,
    library: list,
    event: dict,
) -> tuple[PlanningState, dict]:
    """
    Search-time resolver with hidden information preserved.

    It applies deterministic visible effects (e.g. token creation on resolve), but it does NOT
    draw specific cards from the top of the library. Instead it assigns an expected draw value
    based on the remaining library composition.
    """
    changed_state = state
    created_tokens_resolved: list[str] = []
    expected_draw_value = 0.0

    kind = event.get("kind")
    name = event.get("name")

    if kind == "cast":
        card = get_card_spec(name)

        token_on_resolve = getattr(card, "creates_token_on_resolve", None)
        if token_on_resolve:
            changed_state = _with_added_token(changed_state, token_on_resolve)
            created_tokens_resolved.append(token_on_resolve)

        draw_count = getattr(card, "resolve_draw_cards", 0) + card.etb_draw_cards
        sacrificed_name = event.get("sacrificed")
        if sacrificed_name:
            sacrificed_spec = _lookup_spec(sacrificed_name)
            draw_count += getattr(sacrificed_spec, "ltb_draw_cards", 0)

        expected_draw_value = _estimate_draw_value(library, draw_count, changed_state)

    elif kind == "activate":
        spec = _lookup_spec(name)
        ability = spec.activated_ability
        draw_count = 0
        if ability:
            draw_count += getattr(ability, "draw_cards", 0)
        sacrificed_name = event.get("sacrificed")
        if sacrificed_name:
            sacrificed_spec = _lookup_spec(sacrificed_name)
            draw_count += getattr(sacrificed_spec, "ltb_draw_cards", 0)

        expected_draw_value = _estimate_draw_value(library, draw_count, changed_state)

    resolved_event = dict(event)
    resolved_event["cards_drawn"] = []
    resolved_event["created_tokens_resolved"] = created_tokens_resolved
    resolved_event["expected_draw_value"] = expected_draw_value
    return changed_state, resolved_event


def _search_best_line(
    state: PlanningState,
    library: list,
    depth: int,
) -> tuple[list[str], float]:
    if depth <= 0:
        return [], score_state(state)

    actions = generate_actions(state)
    if not actions:
        return [], score_state(state)

    best_line: list[str] = []
    best_score = float("-inf")

    for action in actions:
        next_state = apply_action(state, action)
        if not next_state.history:
            line_score = score_state(next_state)
            if line_score > best_score:
                best_score = line_score
                best_line = [action.label()]
            continue

        # IMPORTANT: the search does not reveal exact future cards.
        event = next_state.history[-1]
        resolved_state, _ = _resolve_post_action_effects_for_search(next_state, library, event)
        line, future_score = _search_best_line(resolved_state, library, depth - 1)
        score = score_state(resolved_state) + 0.8 * future_score
        if score > best_score:
            best_score = score
            best_line = [action.label()] + line

    return best_line, best_score


def _pick_best_action(
    state: PlanningState,
    library: list,
    depth: int,
):
    actions = generate_actions(state)
    if not actions:
        return None, [], score_state(state)

    best_action = None
    best_line: list[str] = []
    best_score = float("-inf")

    for action in actions:
        next_state = apply_action(state, action)
        if not next_state.history:
            score = score_state(next_state)
            if score > best_score:
                best_score = score
                best_action = action
                best_line = [action.label()]
            continue

        # IMPORTANT: the search does not reveal exact future cards.
        event = next_state.history[-1]
        resolved_state, _ = _resolve_post_action_effects_for_search(next_state, library, event)
        line, future_score = _search_best_line(resolved_state, library, depth - 1)
        score = score_state(resolved_state) + 0.8 * future_score
        if score > best_score:
            best_score = score
            best_action = action
            best_line = [action.label()] + line

    return best_action, best_line, best_score


def _begin_turn(state: PlanningState, library: list, turn: int, on_the_play: bool):
    state = _untap_lands_only(state)
    state = _clear_summoning_flags(state)

    draw_step: list[str] = []
    if turn > 1 or not on_the_play:
        draw_step = _draw_cards(library, 1)
        state = _with_new_hand(state, draw_step)

    return state, draw_step


def _play_turn(state: PlanningState, library: list, turn: int, depth: int):
    decisions = []

    while True:
        action, chosen_line, planner_score = _pick_best_action(state, library, depth)
        if action is None:
            break

        before_score = score_state(state)
        next_state = apply_action(state, action)
        if not next_state.history:
            break

        event = next_state.history[-1]
        resolved_state, resolved_event = _resolve_post_action_effects(
            next_state,
            library,
            event,
        )

        decisions.append(
            {
                "decision_index": len(decisions) + 1,
                "chosen_action": action.label(),
                "chosen_line": chosen_line,
                "planner_score": planner_score,
                "score_breakdown_before_action": {"total": before_score},
                "mana_payment": resolved_event.get("mana_payment", []),
                "colors_produced": resolved_event.get("colors_produced", []),
                "total_mana_spent": resolved_event.get("total_mana_spent", 0),
                "effective_cost": resolved_event.get("effective_cost", 0),
                "sacrificed": resolved_event.get("sacrificed"),
                "discarded": resolved_event.get("discarded"),
                "cards_drawn": resolved_event.get("cards_drawn", []),
                "created_token": resolved_event.get("created_token"),
                "created_tokens_resolved": resolved_event.get("created_tokens_resolved", []),
                "playable_nonland_before_land": resolved_event.get("playable_nonland_before_land", []),
                "playable_nonland_after_land": resolved_event.get("playable_nonland_after_land", []),
                "priority_draw_available_before_land": resolved_event.get("priority_draw_available_before_land"),
                "land_was_needed_immediately": resolved_event.get("land_was_needed_immediately"),
                "land_enabled_priority_action": resolved_event.get("land_enabled_priority_action"),
                "land_entered_tapped": resolved_event.get("land_entered_tapped"),
                "shaman_redundant_cast": resolved_event.get("shaman_redundant_cast", False),
            }
        )

        state = resolved_state
        if not generate_actions(state):
            break

    return state, decisions


def _take_mulligans(deck_cards: list, rng):
    cards = deck_cards[:]
    rng.shuffle(cards)
    mulligans = 0
    attempts = []

    while True:
        hand = cards[:7]
        keepable = keepable_opening_hand(hand)
        attempt = {
            "mulligans_taken_before_this_hand": mulligans,
            "drawn_seven": [c.name for c in hand],
            "kept": keepable,
            "bottomed_by_heuristic": [],
            "post_bottom_hand": [c.name for c in hand],
        }

        if keepable or mulligans >= 6:
            if mulligans:
                ordered = sort_for_bottom(hand)
                bottom = ordered[:mulligans]
                keep = hand[:]
                for card in bottom:
                    keep.remove(card)
                attempt["bottomed_by_heuristic"] = [c.name for c in bottom]
                attempt["post_bottom_hand"] = [c.name for c in keep]
                hand = keep

            attempts.append(attempt)
            final_hand = tuple(hand)
            failed_mulligan = not keepable_opening_hand(final_hand)
            library = cards[7:]
            return final_hand, library, mulligans, attempts, failed_mulligan

        attempts.append(attempt)
        mulligans += 1
        rng.shuffle(cards)


def simulate_game(
    deck: Deck,
    *,
    num_turns=5,
    seed=1,
    on_the_play=True,
    depth=3,
    collect_log=True,
    game_id=1,
):
    rng = random.Random(seed)
    opening_hand, library, mulligans_taken, mulligan_attempts, failed_mulligan = _take_mulligans(
        deck.cards,
        rng,
    )
    state = _initial_state(opening_hand)

    log = {
        "game_id": game_id,
        "seed": seed,
        "on_the_play": on_the_play,
        "opening_hand": [c.name for c in opening_hand],
        "mulligans_taken": mulligans_taken,
        "mulligan_attempts": mulligan_attempts,
        "kept_hand": [c.name for c in opening_hand],
        "failed_mulligan": failed_mulligan,
        "turns": [],
    }

    metrics = GameMetrics(failed_mulligan=failed_mulligan)

    for turn in range(1, num_turns + 1):
        state, draw_step = _begin_turn(state, library, turn, on_the_play)
        hand_start = tuple(state.hand)
        battlefield_start = tuple(state.battlefield)

        state, decisions = _play_turn(state, library, turn, depth)

        turn_metrics = evaluate_turn(
            turn=turn,
            state_end=state,
            hand_start=hand_start,
            battlefield_start=battlefield_start,
            failed_mulligan=failed_mulligan,
        )
        metrics.turns.append(turn_metrics)

        if collect_log:
            log["turns"].append(
                {
                    "turn": turn,
                    "draw_step": draw_step,
                    "hand_start": [c.name for c in hand_start],
                    "battlefield_start": _bf_snapshot(battlefield_start),
                    "decisions": decisions,
                    "hand_end": [c.name for c in state.hand],
                    "battlefield_end": _bf_snapshot(state.battlefield),
                    "graveyard_end": [c.name for c in state.graveyard],
                    "metrics": turn_metrics.metric_values(),
                }
            )

    return GameResult(metrics=metrics, log=log, mulligans_taken=mulligans_taken)
