from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import pi, sin
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from .affinity_catalog import CardSpec
from .mana import effective_total_cost
from .models import Permanent
from .planner import PlanningState


# NOTE:
# All metrics are returned as bounded radar scores in [0, 1],
# where HIGHER = BETTER, even if the metric name describes an
# underlying risk/penalty quantity (e.g. wasted_mana, flood).


METRIC_NAMES = (
    "land_development_strength",
    "all_three_colors_available",
    "needed_colors_met",
    "earliest_all_three_colors_turn",
    "artifact_count",
    "nonland_sacrificable_fodder",
    "wasted_mana",
    "tapland_tempo_loss",
    "low_mana_screw",
    "flood",
    "mulligan_fail_rate",
    "board_presence",
)


@dataclass(frozen=True)
class TurnMetrics:
    turn: int
    land_development_strength: float
    all_three_colors_available: float
    needed_colors_met: float
    artifact_count: float
    nonland_sacrificable_fodder: float
    wasted_mana: float
    tapland_tempo_loss: float
    low_mana_screw: float
    flood: float
    board_presence: float

    def metric_values(self) -> Dict[str, float]:
        values = asdict(self)
        values.pop("turn")
        return values


@dataclass
class GameMetrics:
    turns: List[TurnMetrics] = field(default_factory=list)
    failed_mulligan: bool = False

    def _earliest_all_three_colors_score(self) -> float:
        if not self.turns:
            return 0.0

        max_turns = max(t.turn for t in self.turns)
        first_turn = next(
            (t.turn for t in self.turns if t.all_three_colors_available >= 1.0),
            max_turns + 1,
        )

        # T=1 -> 1.0
        # T=max_turns+1 (never reached) -> 0.0
        score = 1.0 - ((first_turn - 1) / max(1, max_turns))
        return _clamp01(score)

    def _mulligan_fail_rate_score(self) -> float:
        # Current simulator plumbing still passes failed_mulligan=False.
        # Once simulator is patched, this becomes a real game-level signal.
        return 0.0 if self.failed_mulligan else 1.0

    def averages(self) -> Dict[str, float]:
        if not self.turns:
            return {m: 0.0 for m in METRIC_NAMES}

        turn_metric_names = (
            "land_development_strength",
            "all_three_colors_available",
            "needed_colors_met",
            "artifact_count",
            "nonland_sacrificable_fodder",
            "wasted_mana",
            "tapland_tempo_loss",
            "low_mana_screw",
            "flood",
            "board_presence",
        )

        out = {
            m: mean(getattr(t, m) for t in self.turns)
            for m in turn_metric_names
        }

        out["earliest_all_three_colors_turn"] = self._earliest_all_three_colors_score()
        out["mulligan_fail_rate"] = self._mulligan_fail_rate_score()
        return out


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _colors(perms: Iterable[Permanent]) -> set[str]:
    colors = set()
    for permanent in perms:
        if permanent.spec.is_land:
            colors.update(permanent.spec.land_colors)
    return colors


def _all_lands(perms: Iterable[Permanent]) -> tuple[Permanent, ...]:
    return tuple(p for p in perms if p.spec.is_land)


def _untapped_lands(perms: Iterable[Permanent]) -> tuple[Permanent, ...]:
    return tuple(p for p in perms if p.spec.is_land and not p.tapped)


def _cycle_mana(state: PlanningState) -> int:
    # Same intent as the previous implementation:
    # mana already spent this turn + untapped lands still available.
    return state.mana_spent_this_turn + sum(
        1 for p in state.battlefield if p.spec.is_land and not p.tapped
    )


def _can_cover_color_pips_with_lands(
    color_pips: Dict[str, int],
    lands: Sequence[Permanent],
) -> bool:
    """
    Checks only COLOR availability, ignoring tapped state and generic costs.
    This is deliberate for 'needed_colors_met': we want a manabase/color
    availability metric, not a full current-sequencing metric.
    """
    if not color_pips:
        return True

    remaining = dict(color_pips)
    land_color_sets = [tuple(p.spec.land_colors) for p in lands if p.spec.is_land]

    # Greedy is enough here because all current cards use at most one colored pip.
    # Still implemented generally.
    for colors in land_color_sets:
        for c in colors:
            if remaining.get(c, 0) > 0:
                remaining[c] -= 1
                break

    return all(v <= 0 for v in remaining.values())


def _needed_colors_met(state: PlanningState) -> float:
    """
    Ratio of currently relevant hand spells whose COLOR requirements
    are met by the lands on the battlefield.
    A spell is "relevant" if its effective total cost is <= total mana
    available in this turn cycle.
    """
    total_mana = _cycle_mana(state)
    lands = _all_lands(state.battlefield)

    relevant_spells = [
        card
        for card in state.hand
        if not card.is_land and effective_total_cost(card, state.battlefield) <= total_mana
    ]

    if not relevant_spells:
        return 1.0

    met = 0
    for card in relevant_spells:
        if _can_cover_color_pips_with_lands(card.color_pips, lands):
            met += 1

    return _clamp01(met / len(relevant_spells))


def _artifact_count_score(state: PlanningState, turn: int) -> float:
    """
    Dynamic cap:
      artifact cap = 2 * turn
    This gives the metric room to grow without one early spike dominating
    the radar.
    """
    artifacts = sum(1 for p in state.battlefield if p.spec.is_artifact)
    cap = max(1, 2 * turn)
    return _clamp01(min(artifacts, cap) / cap)


def _nonland_sacrificable_fodder_score(state: PlanningState, turn: int) -> float:
    """
    Dynamic cap:
      fodder cap = turn
    This keeps fodder meaningful but prevents it from overwhelming the profile.
    """
    fodder = sum(
        1
        for p in state.battlefield
        if (not p.spec.is_land) and p.spec.sac_fodder
    )
    cap = max(1, turn)
    return _clamp01(min(fodder, cap) / cap)


def _wasted_mana_score(state: PlanningState) -> float:
    """
    Returns a HIGHER-IS-BETTER score:
      1 - (unused_mana / total_available_mana_this_turn)
    """
    total = _cycle_mana(state)
    spent = state.mana_spent_this_turn

    if total <= 0:
        return 1.0

    unused = max(0, total - spent)
    waste_ratio = unused / total
    return _clamp01(1.0 - waste_ratio)


def _tapland_tempo_loss_score(state: PlanningState, turn: int) -> float:
    """
    Keeps the existing logic shape:
      1 - normalized_tempo_loss
    """
    return _clamp01(1.0 - min(1.0, state.tempo_loss_this_turn / max(1, turn)))


def _flood_score(
    hand: Sequence[CardSpec],
    battlefield: Sequence[Permanent],
) -> float:
    """
    Proper flood definition:
      land_density * unusable_spells_ratio

    Returned as a radar score:
      1 - flood_raw
    """
    if not hand:
        return 1.0

    lands_in_hand = sum(1 for card in hand if card.is_land)
    spells_in_hand = [card for card in hand if not card.is_land]

    hand_size = len(hand)
    land_density = lands_in_hand / hand_size

    if not spells_in_hand:
        unusable_spells_ratio = 1.0 if lands_in_hand > 0 else 0.0
    else:
        # Here we keep the current notion of "playable" tied to the battlefield
        # state before the turn actions.
        uncastable = 0
        untapped_land_count = sum(
            1 for p in battlefield if p.spec.is_land and not p.tapped
        )

        for card in spells_in_hand:
            if effective_total_cost(card, battlefield) > untapped_land_count:
                uncastable += 1
                continue
            if not _can_cover_color_pips_with_lands(card.color_pips, _all_lands(battlefield)):
                uncastable += 1

        unusable_spells_ratio = uncastable / len(spells_in_hand)

    flood_raw = land_density * unusable_spells_ratio
    return _clamp01(1.0 - flood_raw)


def _board_presence(state_end: PlanningState, turn: int) -> float:
    # Keep your current definition.
    if turn < 3:
        return 0.0

    creatures = [
        p
        for p in state_end.battlefield
        if p.spec.is_creature and p.spec.name != "Krark-Clan Shaman"
    ]
    return _clamp01(min(len(creatures), 4) / 4)


def evaluate_turn(
    *,
    turn: int,
    state_end: PlanningState,
    hand_start: Sequence[CardSpec],
    battlefield_start: Sequence[Permanent],
    failed_mulligan: bool = False,  # kept for call compatibility
) -> TurnMetrics:
    lands = sum(1 for p in state_end.battlefield if p.spec.is_land)
    colors_now = _colors(state_end.battlefield)

    return TurnMetrics(
        turn=turn,
        land_development_strength=min(lands, turn) / max(1, turn),
        all_three_colors_available=1.0 if {"U", "B", "R"}.issubset(colors_now) else 0.0,
        needed_colors_met=_needed_colors_met(state_end),
        artifact_count=_artifact_count_score(state_end, turn),
        nonland_sacrificable_fodder=_nonland_sacrificable_fodder_score(state_end, turn),
        wasted_mana=_wasted_mana_score(state_end),
        tapland_tempo_loss=_tapland_tempo_loss_score(state_end, turn),
        low_mana_screw=1.0 - (1.0 if lands < turn else 0.0),
        flood=_flood_score(hand_start, battlefield_start),
        board_presence=_board_presence(state_end, turn),
    )


def aggregate_games(games) -> Dict[str, float]:
    if not games:
        return {m: 0.0 for m in METRIC_NAMES}

    rows = [game.averages() for game in games]
    return {m: mean(row[m] for row in rows) for m in METRIC_NAMES}


def normalize_metrics(rows):
    """
    Stable normalization for radar charts:
    metrics are already designed to live in [0,1], so this function simply
    clamps and rescales to [0,100] without depending on the comparison set.
    """
    if not rows:
        return []

    normalized = []
    for row in rows:
        normalized.append(
            {
                m: 100.0 * _clamp01(row.get(m, 0.0))
                for m in METRIC_NAMES
            }
        )
    return normalized


def radar_polygon_area(row, metric_order=METRIC_NAMES):
    radii = [row[m] for m in metric_order]
    n = len(radii)
    return 0.5 * sin(2 * pi / n) * sum(
        radii[i] * radii[(i + 1) % n] for i in range(n)
    )