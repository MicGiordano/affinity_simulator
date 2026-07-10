from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .affinity_catalog import get_card_spec
from .deckio import load_deck_from_text
from .metrics import METRIC_NAMES, aggregate_games, normalize_metrics, radar_polygon_area
from .simulator_hidden_info import simulate_game

DEFAULT_DECK_FILE = 'grixis-affinity-20260706-200124.txt'
DEFAULT_LAND_POOL = ('Drossforge Bridge', 'Mistvault Bridge', 'Silverbluff Bridge', 'Vault of Whispers', 'Seat of the Synod', 'Great Furnace', 'Swamp', 'Island', 'Mountain')
METRIC_LABELS = {m: m.replace('_', ' ').title() for m in METRIC_NAMES}
SAVE_LOG_FILES = False

def _root():
    return Path(__file__).resolve().parent


def _load_text(deck_text=None, deck_path=None):
    if deck_text is None:
        deck_text = Path(deck_path or _root() / DEFAULT_DECK_FILE).read_text()
    return deck_text.split('SIDEBOARD:')[0]


def _load_deck(deck_text=None, deck_path=None):
    return load_deck_from_text(_load_text(deck_text, deck_path))


def _quote(cards):
    return ', '.join(f'"{c}"' for c in cards) if cards else '(none)'


def _bf(perms):
    return ', '.join((p.get('name', '') + (' tapped' if p.get('tapped') else '')) for p in perms) if perms else '(empty)'


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def format_game_log_human(log: Mapping[str, Any]) -> str:
    lines = [
        "=" * 72,
        f'GAME {log.get("game_id")} | seed={log.get("seed")} | on_the_play={log.get("on_the_play")}',
        "=" * 72,
    ]

    lines += [
        f'Opening hand: {_quote(log.get("opening_hand", []))}',
        f'Mulligan: {"yes" if log.get("mulligans_taken") else "no"} ({log.get("mulligans_taken")})',
    ]

    for a in log.get("mulligan_attempts", []):
        bottom = a.get("bottomed_by_heuristic", [])
        lines += [
            f'  Attempt {a.get("mulligans_taken_before_this_hand")}: {"KEEP" if a.get("kept") else "MULLIGAN"}',
            f'    seven: {_quote(a.get("drawn_seven", []))}',
        ]
        if bottom:
            lines.append(f'    bottomed by heuristic: {_quote(bottom)}')
        lines.append(f'    post-bottom: {_quote(a.get("post_bottom_hand", []))}')

    lines += [
        f'Kept hand: {_quote(log.get("kept_hand", []))}',
        f'Failed mulligan: {log.get("failed_mulligan")}',
        "",
    ]

    for t in log.get("turns", []):
        lines += [
            f'T{t.get("turn")}',
            "-" * 72,
            f'Draw step: {_quote(t.get("draw_step", [])) if t.get("draw_step") else "no draw"}',
            f'Hand at start: {_quote(t.get("hand_start", []))}',
            f'Battlefield at start: {_bf(t.get("battlefield_start", []))}',
            "Actions / planner choices:",
        ]

        for d in t.get("decisions", []):
            lines.append(
                f'  {d.get("decision_index")}. {d.get("chosen_action")} [score={d.get("planner_score"):.3f}]'
            )
            lines.append("     planned line: " + " -> ".join(d.get("chosen_line", [])))

            bd = d.get("score_breakdown_before_action") or {}
            if bd:
                lines.append(f'     score breakdown: total={bd.get("total", 0):.2f}')

            lines.append(f'     mana payment: {_quote(d.get("mana_payment", []))}')
            lines.append(f'     colors produced: {_quote(d.get("colors_produced", []))}')
            lines.append(
                f'     total mana spent: {d.get("total_mana_spent", 0)} | effective cost: {d.get("effective_cost", 0)}'
            )

            if d.get("sacrificed"):
                lines.append(f'     sacrificed: {d.get("sacrificed")}')

            if d.get("discarded"):
                lines.append(f'     discarded: {d.get("discarded")}')

            if d.get("created_token"):
                lines.append(f'     created token (planned): {d.get("created_token")}')

            if d.get("created_tokens_resolved"):
                lines.append(
                    f'     created token (resolved): {_quote(d.get("created_tokens_resolved", []))}'
                )

            lines.append(f'     cards drawn: {_quote(d.get("cards_drawn", []))}')

        lines += [
            f'Hand at end: {_quote(t.get("hand_end", []))}',
            f'Battlefield at end: {_bf(t.get("battlefield_end", []))}',
            f'Graveyard at end: {_quote(t.get("graveyard_end", []))}',
            "End-of-turn metrics:",
        ]

        for m in METRIC_NAMES:
            lines.append(f'  - {m}: {float(t.get("metrics", {}).get(m, 0.0)):.4f}')

        lines.append("")

    return "\n".join(lines)


def simulate_one_game(*, deck_text=None, deck_path=None, output_dir='simulation_outputs/single_game', run_name='single_game', game_id=1, seed=1, num_turns=5, on_the_play=True, depth=3, save_json=True, save_text=True):
    game = simulate_game(_load_deck(deck_text, deck_path), num_turns=num_turns, seed=seed, on_the_play=on_the_play, depth=depth, collect_log=True, game_id=game_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    human = format_game_log_human(game.log)
    paths = {}
    stem = f'{run_name}_game_{game_id:06d}'
    if SAVE_LOG_FILES and save_json:
        p = out / f'{stem}.json'
        p.write_text(json.dumps(game.log, indent=2))
        paths['json'] = str(p)
    if SAVE_LOG_FILES and save_text:
        p = out / f'{stem}.txt'
        p.write_text(human)
        paths['text'] = str(p)
    return {'game_id': game_id, 'seed': seed, 'mulligans_taken': game.mulligans_taken, 'failed_mulligan': game.log.get('failed_mulligan'), 'metrics': game.metrics.averages(), 'log': game.log, 'human_log': human, 'paths': paths}


def _flags(log):
    ms = [t.get('metrics', {}) for t in log.get('turns', [])]
    return {'failed_mulligan': int(bool(log.get('failed_mulligan'))), 'any_screw_turn': int(any(m.get('low_mana_screw', 1) == 0 for m in ms)), 'any_flood_turn': int(any(m.get('low_mana_flood', 1) == 0 for m in ms))}


def simulate_games(*, games: int, deck_text=None, deck_path=None, output_dir='simulation_outputs', run_name='grixis_affinity', seed=1, num_turns=5, on_the_play=True, depth=3, save_individual_json=True, save_individual_text=True):
    deck = _load_deck(deck_text, deck_path)
    run = Path(output_dir) / run_name
    jd = run / 'games_json'
    td = run / 'games_text'
    jd.mkdir(parents=True, exist_ok=True)
    td.mkdir(parents=True, exist_ok=True)
    gms = []
    rows = []
    failures = {'failed_mulligans': 0, 'games_with_any_screw_turn': 0, 'games_with_any_flood_turn': 0}
    for i in range(games):
        g = simulate_game(deck, num_turns=num_turns, seed=seed + i, on_the_play=on_the_play, depth=depth, collect_log=True, game_id=i + 1)
        gms.append(g.metrics)
        fl = _flags(g.log)
        failures['failed_mulligans'] += fl['failed_mulligan']
        failures['games_with_any_screw_turn'] += fl['any_screw_turn']
        failures['games_with_any_flood_turn'] += fl['any_flood_turn']
        rows.append({'game_id': i + 1, 'seed': seed + i, 'mulligans_taken': g.mulligans_taken, **fl, **g.metrics.averages()})
        if save_individual_json:
            (jd / f'game_{i+1:06d}.json').write_text(json.dumps(g.log, indent=2))
        if save_individual_text:
            (td / f'game_{i+1:06d}.txt').write_text(format_game_log_human(g.log))
    agg = aggregate_games(gms)
    summary = {'run_name': run_name, 'games': games, 'seed_start': seed, 'seed_end': seed + games - 1, 'num_turns': num_turns, 'on_the_play': on_the_play, 'depth': depth, 'aggregate_metrics': agg, 'failure_counts': failures, 'failure_rates': {k: v / games if games else 0 for k, v in failures.items()}, 'paths': {'run_dir': str(run), 'games_json': str(jd), 'games_text': str(td), 'summary_json': str(run / 'run_summary.json'), 'summary_text': str(run / 'run_summary.txt'), 'aggregate_csv': str(run / 'aggregate_metrics.csv'), 'per_game_csv': str(run / 'per_game_metrics.csv')}}
    run.mkdir(parents=True, exist_ok=True)
    (run / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    (run / 'run_summary.txt').write_text(json.dumps(summary, indent=2))
    _write_csv(run / 'aggregate_metrics.csv', [{'run_name': run_name, **agg}], ['run_name', *METRIC_NAMES])
    _write_csv(run / 'per_game_metrics.csv', rows, ['game_id', 'seed', 'mulligans_taken', 'failed_mulligan', 'any_screw_turn', 'any_flood_turn', *METRIC_NAMES])
    return summary


def validate_system(run_simulation: bool = True):
    print('=' * 72)
    print('AFFINITY SIMULATOR – SYSTEM VALIDATION')
    print('=' * 72)
    print('\n[1] Checking module imports...')
    from .affinity_catalog import get_card_spec
    from .simulator_hidden_info import simulate_game
    from .planner import PlanningState
    from .metrics import METRIC_NAMES
    print('✅ All required imports OK')
    print('\n[2] Checking required symbols...')
    if not callable(get_card_spec) or not callable(simulate_game):
        raise RuntimeError('Missing required functions')
    print('✅ Required symbols OK')
    print('\n[3] Checking PlanningState structure...')
    required_fields = {'hand', 'battlefield', 'graveyard', 'history', 'mana_spent_this_turn', 'tempo_loss_this_turn'}
    state_fields = set(PlanningState.__dataclass_fields__.keys())
    missing_fields = required_fields - state_fields
    if missing_fields:
        raise RuntimeError(f'PlanningState missing fields: {missing_fields}')
    print('✅ PlanningState structure OK')
    print('\n[4] Checking metrics...')
    if 'board_presence' not in METRIC_NAMES:
        raise RuntimeError('board_presence metric missing')
    print('✅ Metrics OK (board_presence present)')
    if run_simulation:
        print('\n[5] Running simulation smoke test...')
        result = simulate_one_game(seed=1, num_turns=3)
        metrics = result.get('metrics', {})
        if not isinstance(metrics, dict):
            raise RuntimeError('Simulation did not return metrics dict')
        if 'board_presence' not in metrics:
            raise RuntimeError('Simulation output missing board_presence')
        print('✅ Simulation smoke test OK')
    print('\n' + '=' * 72)
    print('✅ SYSTEM VALIDATION PASSED')
    print('=' * 72)
