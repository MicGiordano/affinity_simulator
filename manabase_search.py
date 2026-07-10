from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import pi
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt

try:
    from sim.api_hidden_info import simulate_games, DEFAULT_DECK_FILE
    from sim.metrics import METRIC_NAMES, normalize_metrics, radar_polygon_area
except ImportError:
    from api_hidden_info import simulate_games, DEFAULT_DECK_FILE
    from metrics import METRIC_NAMES, normalize_metrics, radar_polygon_area


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

N_GAMES = 750
NUM_TURNS = 5
SEED_START = 42
DEPTH = 4

TOTAL_LANDS = 19
MAX_NONBASIC_COPIES = 4
LIMIT_CONFIGS: int | None = None  # set to a small number for quick testing

# Conservative pruning hooks.
REQUIRE_ALL_THREE_COLORS = True
TOP_K_INCREMENTAL = 20
TOP_K_FINAL = 25

OUTPUT_ROOT = "simulation_outputs"
SEARCH_RUN_NAME = "manabase_search"

ARTIFACT_LANDS = (
    "Drossforge Bridge",
    "Mistvault Bridge",
    "Silverbluff Bridge",
    "Vault of Whispers",
    "Seat of the Synod",
    "Great Furnace",
)

BASIC_LANDS = ("Swamp", "Island", "Mountain")

# Requested restriction: basics only replace the mono-color artifact lands.
REPLACEMENT_BASICS = {
    "Vault of Whispers": ("Swamp",),
    "Seat of the Synod": ("Island",),
    "Great Furnace": ("Mountain",),
}

LAND_COLORS = {
    "Drossforge Bridge": {"B", "R"},
    "Mistvault Bridge": {"U", "B"},
    "Silverbluff Bridge": {"U", "R"},
    "Vault of Whispers": {"B"},
    "Seat of the Synod": {"U"},
    "Great Furnace": {"R"},
    "Swamp": {"B"},
    "Island": {"U"},
    "Mountain": {"R"},
}

METRIC_LABELS = {m: m.replace("_", " ").title() for m in METRIC_NAMES}


# ----------------------------------------------------------
# DATA MODELS
# ----------------------------------------------------------

@dataclass(frozen=True)
class ConfigResult:
    config_id: int
    label: str
    manabase: tuple[tuple[str, int], ...]
    raw_metrics: dict[str, float]
    normalized_metrics: dict[str, float]
    polygon_area: float
    run_dir_play: str
    run_dir_draw: str
    radar_png: str


# ----------------------------------------------------------
# DECK IO
# ----------------------------------------------------------

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _candidate_deck_paths() -> list[Path]:
    base = _script_dir()
    return [
        base / DEFAULT_DECK_FILE,
        base / "sim" / DEFAULT_DECK_FILE,
        base / "grixis-affinity-20260706-200124.txt",
        base / "sim" / "grixis-affinity-20260706-200124.txt",
    ]


def load_baseline_deck_text() -> str:
    for path in _candidate_deck_paths():
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(
        f"Could not find baseline deck file. Checked: {', '.join(str(p) for p in _candidate_deck_paths())}"
    )


def parse_deck_counts(deck_text: str) -> Counter:
    counts: Counter[str] = Counter()
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("SIDEBOARD"):
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        qty, name = parts
        counts[name.strip()] += int(qty)
    return counts


def deck_text_from_counts(counts: Counter) -> str:
    ordered_names = []

    # Nonlands first, then lands, for readability.
    for name in sorted(counts):
        if name not in ARTIFACT_LANDS and name not in BASIC_LANDS:
            ordered_names.append(name)
    for name in ARTIFACT_LANDS:
        if counts.get(name, 0) > 0:
            ordered_names.append(name)
    for name in BASIC_LANDS:
        if counts.get(name, 0) > 0:
            ordered_names.append(name)

    out_lines = []
    for name in ordered_names:
        qty = counts.get(name, 0)
        if qty > 0:
            out_lines.append(f"{qty} {name}")
    return "\n".join(out_lines) + "\n"


# ----------------------------------------------------------
# MANABASE GENERATION / PRUNING
# ----------------------------------------------------------

def _artifact_only_combos(total_lands: int = TOTAL_LANDS, max_copies: int = MAX_NONBASIC_COPIES):
    names = ARTIFACT_LANDS
    for counts in product(range(max_copies + 1), repeat=len(names)):
        if sum(counts) != total_lands:
            continue
        yield Counter({name: ct for name, ct in zip(names, counts) if ct > 0})


def _one_basic_replacements(base_config: Counter):
    out = []
    for land_name, current_count in list(base_config.items()):
        if current_count <= 0:
            continue
        for basic_name in REPLACEMENT_BASICS.get(land_name, ()):  # mono-color replacements only
            variant = Counter(base_config)
            variant[land_name] -= 1
            if variant[land_name] <= 0:
                del variant[land_name]
            variant[basic_name] += 1
            out.append(variant)
    return out


def _colors_of_manabase(manabase: Counter) -> set[str]:
    colors: set[str] = set()
    for land_name, qty in manabase.items():
        if qty > 0:
            colors.update(LAND_COLORS.get(land_name, set()))
    return colors


def _config_makes_sense(manabase: Counter) -> bool:
    # Conservative cut: for a Grixis deck, reject any manabase that cannot produce U/B/R at all.
    if REQUIRE_ALL_THREE_COLORS and not {"U", "B", "R"}.issubset(_colors_of_manabase(manabase)):
        return False
    return True


def generate_manabase_configs(
    total_lands: int = TOTAL_LANDS,
    max_copies: int = MAX_NONBASIC_COPIES,
    limit_configs: int | None = LIMIT_CONFIGS,
) -> list[Counter]:
    """
    Search space:
      1) all artifact-land-only configurations
      2) same configurations with one mono-color artifact land replaced by the matching basic
    Plus conservative pruning to remove color-invalid configurations.
    """
    unique: dict[tuple[tuple[str, int], ...], Counter] = {}

    for artifact_cfg in _artifact_only_combos(total_lands=total_lands, max_copies=max_copies):
        if _config_makes_sense(artifact_cfg):
            key = tuple(sorted((k, v) for k, v in artifact_cfg.items() if v > 0))
            unique[key] = Counter(artifact_cfg)

        for variant in _one_basic_replacements(artifact_cfg):
            if _config_makes_sense(variant):
                vkey = tuple(sorted((k, v) for k, v in variant.items() if v > 0))
                unique[vkey] = Counter(variant)

    configs = [unique[k] for k in sorted(unique)]
    if limit_configs is not None:
        configs = configs[:limit_configs]
    return configs


def count_search_space(
    total_lands: int = TOTAL_LANDS,
    max_copies: int = MAX_NONBASIC_COPIES,
) -> dict[str, int]:
    artifact_only = list(_artifact_only_combos(total_lands=total_lands, max_copies=max_copies))
    artifact_only_pruned = [cfg for cfg in artifact_only if _config_makes_sense(cfg)]

    unique_total: dict[tuple[tuple[str, int], ...], Counter] = {}
    for cfg in artifact_only:
        if _config_makes_sense(cfg):
            unique_total[tuple(sorted(cfg.items()))] = cfg
        for variant in _one_basic_replacements(cfg):
            if _config_makes_sense(variant):
                unique_total[tuple(sorted(variant.items()))] = variant

    return {
        "artifact_only_raw": len(artifact_only),
        "artifact_only_after_pruning": len(artifact_only_pruned),
        "total_unique_after_replacements_and_pruning": len(unique_total),
    }


# ----------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------

def _plot_radar(
    profile_a: dict[str, float],
    profile_b: dict[str, float] | None = None,
    label_a: str = "Run A",
    label_b: str = "Run B",
    title: str = "Deck Performance Radar",
    save_path: str | None = None,
) -> None:
    labels = list(METRIC_NAMES)
    values_a = [float(profile_a.get(m, 0.0)) for m in labels]
    values_a += values_a[:1]

    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], [METRIC_LABELS[m] for m in labels], fontsize=9)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], fontsize=8)
    plt.ylim(0, 100)

    ax.plot(angles, values_a, linewidth=2, linestyle="solid", label=label_a)
    ax.fill(angles, values_a, alpha=0.20)

    if profile_b is not None:
        values_b = [float(profile_b.get(m, 0.0)) for m in labels]
        values_b += values_b[:1]
        ax.plot(angles, values_b, linewidth=2, linestyle="solid", label=label_b)
        ax.fill(angles, values_b, alpha=0.12)

    plt.title(title, size=13, y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.20, 1.12))
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------
# RANKING / REPORTING
# ----------------------------------------------------------

def _format_manabase_label(manabase: Counter) -> str:
    parts = []
    for name in (*ARTIFACT_LANDS, *BASIC_LANDS):
        qty = manabase.get(name, 0)
        if qty > 0:
            parts.append(f"{name}x{qty}")
    text = "__".join(parts)
    return text.replace(" ", "-").replace("'", "")


def _build_deck_text_with_manabase(base_deck_counts: Counter, manabase: Counter) -> str:
    counts = Counter(base_deck_counts)
    for land_name in (*ARTIFACT_LANDS, *BASIC_LANDS):
        if land_name in counts:
            del counts[land_name]
    for land_name, qty in manabase.items():
        counts[land_name] += qty
    return deck_text_from_counts(counts)


def _save_manabase_txt(run_dir: Path, manabase: Counter) -> None:
    """
    Save the manabase as a readable txt file inside the config folder.
    """
    lines = []
    for land_name, qty in sorted(manabase.items()):
        lines.append(f"{qty} {land_name}")

    txt_path = run_dir / "lands.txt"
    txt_path.write_text("\n".join(lines) + "\n")



def _result_to_row(rank: int, r: ConfigResult) -> dict:
    row = {
        "rank": rank,
        "config_id": r.config_id,
        "label": r.label,
        "polygon_area": r.polygon_area,
        "run_dir_play": r.run_dir_play,
        "run_dir_draw": r.run_dir_draw,
        "radar_png": r.radar_png,
    }
    for land_name, qty in r.manabase:
        row[f"mb__{land_name}"] = qty
    for metric in METRIC_NAMES:
        row[f"metric__{metric}"] = r.raw_metrics.get(metric, 0.0)
        row[f"metric_norm__{metric}"] = r.normalized_metrics.get(metric, 0.0)
    return row


def _write_rows_csv_json(base_path: Path, rows: list[dict]) -> None:
    csv_path = base_path.with_suffix(".csv")
    json_path = base_path.with_suffix(".json")

    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        csv_path.write_text("")

    json_path.write_text(json.dumps(rows, indent=2))


def _save_incremental_rankings(search_dir: Path, results: list[ConfigResult], top_k: int = TOP_K_INCREMENTAL) -> None:
    ranked = sorted(results, key=lambda x: x.polygon_area, reverse=True)
    top_rows = [_result_to_row(i + 1, r) for i, r in enumerate(ranked[:top_k])]
    bottom_rows = [_result_to_row(i + 1, r) for i, r in enumerate(list(reversed(ranked[-top_k:])), start=0)]
    _write_rows_csv_json(search_dir / "incremental_top_rankings", top_rows)
    _write_rows_csv_json(search_dir / "incremental_bottom_rankings", bottom_rows)


# ----------------------------------------------------------
# SEARCH EXECUTION
# ----------------------------------------------------------

def run_manabase_search(
    n_games: int = N_GAMES,
    num_turns: int = NUM_TURNS,
    seed_start: int = SEED_START,
    depth: int = DEPTH,
    total_lands: int = TOTAL_LANDS,
    max_nonbasic_copies: int = MAX_NONBASIC_COPIES,
    limit_configs: int | None = LIMIT_CONFIGS,
) -> dict:
    search_dir = Path(OUTPUT_ROOT) / SEARCH_RUN_NAME
    search_dir.mkdir(parents=True, exist_ok=True)

    baseline_text = load_baseline_deck_text()
    base_counts = parse_deck_counts(baseline_text)
    configs = generate_manabase_configs(
        total_lands=total_lands,
        max_copies=max_nonbasic_copies,
        limit_configs=limit_configs,
    )

    print(f"Generated {len(configs)} manabase configurations")
    print(json.dumps(count_search_space(total_lands=total_lands, max_copies=max_nonbasic_copies), indent=2))

    results: list[ConfigResult] = []

    for idx, manabase in enumerate(configs, start=1):
        label = _format_manabase_label(manabase)
        deck_text = _build_deck_text_with_manabase(base_counts, manabase)
        run_name_play = f"{SEARCH_RUN_NAME}/config_{idx:05d}_play"
        run_name_draw = f"{SEARCH_RUN_NAME}/config_{idx:05d}_draw"

        print(f"[{idx}/{len(configs)}] Simulating {label} (play + draw)")

        summary_play = simulate_games(
            games=n_games,
            deck_text=deck_text,
            output_dir=OUTPUT_ROOT,
            run_name=run_name_play,
            seed=seed_start,
            num_turns=num_turns,
            on_the_play=True,
            depth=depth,
            save_individual_json=False,
            save_individual_text=False,
        )

        summary_draw = simulate_games(
            games=n_games,
            deck_text=deck_text,
            output_dir=OUTPUT_ROOT,
            run_name=run_name_draw,
            seed=seed_start + n_games,
            num_turns=num_turns,
            on_the_play=False,
            depth=depth,
            save_individual_json=False,
            save_individual_text=False,
        )

        raw_play = dict(summary_play.get("aggregate_metrics", {}))
        raw_draw = dict(summary_draw.get("aggregate_metrics", {}))
        raw_profile = {
            m: (raw_play.get(m, 0.0) + raw_draw.get(m, 0.0)) / 2.0
            for m in METRIC_NAMES
        }

        normalized_profile = normalize_metrics([raw_profile])[0]
        polygon_area = radar_polygon_area(normalized_profile)

        run_dir_play = Path(OUTPUT_ROOT) / run_name_play
        run_dir_draw = Path(OUTPUT_ROOT) / run_name_draw
        _save_manabase_txt(run_dir_play, manabase)
        radar_path = search_dir / f"config_{idx:05d}_radar.png"
        _plot_radar(
            normalized_profile,
            label_a=f"Config {idx}",
            title=f"Manabase Radar – Config {idx}",
            save_path=str(radar_path),
        )

        result = ConfigResult(
            config_id=idx,
            label=label,
            manabase=tuple(sorted((k, v) for k, v in manabase.items() if v > 0)),
            raw_metrics=raw_profile,
            normalized_metrics=normalized_profile,
            polygon_area=polygon_area,
            run_dir_play=str(run_dir_play),
            run_dir_draw=str(run_dir_draw),
            radar_png=str(radar_path),
        )
        results.append(result)

        # Incremental ranking snapshots after every evaluated configuration.
        _save_incremental_rankings(search_dir, results, top_k=TOP_K_INCREMENTAL)

    if not results:
        raise RuntimeError("No manabase configurations were generated.")

    ranked_results = sorted(results, key=lambda x: x.polygon_area, reverse=True)
    best = ranked_results[0]
    worst = ranked_results[-1]

    comparison_png = search_dir / "best_vs_worst_radar.png"
    _plot_radar(
        best.normalized_metrics,
        profile_b=worst.normalized_metrics,
        label_a=f"Best #{best.config_id}",
        label_b=f"Worst #{worst.config_id}",
        title="Best vs Worst Manabase",
        save_path=str(comparison_png),
    )

    all_rows = [_result_to_row(i + 1, r) for i, r in enumerate(ranked_results)]
    top_rows = all_rows[:TOP_K_FINAL]
    bottom_rows = list(reversed(all_rows[-TOP_K_FINAL:]))
    _write_rows_csv_json(search_dir / "all_configurations_ranked", all_rows)
    _write_rows_csv_json(search_dir / "top_configurations", top_rows)
    _write_rows_csv_json(search_dir / "bottom_configurations", bottom_rows)

    summary_payload = {
        "search_dir": str(search_dir),
        "configs_tested": len(ranked_results),
        "games_per_side_per_config": n_games,
        "total_simulated_games": len(ranked_results) * n_games * 2,
        "search_space_counts": count_search_space(total_lands=total_lands, max_copies=max_nonbasic_copies),
        "best": {
            "rank": 1,
            "config_id": best.config_id,
            "label": best.label,
            "polygon_area": best.polygon_area,
            "run_dir_play": best.run_dir_play,
            "run_dir_draw": best.run_dir_draw,
            "radar_png": best.radar_png,
            "manabase": dict(best.manabase),
        },
        "worst": {
            "rank": len(ranked_results),
            "config_id": worst.config_id,
            "label": worst.label,
            "polygon_area": worst.polygon_area,
            "run_dir_play": worst.run_dir_play,
            "run_dir_draw": worst.run_dir_draw,
            "radar_png": worst.radar_png,
            "manabase": dict(worst.manabase),
        },
        "comparison_png": str(comparison_png),
        "all_rankings_csv": str(search_dir / "all_configurations_ranked.csv"),
        "all_rankings_json": str(search_dir / "all_configurations_ranked.json"),
        "top_rankings_csv": str(search_dir / "top_configurations.csv"),
        "bottom_rankings_csv": str(search_dir / "bottom_configurations.csv"),
        "incremental_top_csv": str(search_dir / "incremental_top_rankings.csv"),
        "incremental_bottom_csv": str(search_dir / "incremental_bottom_rankings.csv"),
    }

    (search_dir / "search_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print("\nSearch complete")
    print(json.dumps(summary_payload, indent=2))
    return summary_payload


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main() -> None:
    run_manabase_search()


if __name__ == "__main__":
    main()
