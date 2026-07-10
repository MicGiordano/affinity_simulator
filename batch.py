from __future__ import annotations

from collections import defaultdict
from math import pi
from pathlib import Path
import json

import matplotlib.pyplot as plt

from sim.api_hidden_info import simulate_one_game


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

N_GAMES = 100
NUM_TURNS = 5
SEED_START = 1

# If None, metrics are auto-detected from the simulator output
SELECTED_METRICS = None

# Optional output files
SAVE_JSON = True
SAVE_PNG = True
OUTPUT_DIR = "batch_outputs"
RUN_NAME = "radar_batch"


# ----------------------------------------------------------
# BATCH SIMULATION
# ----------------------------------------------------------

def run_batch(
    n_games: int = N_GAMES,
    num_turns: int = NUM_TURNS,
    seed_start: int = SEED_START,
) -> dict[int, dict[str, float]]:
    """
    Run multiple simulations and average metrics per turn.

    Returns:
        {
            1: {"metric_a": avg, "metric_b": avg, ...},
            2: {...},
            ...
        }
    """
    aggregated: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for i in range(n_games):
        result = simulate_one_game(
            seed=seed_start + i,
            num_turns=num_turns,
        )

        log = result["log"]

        for turn_data in log["turns"]:
            turn = turn_data["turn"]
            metrics = turn_data["metrics"]

            for metric_name, value in metrics.items():
                aggregated[turn][metric_name].append(float(value))

        if (i + 1) % 25 == 0 or i == n_games - 1:
            print(f"Completed {i + 1}/{n_games} games")

    averaged: dict[int, dict[str, float]] = {}
    for turn, metric_dict in aggregated.items():
        averaged[turn] = {}
        for metric_name, values in metric_dict.items():
            averaged[turn][metric_name] = sum(values) / len(values) if values else 0.0

    return averaged


# ----------------------------------------------------------
# PROFILE BUILDING
# ----------------------------------------------------------

def detect_metric_names(averaged: dict[int, dict[str, float]]) -> list[str]:
    """
    Detect metric names from the averaged data.
    """
    if not averaged:
        return []
    first_turn = next(iter(sorted(averaged.keys())))
    return list(averaged[first_turn].keys())


def collapse_to_single_profile(
    averaged: dict[int, dict[str, float]],
    selected_metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Collapse turn-by-turn metrics into one overall profile
    by averaging each metric across turns.
    """
    if not averaged:
        return {}

    if selected_metrics is None:
        selected_metrics = detect_metric_names(averaged)

    collapsed: dict[str, list[float]] = defaultdict(list)

    for turn in sorted(averaged.keys()):
        for metric in selected_metrics:
            collapsed[metric].append(float(averaged[turn].get(metric, 0.0)))

    return {
        metric: (sum(values) / len(values) if values else 0.0)
        for metric, values in collapsed.items()
    }


# ----------------------------------------------------------
# NORMALIZATION
# ----------------------------------------------------------

def normalize_profile_to_10(profile: dict[str, float]) -> dict[str, float]:
    """
    Normalize metric values to a 0-10 radar scale.

    Assumption:
    Most metrics already live roughly in 0..1 or small positive ranges.
    We clamp after scaling to keep the radar readable.

    If later you want smarter normalization (min/max across multiple builds),
    we can swap this out cleanly.
    """
    normalized = {}
    for metric, value in profile.items():
        scaled = value * 10.0
        normalized[metric] = max(0.0, min(10.0, scaled))
    return normalized


# ----------------------------------------------------------
# DEBUG / PRINTING
# ----------------------------------------------------------

def print_turn_averages(averaged: dict[int, dict[str, float]]) -> None:
    print("\n=== AVERAGED TURN METRICS ===")
    for turn in sorted(averaged.keys()):
        print(f"\nTurn {turn}")
        for metric, value in averaged[turn].items():
            print(f"  {metric}: {value:.4f}")


def print_final_profile(profile: dict[str, float], title: str = "FINAL PROFILE") -> None:
    print(f"\n=== {title} ===")
    for metric, value in profile.items():
        print(f"{metric}: {value:.4f}")


# ----------------------------------------------------------
# RADAR PLOTTING
# ----------------------------------------------------------

def plot_radar(
    profile_a: dict[str, float],
    profile_b: dict[str, float] | None = None,
    label_a: str = "Run A",
    label_b: str = "Run B",
    title: str = "Deck Performance Radar",
    save_path: str | None = None,
) -> None:
    """
    Plot one or two normalized profiles on a radar chart.
    """
    if not profile_a:
        raise ValueError("profile_a is empty")

    metrics = list(profile_a.keys())
    n = len(metrics)
    if n == 0:
        raise ValueError("No metrics available to plot")

    # Angles for radar axes
    angles = [2 * pi * i / n for i in range(n)]
    angles += angles[:1]  # close polygon

    values_a = [profile_a[m] for m in metrics]
    values_a += values_a[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    # Plot profile A
    ax.plot(angles, values_a, linewidth=2, label=label_a)
    ax.fill(angles, values_a, alpha=0.25)

    # Optional comparison profile
    if profile_b is not None:
        values_b = [profile_b[m] for m in metrics]
        values_b += values_b[:1]
        ax.plot(angles, values_b, linewidth=2, label=label_b)
        ax.fill(angles, values_b, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"])

    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)

    plt.show()


# ----------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------

def save_outputs(
    averaged: dict[int, dict[str, float]],
    raw_profile: dict[str, float],
    normalized_profile: dict[str, float],
    output_dir: str = OUTPUT_DIR,
    run_name: str = RUN_NAME,
) -> dict[str, str]:
    """
    Save JSON artifacts for debugging / reuse.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    if SAVE_JSON:
        averaged_path = out / f"{run_name}_turn_averages.json"
        raw_profile_path = out / f"{run_name}_profile_raw.json"
        norm_profile_path = out / f"{run_name}_profile_normalized.json"

        averaged_path.write_text(json.dumps(averaged, indent=2))
        raw_profile_path.write_text(json.dumps(raw_profile, indent=2))
        norm_profile_path.write_text(json.dumps(normalized_profile, indent=2))

        paths["turn_averages_json"] = str(averaged_path)
        paths["profile_raw_json"] = str(raw_profile_path)
        paths["profile_normalized_json"] = str(norm_profile_path)

    return paths


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    averaged = run_batch(
        n_games=N_GAMES,
        num_turns=NUM_TURNS,
        seed_start=SEED_START,
    )

    metric_names = SELECTED_METRICS or detect_metric_names(averaged)

    raw_profile = collapse_to_single_profile(
        averaged,
        selected_metrics=metric_names,
    )

    normalized_profile = normalize_profile_to_10(raw_profile)

    print_turn_averages(averaged)
    print_final_profile(raw_profile, title="RAW PROFILE")
    print_final_profile(normalized_profile, title="NORMALIZED PROFILE (0-10)")

    save_paths = save_outputs(
        averaged=averaged,
        raw_profile=raw_profile,
        normalized_profile=normalized_profile,
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
    )

    png_path = None
    if SAVE_PNG:
        png_path = str(Path(OUTPUT_DIR) / f"{RUN_NAME}_radar.png")

    plot_radar(
        normalized_profile,
        title=f"{RUN_NAME} – Radar Profile",
        save_path=png_path,
    )

    if save_paths:
        print("\nSaved files:")
        for label, path in save_paths.items():
            print(f"  {label}: {path}")
    if png_path:
        print(f"  radar_png: {png_path}")


if __name__ == "__main__":
    main()