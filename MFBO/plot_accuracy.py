from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METHODS = ["EnhancedAdaCent", "AdaCent", "RMFBO", "MES", "KG", "RAND"]
DISPLAY_NAMES = {
    "KG": "KG",
    "MES": "MES",
    "RAND": "Random",
    "RMFBO": "rMFBO",
    "AdaCent": "AdaCent",
    "EnhancedAdaCent": "eAdaCent",
}
COLORS = {
    "KG": "#4C72B0",
    "MES": "#DD8452",
    "RAND": "#55A868",
    "RMFBO": "#C44E52",
    "AdaCent": "#8172B3",
    "EnhancedAdaCent": "#937860",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the final mean accuracy of each method as bars with "
            "standard-error error bars."
        )
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="Problem name used at the beginning of result filenames.",
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=Path("res/records"),
        help="Directory containing numeric result .npy files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method names to plot, from left to right.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Require an exact number of repetitions, such as 10.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. Defaults to the problem name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "res/figures/<problem>_final_accuracy_se_bar.png."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Optional y-axis limits, expressed in the displayed units.",
    )
    parser.add_argument(
        "--no-percent",
        action="store_true",
        help="Do not multiply values in [0, 1] by 100.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Write the mean final value above each error bar.",
    )
    return parser.parse_args()


def _method_pattern(problem: str, method: str) -> re.Pattern[str]:
    return re.compile(
        rf"^{re.escape(problem)}(?:_NEG)?_{re.escape(method)}"
        rf"(?:-DK)?_R(?P<runs>\d+)_"
    )


def find_result_file(
    records_dir: Path,
    problem: str,
    method: str,
    required_runs: int | None,
) -> Path | None:
    pattern = _method_pattern(problem, method)
    candidates: list[tuple[int, float, Path]] = []

    for path in records_dir.glob("*.npy"):
        match = pattern.match(path.name)
        if match is None:
            continue

        runs = int(match.group("runs"))
        if required_runs is not None and runs != required_runs:
            continue

        candidates.append((runs, path.stat().st_mtime, path))

    if not candidates:
        return None


    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def load_numeric_record(path: Path) -> np.ndarray:
    record = np.load(path, allow_pickle=False)
    if record.ndim != 3 or record.shape[-1] != 2:
        raise ValueError(
            f"{path} has shape {record.shape}; expected (runs, iterations, 2)."
        )
    return np.asarray(record, dtype=float)


def final_values(record: np.ndarray) -> np.ndarray:
    values: list[float] = []

    for run in record:
        rewards = run[:, 0]
        costs = run[:, 1]
        valid = np.isfinite(rewards) & np.isfinite(costs) & (costs > 0)
        if np.any(valid):
            values.append(float(rewards[np.flatnonzero(valid)[-1]]))

    if not values:
        raise ValueError("The result file contains no completed evaluations.")

    return np.asarray(values, dtype=float)


def summarize_final(record: np.ndarray) -> tuple[float, float, int]:
    values = final_values(record)
    mean = float(np.mean(values))
    if values.size > 1:
        stderr = float(np.std(values, ddof=1) / math.sqrt(values.size))
    else:
        stderr = 0.0
    return mean, stderr, int(values.size)


def automatic_ylim(means: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(means - errors))
    upper = float(np.max(means + errors))
    spread = max(upper - lower, max(abs(upper), 1.0) * 0.015)
    padding = 0.35 * spread

    ymin = lower - padding
    ymax = upper + padding


    step = 1.0 if max(abs(ymin), abs(ymax)) > 10 else 0.1
    ymin = math.floor(ymin / step) * step
    ymax = math.ceil(ymax / step) * step
    if ymin == ymax:
        ymax = ymin + step
    return ymin, ymax


def main() -> None:
    args = parse_args()
    summaries: list[tuple[str, float, float, int, Path]] = []

    for method in args.methods:
        path = find_result_file(
            args.records_dir,
            args.problem,
            method,
            args.runs,
        )
        if path is None:
            print(f"Skipping {method}: no matching result file found.")
            continue

        record = load_numeric_record(path)
        mean, stderr, n_runs = summarize_final(record)
        summaries.append((method, mean, stderr, n_runs, path))
        print(
            f"{method}: final mean={mean:.6f}, SE={stderr:.6f}, "
            f"runs={n_runs}, file={path}"
        )

    if not summaries:
        raise FileNotFoundError("No matching result files were found.")

    raw_means = np.asarray([item[1] for item in summaries], dtype=float)
    raw_errors = np.asarray([item[2] for item in summaries], dtype=float)
    use_percent = not args.no_percent and np.nanmax(np.abs(raw_means)) <= 1.5
    scale = 100.0 if use_percent else 1.0

    means = scale * raw_means
    errors = scale * raw_errors
    labels = [DISPLAY_NAMES.get(item[0], item[0]) for item in summaries]
    colors = [COLORS.get(item[0], "#4C72B0") for item in summaries]
    x = np.arange(len(summaries), dtype=float)

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "xtick.labelsize": 17,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.4,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(
        x,
        means,
        width=0.68,
        color=colors,
        edgecolor=colors,
        linewidth=1.2,
        yerr=errors,
        capsize=7,
        error_kw={
            "ecolor": "#2F2F2F",
            "elinewidth": 2.0,
            "capthick": 2.0,
        },
        zorder=3,
    )

    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    else:
        ax.set_ylim(*automatic_ylim(means, errors))

    ax.set_xticks(x, labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy (%)" if use_percent else "Objective value")
    ax.set_title(args.title or args.problem, pad=14)
    ax.grid(axis="y", linewidth=1.1, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.05)


    for spine in ax.spines.values():
        spine.set_color("#C8C8C8")
        spine.set_linewidth(1.4)

    if args.show_values:
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for bar, mean, error in zip(bars, means, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + error + 0.018 * y_span,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=15,
            )

    fig.tight_layout()

    output = args.output or (
        Path("res/figures") / f"{args.problem}_final_accuracy_se_bar.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output}")


if __name__ == "__main__":
    main()
