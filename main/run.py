from __future__ import annotations

import importlib
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from yahpo_gym import benchmark_set

from .utils import estimate_epsilon, get_all_task_ids, track_objective_function

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


SUITE_NAME = "lcbench"
TASK_IDS: List[str] = []
SEEDS = list(range(30))
MAX_EPOCH = 52
HORIZON = 20 * MAX_EPOCH
HB_BUDGET = 589
OUTDIR = None

SELECTED_ALGOS: Dict[str, Dict] = {
    "Random Search": {"trials": 20},
    "Hyperband": {"iters": 6},
    "BOHB": {"iters": 6},
    "AdaCent": {"k": 25},
    "EnhancedAdaCent": {"k": 25},
}


def get_distinct_colors(n):
    return sns.color_palette("bright", n)


def trace_to_array(trace: List[Tuple[int, float]], horizon: int) -> np.ndarray:
    arr = np.empty(horizon, dtype=float)
    arr[:] = np.nan
    last, idx = 0.0, 0
    for b in range(1, horizon + 1):
        while idx < len(trace) and trace[idx][0] <= b:
            last = trace[idx][1]
            idx += 1
        arr[b - 1] = last
    return arr


def aggregate_results(
    runs: List[Dict[str, Tuple[np.ndarray, int]]],
    horizon: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, float],
    Dict[str, float],
]:
    algos = runs[0].keys()
    means, stds, budget_means, budget_stds = {}, {}, {}, {}

    for algo in algos:
        traces = np.stack([d[algo][0] for d in runs], axis=0)
        lengths = np.asarray([d[algo][1] for d in runs], dtype=float)

        means[algo] = np.nanmean(traces, axis=0)
        stds[algo] = np.nanstd(traces, axis=0)
        budget_means[algo] = float(np.mean(lengths))
        budget_stds[algo] = float(np.std(lengths))

    return means, stds, budget_means, budget_stds


def plot_task(
    task_id: str,
    means: Dict[str, np.ndarray],
    stds: Dict[str, np.ndarray],
    budget_means: Dict[str, float],
    budget_stds: Dict[str, float],
    horizon: int,
    hb_budget: int,
    eps_est: float,
    plots_dir: Path,
    num_seeds: int,
    algo_colours: Dict[str, any],
):
    xs = np.arange(1, horizon + 1)
    vals = [means[a][horizon - 1] for a in means]
    padding = (max(vals) - min(vals)) * 1.5 if max(vals) != min(vals) else 1.0
    y_min, y_max = min(vals) - padding, min(100.0, max(vals) + padding)
    x_max = max(budget_means[a] + budget_stds[a] for a in budget_means) + 50

    plt.figure(figsize=(8, 5))
    plt.ylim([y_min, y_max])
    plt.xlim([0, x_max])

    for algo, mean_arr in means.items():
        std_arr = stds[algo]
        mu_b = budget_means[algo]
        sigma_b = budget_stds[algo]
        idx = max(0, min(horizon - 1, int(round(mu_b + sigma_b)) - 1))
        xs_t = xs[:idx + 1]
        plt.plot(xs_t, mean_arr[:idx + 1], label=algo, color=algo_colours[algo])
        plt.scatter([mu_b + sigma_b], [mean_arr[idx]], color=algo_colours[algo], s=36)

    plt.xlabel("Total epochs (budget spent)")
    plt.ylabel("Best validation accuracy (mean ± std)")
    plt.title(f"OpenML {task_id} — avg over {num_seeds} seeds")
    plt.legend(title=f"estimated ε = {eps_est:.2f}")
    plt.grid(True)
    plt.tight_layout()

    out_path = plots_dir / f"{task_id}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot to {out_path}")


def make_label(name: str, params: Dict) -> str:
    if "k" in params:
        return f"{name} (k={params['k']})"
    return name


def build_algorithm(name: str, params: Dict, seed: int, max_epoch: int, horizon: int, eps: float):
    if name == "AdaCent":
        module = importlib.import_module(".AdaCent", package=__package__)
        cls = getattr(module, "AdaCent")
        return cls(n=1000, k=params["k"], T=max_epoch, B=horizon, seed=seed)

    if name == "EnhancedAdaCent":
        module = importlib.import_module(".EnhancedAdaCent", package=__package__)
        cls = getattr(module, "EnhancedAdaCent")
        return cls(
            n=1000,
            k=params["k"],
            T=max_epoch,
            B=horizon,
            epsilon=eps,
            seed=seed,
        )

    if name == "FullCent":
        module = importlib.import_module(".FullCent", package=__package__)
        cls = getattr(module, "FullCent")
        return cls(n=1000, k=params["k"], T=max_epoch, B=horizon, seed=seed)

    if name == "EnhancedFullCent":
        module = importlib.import_module(".EnhancedFullCent", package=__package__)
        cls = getattr(module, "EnhancedFullCent")
        return cls(
            n=1000,
            k=params["k"],
            T=max_epoch,
            B=horizon,
            epsilon=eps,
            seed=seed,
        )

    return None


def run_external_algorithm(name: str, params: Dict, bench, task: str, seed: int):
    module = importlib.import_module(".other_algorithms", package=__package__)

    if name == "Random Search":
        return module.random_search(bench, task, seed=seed, n_trials=params.get("trials"))

    if name == "Hyperband":
        return module.hyperband(bench, task, seed=seed, n_iterations=params.get("iters"))

    if name == "BOHB":
        return module.bohb_search(bench, task, seed=seed, n_iterations=params.get("iters"))

    if name == "SMAC":
        return module.smac_search(bench, task, seed=seed, n_trials=params.get("trials"))

    raise ValueError(f"Unknown algorithm: {name}")


def run_selected_algorithms(
    suite_name: str,
    algorithms: Dict[str, Dict],
    task_ids: List[str],
    seeds: List[int],
    max_epoch: int,
    horizon: int,
    hb_budget: int,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    avg_root = outdir / "avg_traces"
    plots_dir.mkdir(parents=True, exist_ok=True)
    avg_root.mkdir(parents=True, exist_ok=True)

    labels = [make_label(name, params) for name, params in algorithms.items()]
    algo_colours: Dict[str, any] = {}
    colors = get_distinct_colors(len(labels))
    for lbl, col in zip(labels, colors):
        algo_colours[lbl] = col

    overall_used_config_counts: Dict[str, List[int]] = {label: [] for label in labels}

    total_runs = len(task_ids) * len(seeds) * len(algorithms)
    completed_runs = 0

    for task_idx, task in enumerate(task_ids, start=1):
        bench = benchmark_set.BenchmarkSet(suite_name, instance=task, multithread=False)

        print(f"→ Running task {task} ({task_idx}/{len(task_ids)})")
        eps = estimate_epsilon(bench, task, 10_000, int(0.1 * max_epoch), max_epoch)
        print(f"  estimated epsilon: {eps:.6f}")

        all_results: List[Dict[str, Tuple[np.ndarray, int]]] = []
        used_config_counts: Dict[str, List[int]] = {label: [] for label in labels}

        for seed_idx, seed in enumerate(seeds, start=1):
            print(f"  Seed {seed} ({seed_idx}/{len(seeds)})")
            result: Dict[str, Tuple[np.ndarray, int]] = {}

            for algo_idx, (name, params) in enumerate(algorithms.items(), start=1):
                label = make_label(name, params)
                overall_idx = completed_runs + 1
                progress_pct = 100.0 * overall_idx / total_runs if total_runs > 0 else 100.0
                print(
                    f"    [{overall_idx}/{total_runs} | {progress_pct:6.2f}%] "
                    f"Running {label} (algorithm {algo_idx}/{len(algorithms)})"
                )

                with track_objective_function(bench) as tracker:
                    alg = build_algorithm(name, params, seed, max_epoch, horizon, eps)
                    if alg is not None:
                        trace = alg.run(bench, task)
                    else:
                        trace = run_external_algorithm(name, params, bench, task, seed)
                    used_configs = tracker.num_unique_configs

                used_config_counts[label].append(used_configs)
                overall_used_config_counts[label].append(used_configs)

                aligned = trace_to_array(trace, horizon)
                used_budget = trace[-1][0] if trace else 0
                result[label] = (aligned, used_budget)

                completed_runs += 1
                print(
                    f"      done: budget used = {used_budget}, "
                    f"used configs = {used_configs}"
                )

            all_results.append(result)

        means, stds, budget_means, budget_stds = aggregate_results(all_results, horizon)

        for label, mean_arr in means.items():
            std_arr = stds[label]
            algo_dir = avg_root / label.replace(" ", "_")
            algo_dir.mkdir(parents=True, exist_ok=True)
            path = algo_dir / f"{task}.npz"
            np.savez_compressed(path, mean=mean_arr, std=std_arr)
            print(f"✓ Saved average trace to {path}")

        print("Average number of used configs:")
        for label, counts in used_config_counts.items():
            print(f"  {label}: {np.mean(counts):.2f} ± {np.std(counts):.2f}")

        plot_task(task, means, stds, budget_means, budget_stds, horizon, hb_budget, eps, plots_dir, len(seeds), algo_colours)

    print("Overall average number of used configs across all tasks:")
    for label, counts in overall_used_config_counts.items():
        print(f"  {label}: {np.mean(counts):.2f} ± {np.std(counts):.2f}")

    zip_path = outdir / "avg_traces.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=outdir, base_dir="avg_traces")
    print(f"✓ Created zip: {zip_path}")


def main() -> None:
    task_ids = [str(task) for task in TASK_IDS] if TASK_IDS else get_all_task_ids(SUITE_NAME)
    outdir = Path(f"custom_plots_{SUITE_NAME}") if OUTDIR is None else Path(OUTDIR)

    run_selected_algorithms(
        suite_name=SUITE_NAME,
        algorithms=SELECTED_ALGOS,
        task_ids=task_ids,
        seeds=list(SEEDS),
        max_epoch=MAX_EPOCH,
        horizon=HORIZON,
        hb_budget=HB_BUDGET,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()
