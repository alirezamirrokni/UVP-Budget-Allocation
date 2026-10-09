from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine


def _linear_tail(
    hist: List[Tuple[float, float]],
    target_fidelity: float,
    tail_x: float = 0.3,
) -> float:
    if len(hist) == 0:
        return 0.0

    if len(hist) == 1:
        return hist[-1][1]

    idx0 = max(0, int(np.ceil(tail_x * len(hist))) - 1)
    (t0, y0), (t1, y1) = hist[idx0], hist[-1]

    if t1 == t0:
        return y1

    slope = (y1 - y0) / (t1 - t0)
    return float(np.clip(y1 + slope * (target_fidelity - t1), 0.0, 1e9))


def _draw_candidate_pool(problem, n_candidates: int, tkwargs: dict) -> torch.Tensor:
    sobol = SobolEngine(dimension=problem.dim - 1, scramble=True)
    base = sobol.draw(n_candidates).to(**tkwargs)


    bounds = problem.bounds[:, :-1].to(**tkwargs)
    candidates = unnormalize(base, bounds=bounds)
    lower = bounds[0].unsqueeze(0)
    upper = bounds[1].unsqueeze(0)
    return torch.maximum(lower, torch.minimum(candidates, upper))


def _eval_at_fidelity(problem, vec: torch.Tensor, fidelity: float, tkwargs: dict) -> float:
    x_full = torch.cat(
        [
            vec.reshape(1, -1),
            torch.tensor([[fidelity]], **tkwargs),
        ],
        dim=1,
    )
    y = problem(x_full).reshape(-1)[0]
    return float(y.item())


def k_center_with_history(
    vecs: np.ndarray,
    k: int,
    rng: np.random.Generator,
    prev_centers: np.ndarray | None = None,
) -> List[int]:
    n_points = vecs.shape[0]

    if prev_centers is None or prev_centers.size == 0:
        init_idx = rng.integers(n_points)
        centers = [init_idx]

        while len(centers) < k:
            selected = vecs[centers]
            dists = np.min(
                np.linalg.norm(
                    vecs[:, None, :] - selected[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            next_idx = int(np.argmax(dists))
            centers.append(next_idx)

        return centers

    dists_prev = np.min(
        np.linalg.norm(
            vecs[:, None, :] - prev_centers[None, :, :],
            axis=2,
        ),
        axis=1,
    )

    centers: List[int] = []

    for _ in range(k):
        if centers:
            selected = vecs[centers]
            dists_new = np.min(
                np.linalg.norm(
                    vecs[:, None, :] - selected[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            dists = np.minimum(dists_prev, dists_new)
        else:
            dists = dists_prev

        next_idx = int(np.argmax(dists))
        centers.append(next_idx)

    return centers


def k_center_with_history_values(
    vecs: np.ndarray,
    k: int,
    rng: np.random.Generator,
    epsilon: float,
    prev_centers: np.ndarray | None = None,
    prev_vals: np.ndarray | None = None,
) -> List[int]:
    n_points = vecs.shape[0]
    available = np.ones(n_points, dtype=bool)

    if prev_centers is not None and prev_centers.size > 0:
        thr = 1e-14
        diff2 = np.sum(
            (vecs[:, None, :] - prev_centers[None, :, :]) ** 2,
            axis=2,
        )
        available[np.any(diff2 < thr, axis=1)] = False

        v_max = float(np.max(prev_vals))
        scales = v_max / prev_vals

        base_d = np.linalg.norm(
            vecs[:, None, :] - prev_centers[None, :, :],
            axis=2,
        )

        mod_d = (
            scales[None, :] * base_d
            - (scales[None, :] - 1.0) / epsilon
        )

        cur_min = np.min(mod_d, axis=1)
    else:
        cur_min = np.full(n_points, np.inf)

    centers: List[int] = []

    feasible = np.flatnonzero(available)
    if feasible.size == 0:
        return centers

    first = int(rng.choice(feasible))
    centers.append(first)
    available[first] = False

    new_d = np.linalg.norm(vecs - vecs[first], axis=1)
    cur_min = np.minimum(cur_min, new_d)

    while len(centers) < k:
        feasible = np.flatnonzero(available)
        if feasible.size == 0:
            break

        best_idx = feasible[np.argmax(cur_min[feasible])]
        centers.append(int(best_idx))
        available[best_idx] = False

        new_d = np.linalg.norm(vecs - vecs[best_idx], axis=1)
        cur_min = np.minimum(cur_min, new_d)

    return centers


def _serialize_adacent_state(
    *,
    pool_np: np.ndarray,
    prev_centers: np.ndarray,
    prev_vals: np.ndarray,
    cands: list[dict],
    phase: str,
    cursor: int,
    batch_indices: list[int],
    best_overall: float,
    rng: np.random.Generator,
) -> dict:
    return {
        "pool_np": np.asarray(pool_np),
        "prev_centers": np.asarray(prev_centers),
        "prev_vals": np.asarray(prev_vals),
        "cands": [
            {
                "idx": int(c["idx"]),
                "spent_idx": int(c["spent_idx"]),
                "hist": [(float(fid), float(value)) for fid, value in c["hist"]],
                "active": bool(c["active"]),
            }
            for c in cands
        ],
        "phase": str(phase),
        "cursor": int(cursor),
        "batch_indices": [int(idx) for idx in batch_indices],
        "best_overall": float(best_overall),
        "rng_state": rng.bit_generator.state,
    }


def _run_adacent_core(
    problem,
    fidelities: torch.Tensor,
    config,
    tkwargs: dict,
    *,
    enhanced: bool,
    train_x: torch.Tensor | None = None,
    train_obj: torch.Tensor | None = None,
    cumulative_cost: list[float] | None = None,
    algorithm_state: dict | None = None,
    checkpoint_callback=None,
    show_progress: bool = True,
):
    from tqdm import tqdm

    n_candidates = int(config.algorithm.n_candidates)
    k = int(config.algorithm.k)
    exploration = float(config.algorithm.exploration)
    tail_x = float(config.algorithm.tail_x)
    epsilon = float(config.algorithm.epsilon) if enhanced else None

    budget_limit = float(config.problem.budget)
    low_cost = float(config.problem.low_cost)

    fidelity_values = [float(f.item()) for f in fidelities]
    target_fidelity = float(config.problem.target_fidelity)
    n_fids = len(fidelity_values)
    explore_until = max(1, int(np.ceil(exploration * n_fids)))
    full_completion_cost = sum(low_cost + fid for fid in fidelity_values)
    protected_target = int(budget_limit // max(full_completion_cost, 1e-12))

    cumulative_cost = [] if cumulative_cost is None else list(cumulative_cost)
    if train_x is None:
        train_x = torch.empty((0, problem.dim), **tkwargs)
    if train_obj is None:
        train_obj = torch.empty((0, 1), **tkwargs)

    state = algorithm_state or {}
    rng = np.random.default_rng()
    if state.get("rng_state") is not None:
        rng.bit_generator.state = state["rng_state"]

    if state.get("pool_np") is None:
        pool_t = _draw_candidate_pool(problem, n_candidates, tkwargs)
        pool_np = pool_t.detach().cpu().numpy()
    else:
        pool_np = np.asarray(state["pool_np"], dtype=float)
        pool_t = torch.as_tensor(pool_np, **tkwargs)

    prev_centers = np.asarray(
        state.get("prev_centers", np.empty((0, problem.dim - 1), dtype=float)),
        dtype=float,
    )
    prev_vals = np.asarray(state.get("prev_vals", np.empty((0,), dtype=float)), dtype=float)
    cands = [dict(c) for c in state.get("cands", [])]
    phase = str(state.get("phase", "select"))
    cursor = int(state.get("cursor", 0))
    batch_indices = [int(idx) for idx in state.get("batch_indices", [])]
    best_overall = float(
        state.get(
            "best_overall",
            train_obj.max().item() if train_obj.numel() else -float("inf"),
        )
    )

    max_evals = int(np.ceil(budget_limit / max(low_cost + min(fidelity_values), 1e-12))) + 1
    progress = tqdm(
        total=max_evals,
        initial=min(len(cumulative_cost), max_evals),
        desc=config.algorithm.name,
        unit="eval",
        leave=False,
        disable=not show_progress,
    )

    def save_after_evaluation() -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            train_x,
            train_obj,
            cumulative_cost,
            _serialize_adacent_state(
                pool_np=pool_np,
                prev_centers=prev_centers,
                prev_vals=prev_vals,
                cands=cands,
                phase=phase,
                cursor=cursor,
                batch_indices=batch_indices,
                best_overall=best_overall,
                rng=rng,
            ),
        )

    try:
        while True:
            if phase == "select":
                if sum(cumulative_cost) >= budget_limit:
                    break

                protected_resolved = min(len(prev_centers), protected_target)
                if protected_resolved < protected_target:
                    batch_k = min(k, protected_target - protected_resolved)
                else:
                    batch_k = k

                if enhanced:
                    idxs = k_center_with_history_values(
                        pool_np,
                        batch_k,
                        rng,
                        epsilon=epsilon,
                        prev_centers=prev_centers,
                        prev_vals=prev_vals,
                    )
                else:
                    idxs = k_center_with_history(
                        pool_np,
                        batch_k,
                        rng,
                        prev_centers=prev_centers,
                    )

                if len(idxs) == 0:
                    break

                batch_indices = [int(idx) for idx in idxs]
                cands = [
                    {
                        "idx": int(idx),
                        "spent_idx": 0,
                        "hist": [],
                        "active": True,
                    }
                    for idx in batch_indices
                ]
                cursor = 0
                phase = "evaluate"

            if phase == "evaluate":
                if sum(cumulative_cost) > budget_limit:
                    phase = "finalize"
                    continue

                while cursor < len(cands):
                    c = cands[cursor]
                    cursor += 1

                    if not c["active"]:
                        continue
                    if c["spent_idx"] >= n_fids:
                        c["active"] = False
                        continue
                    fid = fidelity_values[c["spent_idx"]]
                    eval_cost = low_cost + fid
                    if sum(cumulative_cost) + eval_cost > budget_limit:
                        phase = "finalize"
                        break
                    vec_t = pool_t[c["idx"]]
                    y = _eval_at_fidelity(problem, vec_t, fid, tkwargs)

                    x_full = torch.cat(
                        [
                            vec_t.reshape(1, -1),
                            torch.tensor([[fid]], **tkwargs),
                        ],
                        dim=1,
                    )
                    y_t = torch.tensor([[y]], **tkwargs)
                    train_x = torch.cat([train_x, x_full], dim=0)
                    train_obj = torch.cat([train_obj, y_t], dim=0)

                    c["hist"].append((fid, y))
                    c["spent_idx"] += 1
                    cumulative_cost.append(eval_cost)
                    best_overall = max(best_overall, y)
                    progress.update(1)
                    progress.set_postfix_str(
                        f"best={best_overall:.4f}, budget={sum(cumulative_cost):.2f}/{budget_limit:g}"
                    )

                    save_after_evaluation()

                    if sum(cumulative_cost) >= budget_limit:
                        phase = "finalize"
                        break

                if phase == "evaluate" and cursor >= len(cands):
                    phase = "prune"

            if phase == "prune":
                live = [c for c in cands if c["active"]]
                for c in live:
                    if c["spent_idx"] < explore_until:
                        continue
                    pred = _linear_tail(c["hist"], target_fidelity, tail_x=tail_x)
                    if pred < best_overall:
                        c["active"] = False

                for c in cands:
                    if c["active"] and c["spent_idx"] >= n_fids:
                        c["active"] = False

                if any(c["active"] for c in cands):
                    cursor = 0
                    phase = "evaluate"
                else:
                    phase = "finalize"

            if phase == "finalize":
                vals = np.array(
                    [
                        max(
                            max(y for _, y in c["hist"])
                            if c["hist"]
                            else -float("inf"),
                            _linear_tail(c["hist"], target_fidelity, tail_x=tail_x)
                            if c["hist"]
                            else -float("inf"),
                        )
                        for c in cands
                    ],
                    dtype=float,
                )
                if batch_indices:
                    prev_centers = np.vstack([prev_centers, pool_np[batch_indices]])
                    prev_vals = np.concatenate([prev_vals, vals])

                cands = []
                batch_indices = []
                cursor = 0
                phase = "select"

                if sum(cumulative_cost) >= budget_limit:
                    break
    finally:
        progress.close()

    final_state = _serialize_adacent_state(
        pool_np=pool_np,
        prev_centers=prev_centers,
        prev_vals=prev_vals,
        cands=cands,
        phase=phase,
        cursor=cursor,
        batch_indices=batch_indices,
        best_overall=best_overall,
        rng=rng,
    )
    return train_x, train_obj, cumulative_cost, final_state


def run_adacent(problem, fidelities, config, tkwargs, **kwargs):
    return _run_adacent_core(
        problem,
        fidelities,
        config,
        tkwargs,
        enhanced=False,
        **kwargs,
    )


def run_enhanced_adacent(problem, fidelities, config, tkwargs, **kwargs):
    return _run_adacent_core(
        problem,
        fidelities,
        config,
        tkwargs,
        enhanced=True,
        **kwargs,
    )
