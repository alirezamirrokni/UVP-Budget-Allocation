from .utils import get_acc, get_suite_dim

import numpy as np

from typing import List, Tuple
from torch.quasirandom import SobolEngine


# --------------------------------------------------------------------- #
#                 enhanced centre selection helper                      #
# --------------------------------------------------------------------- #
def k_center_with_history_values(
    vecs: np.ndarray,
    k: int,
    rng: np.random.Generator,
    epsilon: float,
    prev_centers: np.ndarray = None,
    prev_vals: np.ndarray = None,
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


class EnhancedFullCent:
    """
    Batched, non-adaptive K-Center full-evaluation HPO with enhanced distances.

    Parameters
    ----------
    n : int
        Number of Sobol candidate vectors to draw.
    k : int
        Centres selected per batch.
    T : int
        Maximum training epochs for any configuration.
    B : int
        Global training-step budget across all batches.
    epsilon : float
        Smoothness parameter used in the enhanced-distance formula.
    seed : int, optional
        RNG seed.
    """
    def __init__(
        self,
        n: int,
        k: int,
        T: int,
        B: int,
        epsilon: float,
        seed: int = 0,
    ):
        self.n = n
        self.k = k
        self.T = T
        self.B = B
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.last_num_used_configs = 0

    # ----------------------------------------------------------------- #
    #                             main loop                             #
    # ----------------------------------------------------------------- #
    def run(self, bench, task_id: str) -> List[Tuple[int, float]]:
        dim = get_suite_dim(bench)
        vecs = SobolEngine(
            dim,
            scramble=True,
            seed=self.seed,
        ).draw(self.n).numpy()

        prev_centers = np.empty((0, dim), dtype=float)
        prev_vals = np.empty((0,), dtype=float)

        total_spent = 0
        best_overall = -float("inf")
        trace: List[Tuple[int, float]] = []
        used_configs = 0

        while total_spent < self.B:
            # ------------------- 4-A. choose new centres -------------------
            idxs = k_center_with_history_values(
                vecs,
                self.k,
                self.rng,
                self.epsilon,
                prev_centers,
                prev_vals,
            )

            new_centers = vecs[idxs]

            cands = [
                {
                    "vec": vec,
                    "spent": 0,
                    "value": None,
                }
                for vec in new_centers
            ]

            # ------------------- 4-B. fully evaluate centres -------------------
            while any(c["spent"] < self.T for c in cands) and total_spent < self.B:
                for c in cands:
                    if c["spent"] >= self.T or total_spent >= self.B:
                        continue

                    if c["spent"] == 0:
                        used_configs += 1

                    c["spent"] += 1
                    total_spent += 1

                    acc = get_acc(c["vec"], task_id, c["spent"], bench, self.T)
                    c["value"] = acc

                    best_overall = max(best_overall, acc)
                    trace.append((total_spent, best_overall))

            # ------------------- 4-C. update centre history -------------------
            vals = np.array(
                [
                    c["value"] if c["value"] is not None else 0.0
                    for c in cands
                ]
            )

            prev_centers = np.vstack([prev_centers, new_centers])
            prev_vals = np.concatenate([prev_vals, vals])

        self.last_num_used_configs = used_configs
        return trace
