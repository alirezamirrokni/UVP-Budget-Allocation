from .utils import get_acc, get_suite_dim

import numpy as np

from typing import List, Tuple
from torch.quasirandom import SobolEngine


# --------------------------------------------------------------------- #
#                          centre selection helper                      #
# --------------------------------------------------------------------- #
def k_center_with_history(
    vecs: np.ndarray,
    k: int,
    rng: np.random.Generator,
    prev_centers: np.ndarray = None,
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


# --------------------------------------------------------------------- #
#                         pruning / extrapolation helper                #
# --------------------------------------------------------------------- #
def _linear_tail(
    hist: List[Tuple[int, float]],
    T: int,
    tail_x: float = 0.3,
) -> float:
    idx0 = max(0, int(np.ceil(tail_x * len(hist))) - 1)
    (t0, y0), (t1, y1) = hist[idx0], hist[-1]

    if t1 == t0:
        return y1

    slope = (y1 - y0) / (t1 - t0)

    return float(
        np.clip(
            y1 + slope * (T - t1),
            0.0,
            100.0,
        )
    )


class AdaCent:
    """
    Batched, adaptive K-Center early-stopping HPO.

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
    seed : int, optional
        RNG seed.
    """
    def __init__(
        self,
        n: int,
        k: int,
        T: int,
        B: int,
        seed: int = 0,
    ):
        self.n = n
        self.k = k
        self.T = T
        self.B = B
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

        total_spent = 0
        best_overall = -float("inf")
        trace: List[Tuple[int, float]] = []
        used_configs = 0

        while total_spent < self.B:
            # ------------------- 4-A. choose new centres -------------------
            idxs = k_center_with_history(
                vecs,
                self.k,
                self.rng,
                prev_centers,
            )

            new_centers = vecs[idxs]
            prev_centers = np.vstack([prev_centers, new_centers])

            cands = [
                {
                    "vec": vec,
                    "spent": 0,
                    "hist": [(0, 0.0)],
                    "active": True,
                }
                for vec in new_centers
            ]

            # --------------- 4-B. evaluate with early stopping --------------
            while any(c["active"] for c in cands) and total_spent < self.B:
                for c in cands:
                    if not c["active"] or total_spent >= self.B:
                        continue

                    if c["spent"] == 0:
                        used_configs += 1

                    c["spent"] += 1
                    total_spent += 1

                    acc = get_acc(c["vec"], task_id, c["spent"], bench, self.T)
                    c["hist"].append((c["spent"], acc))

                    best_overall = max(best_overall, acc)
                    trace.append((total_spent, best_overall))

                live = [c for c in cands if c["active"]]
                if not live:
                    break

                for c in live:
                    pred = _linear_tail(c["hist"], self.T)
                    if pred < best_overall:
                        c["active"] = False

                for c in cands:
                    if c["active"] and c["spent"] >= self.T:
                        c["active"] = False

        self.last_num_used_configs = used_configs
        return trace
