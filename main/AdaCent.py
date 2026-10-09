from typing import List, Tuple

import numpy as np
from torch.quasirandom import SobolEngine

from .utils import get_acc, get_suite_dim


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
    return float(np.clip(y1 + slope * (T - t1), 0.0, 100.0))


class AdaCent:
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

    def run(self, bench, task_id: str) -> List[Tuple[int, float]]:
        dim = get_suite_dim(bench)
        vecs = SobolEngine(
            dim,
            scramble=True,
            seed=self.seed,
        ).draw(self.n).numpy()

        prev_centers = np.empty((0, dim), dtype=float)
        protected_target = min(self.n, self.B // self.T)
        protected_selected = 0

        total_spent = 0
        best_overall = -float("inf")
        trace: List[Tuple[int, float]] = []
        used_configs = 0

        while total_spent < self.B:
            if protected_selected < protected_target:
                batch_k = min(self.k, protected_target - protected_selected)
                protected_batch = True
            else:
                batch_k = self.k
                protected_batch = False

            if batch_k <= 0:
                break

            idxs = k_center_with_history(
                vecs,
                batch_k,
                self.rng,
                prev_centers,
            )

            new_centers = vecs[idxs]
            prev_centers = np.vstack([prev_centers, new_centers])

            cands = [
                {
                    "idx": int(idx),
                    "vec": vec,
                    "spent": 0,
                    "hist": [(0, 0.0)],
                    "active": True,
                    "protected": protected_batch,
                }
                for idx, vec in zip(idxs, new_centers)
            ]

            if protected_batch:
                protected_selected += len(cands)

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
                    if c["spent"] >= self.T:
                        c["active"] = False
                        continue

                    pred = _linear_tail(c["hist"], self.T)
                    if pred < best_overall:
                        c["active"] = False

        self.last_num_used_configs = used_configs
        return trace
