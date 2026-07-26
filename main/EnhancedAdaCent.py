from .utils import get_acc, get_suite_dim
from .AdaCent import _linear_tail

import numpy as np

from typing import List, Tuple
from torch.quasirandom import SobolEngine


class EnhancedAdaCent:
    """
    Batched, exploration-aware K-Center early-stopping HPO.

    Parameters
    ----------
    n : int
        Number of Sobol candidate vectors to draw.
    k : int
        Centres per batch.
    T : int
        Maximum training epochs for any configuration.
    B : int
        Global training-step budget (across *all* batches).
    exploration : float, optional (default = 0.3)
        Fraction of `T` used for the *exploration* phase of a centre.
    epsilon : float, optional (default = 0.1)
        Tunable constant from the enhanced-distance formula.
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
        exploration: float = 0.1,
        seed: int = 0,
    ):
        self.n = n
        self.k = k
        self.T = T
        self.B = B
        self.explore_until = int(max(1, exploration * T))
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.last_num_used_configs = 0

    # --------------------------------------------------------------------- #
    #                                helpers                                #
    # --------------------------------------------------------------------- #
    def _choose_next_center(
        self,
        remaining_idx: np.ndarray,
        selected_idx: np.ndarray,
        perf: np.ndarray,
        best_perf: float,
        dist_mat: np.ndarray,
    ) -> int:
        """Choose the next centre with the enhanced value-weighted metric."""
        if selected_idx.size == 0:
            return int(self.rng.choice(remaining_idx))

        base = dist_mat[np.ix_(remaining_idx, selected_idx)] 
        scales = best_perf / perf[selected_idx]


        adjusted = base * scales - (1.0 / self.epsilon) * (scales - 1.0)
        scores = adjusted.min(axis=1)
        return int(remaining_idx[np.argmax(scores)])

    # --------------------------------------------------------------------- #
    #                               main loop                               #
    # --------------------------------------------------------------------- #
    def run(self, bench, task_id: str) -> List[Tuple[int, float]]:
        dim = get_suite_dim(bench)
        vecs = SobolEngine(dim, scramble=True, seed=self.seed).draw(self.n).numpy()
        diff = vecs[:, None, :] - vecs[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)

        remaining = np.ones(self.n, dtype=bool)
        perf = np.zeros(self.n, dtype=np.float32)
        total_spent, best_global = 0, -np.inf
        trace: List[Tuple[int, float]] = []
        best_perf = 0.0
        used_configs = 0

        while total_spent < self.B and remaining.any():
            selected_batch = []

            # ------------------- 3-A. pick & explore k centres -------------------
            for _ in range(self.k):
                # stop early if nothing left to pick
                remaining_idx = np.nonzero(remaining)[0]
                if len(remaining_idx) == 0 or total_spent >= self.B:
                    break

                idx = self._choose_next_center(
                    remaining_idx,
                    np.array(selected_batch, dtype=int),
                    perf,
                    max(best_perf, 1e-8),
                    dist_mat,
                )
                selected_batch.append(idx)
                remaining[idx] = False

                # ----- exploration phase -----
                hist = []
                for t in range(1, self.explore_until + 1):
                    if not hist:
                        used_configs += 1
                    acc = get_acc(vecs[idx], task_id, t, bench, self.T)
                    hist.append((t, acc))
                    total_spent += 1
                    best_global = max(best_global, acc)
                    trace.append((total_spent, best_global))
                    if total_spent >= self.B:
                        break
                if total_spent >= self.B:
                    break

                perf[idx] = hist[-1][1]
                best_perf = max(best_perf, perf[idx])

            if total_spent >= self.B or not selected_batch:
                break

            # --------------- 3-B. jointly train them with early stop -------------
            cands = []
            for idx in selected_batch:
                cands.append({
                    "idx": idx,
                    "vec": vecs[idx],
                    "spent": self.explore_until,
                    "hist": [(e, get_acc(vecs[idx], task_id, e, bench, self.T))
                             for e in range(1, self.explore_until + 1)],
                    "active": True,
                })

            while any(c["active"] for c in cands) and total_spent < self.B:
                for c in cands:
                    if not c["active"] or total_spent >= self.B:
                        continue
                    c["spent"] += 1
                    total_spent += 1
                    acc = get_acc(c["vec"], task_id, c["spent"], bench, self.T)
                    c["hist"].append((c["spent"], acc))
                    best_global = max(best_global, acc)
                    trace.append((total_spent, best_global))

                active_last_acc = [c["hist"][-1][1] for c in cands if c["active"]]
                if active_last_acc:
                    threshold = max(active_last_acc)
                    for c in cands:
                        if not c["active"]:
                            continue
                        pred = _linear_tail(c["hist"], self.T)
                        if pred < threshold or c["spent"] >= self.T:
                            c["active"] = False

        self.last_num_used_configs = used_configs
        return trace
