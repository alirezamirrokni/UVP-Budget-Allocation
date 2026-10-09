from typing import List, Tuple

import numpy as np
from torch.quasirandom import SobolEngine

from .AdaCent import _linear_tail
from .utils import get_acc, get_suite_dim


class EnhancedAdaCent:
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

    def _choose_next_center(
        self,
        remaining_idx: np.ndarray,
        selected_idx: np.ndarray,
        perf: np.ndarray,
        best_perf: float,
        dist_mat: np.ndarray,
    ) -> int:
        if selected_idx.size == 0:
            return int(self.rng.choice(remaining_idx))

        base = dist_mat[np.ix_(remaining_idx, selected_idx)]
        scales = best_perf / perf[selected_idx]
        adjusted = base * scales - (1.0 / self.epsilon) * (scales - 1.0)
        scores = adjusted.min(axis=1)
        return int(remaining_idx[np.argmax(scores)])

    def run(self, bench, task_id: str) -> List[Tuple[int, float]]:
        dim = get_suite_dim(bench)
        vecs = SobolEngine(
            dim,
            scramble=True,
            seed=self.seed,
        ).draw(self.n).numpy()
        diff = vecs[:, None, :] - vecs[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=-1)

        remaining = np.ones(self.n, dtype=bool)
        perf = np.zeros(self.n, dtype=np.float32)
        total_spent = 0
        best_global = -np.inf
        trace: List[Tuple[int, float]] = []
        best_perf = 0.0
        used_configs = 0
        protected_target = min(self.n, self.B // self.T)
        protected_selected = 0

        while total_spent < self.B and remaining.any():
            if protected_selected < protected_target:
                batch_k = min(self.k, protected_target - protected_selected)
                protected_batch = True
            else:
                batch_k = self.k
                protected_batch = False

            selected_batch = []
            cands = []

            for _ in range(batch_k):
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

                hist = []
                for t in range(1, self.explore_until + 1):
                    if total_spent >= self.B:
                        break
                    if not hist:
                        used_configs += 1
                    acc = get_acc(vecs[idx], task_id, t, bench, self.T)
                    hist.append((t, acc))
                    total_spent += 1
                    best_global = max(best_global, acc)
                    trace.append((total_spent, best_global))

                if not hist:
                    break

                perf[idx] = hist[-1][1]
                best_perf = max(best_perf, perf[idx])
                cands.append(
                    {
                        "idx": idx,
                        "vec": vecs[idx],
                        "spent": len(hist),
                        "hist": hist,
                        "active": len(hist) < self.T,
                        "protected": protected_batch,
                    }
                )

                if total_spent >= self.B:
                    break

            if protected_batch:
                protected_selected += len(cands)

            if not cands:
                break

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

                active_last_acc = [
                    c["hist"][-1][1] for c in cands if c["active"]
                ]
                if not active_last_acc:
                    break

                threshold = max(active_last_acc)
                for c in cands:
                    if not c["active"]:
                        continue
                    if c["spent"] >= self.T:
                        c["active"] = False
                        continue

                    pred = _linear_tail(c["hist"], self.T)
                    if pred < threshold:
                        c["active"] = False

        self.last_num_used_configs = used_configs
        return trace
