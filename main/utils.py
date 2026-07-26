# ───────────────────────── imports ──────────────────────────
import contextlib
import json

import numpy as np
from torch.quasirandom import SobolEngine
from yahpo_gym import BenchmarkSet


# ─────────────────── suite helpers ───────────────────
SUITE_DIMS = {
    "lcbench": 7,
    "rbv2_aknn": 6,
    "rbv2_rpart": 5,
}


SUITE_FIDELITY_KEYS = {
    "lcbench": "epoch",
    "rbv2_aknn": "trainsize",
    "rbv2_rpart": "trainsize",
}


def infer_suite_name(bench) -> str:
    config_id = getattr(getattr(bench, "config", None), "config_id", "")
    for suite_name in SUITE_DIMS:
        if suite_name in str(config_id):
            return suite_name
    raise ValueError(f"Could not infer suite name from benchmark config id: {config_id}")


def get_suite_dim(bench) -> int:
    return SUITE_DIMS[infer_suite_name(bench)]


def get_fidelity_key(bench) -> str:
    return SUITE_FIDELITY_KEYS[infer_suite_name(bench)]


def get_all_task_ids(suite_name: str) -> list[str]:
    bench = BenchmarkSet(suite_name, multithread=False)

    for attr in ["instances", "instance_names", "available_instances"]:
        val = getattr(bench, attr, None)
        if val is not None:
            try:
                task_ids = [str(v) for v in list(val)]
                if task_ids:
                    return sorted(task_ids, key=lambda x: (len(x), x))
            except Exception:
                pass

    cs = getattr(bench, "config_space", None)
    if cs is not None:
        for hp_name in ["OpenML_task_id", "task_id"]:
            try:
                hp = cs.get_hyperparameter(hp_name)
                choices = getattr(hp, "choices", None)
                if choices is not None and len(choices) > 0:
                    task_ids = [str(v) for v in list(choices)]
                    return sorted(task_ids, key=lambda x: (len(x), x))
            except Exception:
                pass

    get_opt_space = getattr(bench, "get_opt_space", None)
    if callable(get_opt_space):
        try:
            cs = get_opt_space()
            for hp_name in ["OpenML_task_id", "task_id"]:
                try:
                    hp = cs.get_hyperparameter(hp_name)
                    choices = getattr(hp, "choices", None)
                    if choices is not None and len(choices) > 0:
                        task_ids = [str(v) for v in list(choices)]
                        return sorted(task_ids, key=lambda x: (len(x), x))
                except Exception:
                    pass
        except Exception:
            pass

    raise RuntimeError(f"Could not discover task IDs for suite '{suite_name}'.")


# ───────────────────────── tracking helpers ─────────────────────────
def _normalize_for_hash(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_for_hash(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _normalize_for_hash(v)) for k, v in value.items()))
    return str(value)


class ObjectiveTracker:
    def __init__(self, bench):
        self.bench = bench
        self.suite_name = infer_suite_name(bench)
        self.fidelity_key = get_fidelity_key(bench)
        self.unique_configs = set()
        self.total_queries = 0

    def _register_config(self, config):
        if isinstance(config, (list, tuple)) and len(config) > 0:
            config = config[0]
        if hasattr(config, "get_dictionary"):
            config = config.get_dictionary()
        if not isinstance(config, dict):
            return

        cfg = dict(config)
        cfg.pop(self.fidelity_key, None)
        key = tuple(sorted((str(k), _normalize_for_hash(v)) for k, v in cfg.items()))
        self.unique_configs.add(key)
        self.total_queries += 1

    @property
    def num_unique_configs(self) -> int:
        return len(self.unique_configs)


@contextlib.contextmanager
def track_objective_function(bench):
    tracker = ObjectiveTracker(bench)
    original = bench.objective_function

    def wrapped_objective_function(config, *args, **kwargs):
        tracker._register_config(config)
        return original(config, *args, **kwargs)

    bench.objective_function = wrapped_objective_function
    try:
        yield tracker
    finally:
        bench.objective_function = original


# ─────────────────── YAHPO helpers ─────────────────
# for lcbench
def vector_to_config(x: np.ndarray, task_id: str, epoch: int) -> dict:
    x0 = np.clip(x[0], 0.0, 1.0)
    low = np.float64(0.00010000000000000009)
    high = np.float64(0.10000000000000002)
    lr = np.float64(10 ** (np.log10(low) + x0 * (np.log10(high) - np.log10(low))))
    lr = np.clip(lr, low, high)

    return {
        "OpenML_task_id": str(task_id),
        "epoch": int(epoch),
        "learning_rate": lr,
        "max_dropout": float(x[1]),
        "max_units": int(round(2 ** (np.log2(64) + x[2] * (np.log2(1024) - np.log2(64))))),
        "momentum": float(0.1 + x[3] * (0.99 - 0.1)),
        "num_layers": int(round(1 + x[4] * 4)),
        "weight_decay": float(10 ** (np.log10(1e-5) + x[5] * (np.log10(1e-1) - np.log10(1e-5)))),
        "batch_size": int(round(2 ** (np.log2(16) + x[6] * (np.log2(512) - np.log2(16)))))
    }


def get_val_accuracy(x: np.ndarray, task_id: str, epoch: int, bench) -> float:
    """One surrogate call."""
    res = bench.objective_function(vector_to_config(x, task_id, epoch))
    return res[0]["val_accuracy"]


# ─────────────────────────────────────────────────────────
# for aknn
def vector_to_config_(x: np.ndarray, task_id: str, trainsize: float, impute: str) -> dict:
    M = int(round(18 + x[0] * (50 - 18)))
    M = int(np.clip(M, 18, 50))

    dist_choices = ["l2", "cosine", "ip"]
    idx = min(int(x[1] * len(dist_choices)), len(dist_choices) - 1)
    distance = dist_choices[idx]

    ef_low, ef_high = 8, 256
    ef = int(round(10 ** (np.log10(ef_low) + x[2] * (np.log10(ef_high) - np.log10(ef_low)))))
    ef = int(np.clip(ef, ef_low, ef_high))

    ec_low, ec_high = 8, 512
    ef_construction = int(round(10 ** (np.log10(ec_low) + x[3] * (np.log10(ec_high) - np.log10(ec_low)))))
    ef_construction = int(np.clip(ef_construction, ec_low, ec_high))

    k = int(round(1 + x[4] * (50 - 1)))
    k = int(np.clip(k, 1, 50))

    repl = int(round(1 + x[5] * (10 - 1)))
    repl = int(np.clip(repl, 1, 10))

    return {
        "task_id": str(task_id),
        "trainsize": float(trainsize),
        "num.impute.selected.cpo": impute,
        "M": M,
        "distance": distance,
        "ef": ef,
        "ef_construction": ef_construction,
        "k": k,
        "repl": repl,
    }


# for rpart
def vector_to_config__(x: np.ndarray, task_id: str, trainsize: float, impute: str) -> dict:
    cp_low, cp_high = 0.0009118819655545162, 1.0
    cp = 10 ** (
        np.log10(cp_low)
        + x[0] * (np.log10(cp_high) - np.log10(cp_low))
    )
    cp = float(np.clip(cp, cp_low, cp_high))

    md_low, md_high = 1, 30
    maxdepth = int(round(md_low + x[1] * (md_high - md_low)))
    maxdepth = int(np.clip(maxdepth, md_low, md_high))

    mb_low, mb_high = 1, 100
    minbucket = int(round(mb_low + x[2] * (mb_high - mb_low)))
    minbucket = int(np.clip(minbucket, mb_low, mb_high))

    ms_low, ms_high = 1, 100
    minsplit = int(round(ms_low + x[3] * (ms_high - ms_low)))
    minsplit = int(np.clip(minsplit, ms_low, ms_high))

    repl_low, repl_high = 1, 10
    repl = int(round(repl_low + x[4] * (repl_high - repl_low)))
    repl = int(np.clip(repl, repl_low, repl_high))

    return {
        "cp": cp,
        "maxdepth": maxdepth,
        "minbucket": minbucket,
        "minsplit": minsplit,
        "num.impute.selected.cpo": impute,
        "repl": repl,
        "task_id": str(task_id),
        "trainsize": float(trainsize),
    }


# for all suites
def get_acc(x: np.ndarray, task_id: str, fidelity_step: int, bench, max_steps: int = 52) -> float:
    """One surrogate call."""
    suite_name = infer_suite_name(bench)

    if suite_name == "lcbench":
        return get_val_accuracy(x, task_id, fidelity_step, bench)

    trainsize = 0.03 + float(fidelity_step / max_steps) * 0.97

    if suite_name == "rbv2_aknn":
        config = vector_to_config_(x, task_id, trainsize, impute="impute.mean")
    elif suite_name == "rbv2_rpart":
        config = vector_to_config__(x, task_id, trainsize, impute="impute.mean")
    else:
        raise ValueError(f"Unknown suite name: {suite_name}")

    res = bench.objective_function(config)
    return res[0]["acc"]


def estimate_epsilon(
    bench,
    task_id,
    n_samples: int,
    exploration: int,
    max_steps: int = 52,
    seed: int = 0,
) -> float:
    vecs = SobolEngine(get_suite_dim(bench), scramble=True, seed=seed).draw(n_samples).numpy()

    vals = np.empty(n_samples, dtype=np.float32)
    for i, vec in enumerate(vecs):
        vals[i] = get_acc(vec, task_id=task_id, fidelity_step=exploration, bench=bench, max_steps=max_steps)

    diff = vecs[:, None, :] - vecs[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=2)

    i_idx, j_idx = np.triu_indices(n_samples, k=1)
    v_i = vals[i_idx]
    v_j = vals[j_idx]
    v_g = np.maximum(v_i, v_j)
    v_f = np.minimum(v_i, v_j)
    d = dist_mat[i_idx, j_idx]

    mask = (d > 0) & (v_g > 0)
    eps_samples = (1 - v_f[mask] / v_g[mask]) / d[mask]

    eps_arr = eps_samples[np.isfinite(eps_samples) & (eps_samples > 0)]
    if eps_arr.size == 0:
        raise ValueError("No valid pairs to estimate epsilon.")

    return float(np.percentile(eps_arr, 95))
