from __future__ import annotations

import contextlib
import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import ConfigSpace as CS

logging.getLogger("hpbandster").setLevel(logging.ERROR)
logging.getLogger("Pyro4").setLevel(logging.ERROR)
logging.getLogger("smac").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────
# Helper: detect fidelity, target & “integer scale” once
# ──────────────────────────────────────────────────────────────
def _fidelity_info(bench) -> tuple[str, bool]:
    """Return (fidelity_param_id, on_integer_scale)."""
    fid_space = bench.get_fidelity_space()
    if "rbv2_" in bench.config.config_id:
        fid_param = "trainsize"
    else:
        fid_param = fid_space.get_hyperparameter_names()[0]
    hp = fid_space.get_hyperparameter(fid_param)
    on_int = isinstance(hp, (CS.UniformIntegerHyperparameter, CS.OrdinalHyperparameter))
    return fid_param, on_int


def _default_target(bench) -> str:
    return "val_accuracy" if "lcbench" in bench.config.config_id else "acc"


def _spent_from_budget(bench, fidelity_param_id: str, budget: float) -> int:
    if fidelity_param_id == "epoch" or "lcbench" in bench.config.config_id:
        return max(1, int(round(float(budget))))

    steps = (float(budget) - 0.03) / 0.97 * 52.0
    return max(1, int(round(steps)))


def _build_trace_from_archive(bench, fidelity_param_id: str,
                              target: str | None,
                              minimize: bool) -> List[Tuple[int, float]]:
    target = _default_target(bench) if target is None else target
    spent, best = 0, (np.inf if minimize else -np.inf)
    trace: list[tuple[int, float]] = []

    for row in bench.archive:
        budget = row["x"][fidelity_param_id]
        y = row["y"]
        if target not in y:
            fallback = _default_target(bench)
            if fallback in y:
                metric = y[fallback]
            else:
                raise KeyError(
                    f"Metric '{target}' not found in archive row; available keys: {list(y.keys())}"
                )
        else:
            metric = y[target]

        spent += _spent_from_budget(bench, fidelity_param_id, budget)
        best = min(best, metric) if minimize else max(best, metric)
        trace.append((spent, best))

    return trace


# ──────────────────────────────────────────────────────────────
# Random Search
# ──────────────────────────────────────────────────────────────
def random_search(bench,
                  task_id: str,
                  *,
                  target: str | None = None,
                  minimize: bool = False,
                  n_trials: int = 32,
                  seed: int | None = None) -> List[Tuple[int, float]]:
    random.seed(seed)
    np.random.seed(seed)
    target = _default_target(bench) if target is None else target

    fid_param, on_int = _fidelity_info(bench)
    max_budget = bench.get_fidelity_space().get_hyperparameter(fid_param).upper

    bench.archive = []

    opt_space = bench.get_opt_space(task_id)
    opt_space.seed(seed)

    for _ in range(n_trials):
        cfg = opt_space.sample_configuration().get_dictionary()
        if "rbv2_" in bench.config.config_id:
            cfg.update({"repl": 10})
        cfg[fid_param] = int(max_budget) if on_int else max_budget
        bench.objective_function(cfg, logging=True, multithread=False)

    return _build_trace_from_archive(
        bench, fidelity_param_id=fid_param,
        target=target, minimize=minimize,
    )


# ──────────────────────────────────────────────────────────────
# BOHB & Hyperband
# ──────────────────────────────────────────────────────────────
def _hb_or_bohb(bench,
                task_id: str,
                *,
                optimizer_name: str,
                n_iterations: int,
                target: str | None,
                minimize: bool,
                seed: int | None) -> List[Tuple[int, float]]:
    from hpbandster.core.worker import Worker
    from hpbandster.core.nameserver import NameServer
    from hpbandster.optimizers import BOHB as _BOHB, HyperBand as _HB

    class _YahpoWorker(Worker):
        def __init__(self, bench, fidelity_param_id, target, minimize, on_integer_scale, **kwargs):
            super().__init__(**kwargs)
            self._bench = bench
            self._fid = fidelity_param_id
            self._target = target
            self._factor = 1 if minimize else -1
            self._on_int = on_integer_scale
            self.sleep_interval = 1e-2

        def compute(self, config, budget, **_):
            cfg = config.get_dictionary().copy() if hasattr(config, "get_dictionary") else dict(config)
            if "rbv2_" in self._bench.config.config_id:
                cfg["repl"] = 10
            cfg[self._fid] = int(round(budget)) if self._on_int else budget
            y = self._bench.objective_function(cfg, logging=True, multithread=False)[0]
            return {
                "loss": self._factor * float(y[self._target]),
                "info": {"budget": cfg[self._fid]},
            }

    target = _default_target(bench) if target is None else target
    optimizer_cls = _BOHB if optimizer_name == "bohb" else _HB

    random.seed(seed)
    np.random.seed(seed)
    fid_param, on_int = _fidelity_info(bench)
    min_b = bench.get_fidelity_space().get_hyperparameter(fid_param).lower
    max_b = bench.get_fidelity_space().get_hyperparameter(fid_param).upper
    randport = random.randrange(55536, 65535)

    bench.archive = []

    ns = NameServer(run_id="yahpo", host="127.0.0.1", port=randport)
    ns.start()

    w = _YahpoWorker(
        bench=bench,
        fidelity_param_id=fid_param,
        target=target,
        minimize=minimize,
        on_integer_scale=on_int,
        nameserver="127.0.0.1",
        nameserver_port=randport,
        run_id="yahpo",
    )
    w.run(background=True)

    cs = bench.get_opt_space(task_id)
    cs.seed(seed)

    opt_kwargs = dict(
        configspace=cs,
        eta=3,
        run_id="yahpo",
        nameserver="127.0.0.1",
        nameserver_port=randport,
        min_budget=min_b,
        max_budget=max_b,
    )
    opt = optimizer_cls(**opt_kwargs)
    with open(os.devnull, "w") as devnull,          contextlib.redirect_stdout(devnull),          contextlib.redirect_stderr(devnull):
        opt.run(n_iterations=n_iterations)

    opt.shutdown()
    w.shutdown()
    ns.shutdown()

    return _build_trace_from_archive(bench, fid_param, target, minimize)


def bohb_search(bench,
                task_id: str,
                *,
                target: str | None = None,
                minimize: bool = False,
                n_iterations: int = 16,
                seed: int | None = None) -> List[Tuple[int, float]]:
    return _hb_or_bohb(
        bench,
        task_id,
        optimizer_name="bohb",
        n_iterations=n_iterations,
        target=target,
        minimize=minimize,
        seed=seed,
    )


def hyperband(bench,
              task_id: str,
              *,
              target: str | None = None,
              minimize: bool = False,
              n_iterations: int = 16,
              seed: int | None = None) -> List[Tuple[int, float]]:
    return _hb_or_bohb(
        bench,
        task_id,
        optimizer_name="hyperband",
        n_iterations=n_iterations,
        target=target,
        minimize=minimize,
        seed=seed,
    )


# ──────────────────────────────────────────────────────────────
# SMAC-HPO (single-fidelity, full budget)
# ──────────────────────────────────────────────────────────────
def smac_search(bench,
                task_id: str,
                *,
                target: str | None = None,
                minimize: bool = False,
                n_trials: int = 32,
                seed: int | None = None) -> List[Tuple[int, float]]:
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_hpo_facade import SMAC4HPO

    target = _default_target(bench) if target is None else target
    random.seed(seed)
    np.random.seed(seed)

    fid_param, on_int = _fidelity_info(bench)
    max_b = bench.get_fidelity_space().get_hyperparameter(fid_param).upper
    factor = 1 if minimize else -1
    opt_space = bench.get_opt_space(task_id)
    tmp_dir = Path(f"smac_hpo_tmp_{seed}_{random.randrange(49152,65535)}")

    bench.archive = []

    scenario = Scenario({
        "run_obj": "quality",
        "runcount-limit": n_trials,
        "cs": opt_space,
        "deterministic": "true",
        "output_dir": str(tmp_dir),
    })

    def tae_full_budget(cfg):
        cfg = cfg.get_dictionary().copy() if hasattr(cfg, "get_dictionary") else dict(cfg)
        if "rbv2_" in bench.config.config_id:
            cfg["repl"] = 10
        cfg[fid_param] = int(max_b) if on_int else max_b

        y = bench.objective_function(cfg, logging=True, multithread=False)[0]
        return factor * float(y[target])

    smac = SMAC4HPO(
        scenario=scenario,
        rng=np.random.RandomState(seed),
        tae_runner=tae_full_budget,
    )
    smac.optimize()

    out = _build_trace_from_archive(
        bench,
        fidelity_param_id=fid_param,
        target=target,
        minimize=minimize,
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out
