from __future__ import annotations

import datetime
import itertools
import json
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from botorch import fit_gpytorch_mll
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.cost import AffineFidelityCostModel

from src.adacent_baselines import run_adacent, run_enhanced_adacent
from src.checkpointing import (
    CHECKPOINT_VERSION,
    atomic_save_checkpoint,
    capture_rng_state,
    config_signature,
    deserialize_run_state,
    load_checkpoint,
    make_checkpoint_path,
    restore_rng_state,
    serialize_run_state,
)
from src.model import init_model_srdk
from src.opt import bo_step_kg, bo_step_mes, rmfbo, rmfbo_pseudo, rmfbo_random
from src.test_functions import (
    LCBench,
    iaml_rpart,
    iaml_xgboost,
)
from src.utility import generate_initial_data, get_project, initialize_mf_model


PROJECT_ROOT = Path(__file__).resolve().parent

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
torch.set_printoptions(precision=3, sci_mode=False)


def _build_problem(config: DictConfig):
    negate = config.problem.negate
    scale = config.problem.scale

    if config.problem.name == "LCBench":
        instance = config.problem.instance if hasattr(config.problem, "instance") else "3945"
        problem = LCBench(negate=negate, instance=instance).to(**tkwargs)
        problem.scale = scale

    elif config.problem.name == "RPart":
        instance = config.problem.instance if hasattr(config.problem, "instance") else "1489"
        problem = iaml_rpart(negate=negate, instance=instance).to(**tkwargs)
        problem.scale = scale

    elif config.problem.name == "XGBoost":
        instance = config.problem.instance if hasattr(config.problem, "instance") else "1489"
        booster = config.problem.booster if hasattr(config.problem, "booster") else "dart"
        problem = iaml_xgboost(
            negate=negate,
            instance=instance,
            booster=booster,
        ).to(**tkwargs)
        problem.scale = scale

    else:
        raise NotImplementedError(f"Problem {config.problem.name} not implemented")

    return problem


def _allocate_record(config: DictConfig) -> np.ndarray:
    n_repeat = int(config.problem.n_repeat)

    if config.algorithm.name in ["AdaCent", "EnhancedAdaCent"]:
        min_eval_cost = float(config.problem.low_cost + min(config.problem.fidelities))
        max_record_len = int(np.ceil(config.problem.budget / min_eval_cost)) + 5
    else:
        max_record_len = int(config.problem.max_n_iter)

    return np.zeros((n_repeat, max_record_len, 2), dtype=float)


def _unique_ints(values):
    return sorted({max(1, int(round(v))) for v in values})


def _unique_probs(values):
    vals = [min(0.99, max(0.01, float(v))) for v in values]
    return sorted({round(v, 6) for v in vals})


def _unique_positive(values):
    vals = [max(1e-6, float(v)) for v in values]
    return sorted({round(v, 6) for v in vals})


def _make_adacent_grid(config: DictConfig):
    alg = config.algorithm

    grid = {
        "k": _unique_ints([alg.k - 10, alg.k, alg.k + 10]),
        "n_candidates": _unique_ints(
            [alg.n_candidates * 0.5, alg.n_candidates, alg.n_candidates * 1.5]
        ),
        "exploration": _unique_probs(
            [alg.exploration - 0.05, alg.exploration, alg.exploration + 0.05]
        ),
        "tail_x": _unique_probs(
            [alg.tail_x - 0.10, alg.tail_x, alg.tail_x + 0.10]
        ),
    }

    if alg.name == "EnhancedAdaCent":
        grid["epsilon"] = _unique_positive(
            [alg.epsilon * 0.5, alg.epsilon, alg.epsilon * 2.0]
        )

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [
        {key: value for key, value in zip(keys, combo)}
        for combo in itertools.product(*values)
    ]


def _clone_config_with_algorithm_params(config: DictConfig, params: dict) -> DictConfig:
    new_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    for key, value in params.items():
        new_config.algorithm[key] = value
    return new_config


def _mean_final_reward(record: np.ndarray) -> float:
    final_rewards = []
    for repeat_idx in range(record.shape[0]):
        used = np.flatnonzero(record[repeat_idx, :, 1] > 0)
        final_rewards.append(
            -np.inf if used.size == 0 else float(record[repeat_idx, used[-1], 0])
        )
    return float(np.mean(final_rewards))


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _update_record_row(
    record: np.ndarray,
    repeat_idx: int,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    cumulative_cost: list[float],
    config: DictConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    record[repeat_idx] = 0.0
    if not cumulative_cost:
        return None, None

    target_fidelity = torch.as_tensor(
        float(config.problem.target_fidelity),
        device=train_x.device,
        dtype=train_x.dtype,
    )
    target_mask = torch.isclose(train_x[:, -1], target_fidelity)
    fallback_value = (
        -float("inf") if bool(config.problem.negate) else float(config.problem.min_value)
    ) * float(config.problem.scale)

    maximum_reward = torch.where(
        target_mask.unsqueeze(-1),
        train_obj.detach(),
        fallback_value * torch.ones_like(train_obj),
    )
    maximum_reward = torch.cummax(maximum_reward, dim=0).values

    cumulative_cost_t = torch.cumsum(
        torch.as_tensor(cumulative_cost, **tkwargs),
        dim=0,
    )


    record_length = min(
        len(cumulative_cost),
        maximum_reward.shape[0],
        record.shape[1],
    )
    if record_length:
        rewards = maximum_reward[-record_length:].reshape(-1).detach().cpu().numpy()
        costs = cumulative_cost_t[-record_length:].reshape(-1).detach().cpu().numpy()
        record[repeat_idx, :record_length, 0] = rewards
        record[repeat_idx, :record_length, 1] = costs

    return maximum_reward, cumulative_cost_t


def _build_file_name(config: DictConfig, dk: bool) -> str:
    stopping_criteria = config.problem.stopping_criteria
    negate = config.problem.negate
    n_repeat = config.problem.n_repeat
    n_init = config.problem.n_init
    low_cost = config.problem.low_cost
    sf = config.problem.SF if hasattr(config.problem, "SF") else False
    n_iter = (
        config.problem.n_iter
        if stopping_criteria != "budget"
        else config.problem.max_n_iter
    )

    file_name = (
        f"{config.problem.name}"
        f"{'_NEG' if negate else ''}_"
        f"{config.algorithm.name}"
        f"{'-' + config.algorithm.acq_name if config.algorithm.name == 'RMFBO-PSEUDO' else ''}"
    )
    file_name += f"{'-DK' if dk else ''}"
    file_name += f"_R{n_repeat}_NI{n_init}_C{low_cost}"

    if stopping_criteria == "budget":
        file_name += f"_B{config.problem.budget}"
    else:
        file_name += f"_NI{n_iter}"

    if sf:
        file_name += "_SF"

    if (
        hasattr(config.algorithm, "find_best")
        and config.algorithm.find_best
        and config.algorithm.name in ["AdaCent", "EnhancedAdaCent"]
    ):
        file_name += "_BEST"

    return file_name


def _checkpoint_options(config: DictConfig) -> tuple[bool, bool, str]:
    if not hasattr(config, "checkpoint"):
        return True, False, "res/checkpoints"
    enabled = bool(config.checkpoint.enabled) if hasattr(config.checkpoint, "enabled") else True
    reset = bool(config.checkpoint.reset) if hasattr(config.checkpoint, "reset") else False
    directory = (
        str(config.checkpoint.directory)
        if hasattr(config.checkpoint, "directory")
        else "res/checkpoints"
    )
    return enabled, reset, directory


def _run_single_setting(
    config: DictConfig,
    problem,
    fidelities: torch.Tensor,
    cost_model,
    cost_aware_utility,
    project,
    *,
    show_progress: bool = True,
):
    stopping_criteria = config.problem.stopping_criteria
    n_init = int(config.problem.n_init)
    n_iter = int(
        config.problem.n_iter
        if stopping_criteria != "budget"
        else config.problem.max_n_iter
    )
    num_restarts = int(config.problem.num_restarts)
    raw_samples = int(config.problem.raw_samples)
    batch_size = int(config.problem.batch_size)
    min_value = float(config.problem.min_value)
    n_repeat = int(config.problem.n_repeat)
    dk = bool(config.algorithm.dk) if hasattr(config.algorithm, "dk") else False
    max_opt_iter = (
        int(config.algorithm.max_opt_iter)
        if hasattr(config.algorithm, "max_opt_iter")
        else 50
    )

    low_cost = float(config.problem.low_cost)
    fid_idx = problem.dim - 1
    target_fidelity = float(config.problem.target_fidelity)
    fixed_features_list = [{fid_idx: fid} for fid in fidelities]

    record = _allocate_record(config)
    file_name = _build_file_name(config, dk)
    signature = config_signature(config)
    checkpoint_enabled, reset_checkpoint, checkpoint_directory = _checkpoint_options(config)
    checkpoint_path = make_checkpoint_path(
        PROJECT_ROOT,
        file_name,
        config,
        directory=checkpoint_directory,
    )

    if checkpoint_enabled and reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    payload = load_checkpoint(checkpoint_path) if checkpoint_enabled else None
    if payload is not None:
        if payload.get("signature") != signature:
            raise ValueError(
                f"Checkpoint configuration mismatch: {checkpoint_path}. "
                "Run with checkpoint.reset=true to start over."
            )
        loaded_record = np.asarray(payload.get("record"))
        if loaded_record.shape != record.shape:
            raise ValueError(
                f"Checkpoint record shape {loaded_record.shape} does not match "
                f"the requested shape {record.shape}. Use checkpoint.reset=true."
            )
        record = loaded_record.copy()
        restore_rng_state(payload.get("rng_state"))
        if show_progress:
            status = payload.get("status", "running")
            completed = int(payload.get("next_repeat_idx", 0))
            tqdm.write(
                f"Checkpoint {status}: {config.problem.name}/{config.algorithm.name}, "
                f"completed repetitions={completed}/{n_repeat}"
            )
    else:
        payload = {
            "version": CHECKPOINT_VERSION,
            "signature": signature,
            "status": "running",
            "record": record,
            "next_repeat_idx": 0,
            "current_run": None,
            "last_run": None,
            "rng_state": capture_rng_state(),
        }
        if checkpoint_enabled:
            atomic_save_checkpoint(checkpoint_path, payload)

    def save_payload() -> None:
        payload["record"] = record
        payload["rng_state"] = capture_rng_state()
        if checkpoint_enabled:
            atomic_save_checkpoint(checkpoint_path, payload)

    if payload.get("status") == "complete":
        last_run = payload.get("last_run")
        if last_run is None:
            return record, None, None, None
        last_x, last_y, last_cost, _ = deserialize_run_state(last_run, tkwargs)
        maximum_reward, cumulative_cost_t = _update_record_row(
            record,
            int(last_run["repeat_idx"]),
            last_x,
            last_y,
            last_cost,
            config,
        )
        return record, last_x, maximum_reward, cumulative_cost_t

    current_run = payload.get("current_run")
    start_repeat_idx = int(payload.get("next_repeat_idx", 0))
    if current_run is not None:
        start_repeat_idx = int(current_run["repeat_idx"])

    last_train_x = None
    last_maximum_reward = None
    last_cumulative_cost = None

    repeat_iterator = tqdm(
        range(start_repeat_idx, n_repeat),
        desc=f"{config.algorithm.name} repetitions",
        unit="run",
        leave=False,
        disable=not show_progress,
    )

    for repeat_idx in repeat_iterator:
        resumed_this_repeat = (
            current_run is not None and int(current_run["repeat_idx"]) == repeat_idx
        )
        if resumed_this_repeat:
            train_x, train_obj, cumulative_cost, algorithm_state = deserialize_run_state(
                current_run,
                tkwargs,
            )
        else:
            algorithm_state = {}
            cumulative_cost = []
            if config.algorithm.name in ["AdaCent", "EnhancedAdaCent"]:
                train_x = torch.empty((0, problem.dim), **tkwargs)
                train_obj = torch.empty((0, 1), **tkwargs)
            elif config.algorithm.name in ["RMFBO", "RMFBO-RANDOM", "RMFBO-PSEUDO"]:


                train_x = None
                train_obj = None
            else:
                train_x, train_obj = generate_initial_data(
                    problem=problem,
                    fidelities=fidelities,
                    n=n_init,
                    **tkwargs,
                )

        def checkpoint_callback(
            current_x: torch.Tensor,
            current_y: torch.Tensor,
            current_cost: list[float],
            current_algorithm_state: dict | None,
        ) -> None:
            nonlocal last_train_x, last_maximum_reward, last_cumulative_cost
            maximum_reward, cumulative_cost_t = _update_record_row(
                record,
                repeat_idx,
                current_x,
                current_y,
                current_cost,
                config,
            )
            serialized = serialize_run_state(
                repeat_idx,
                current_x,
                current_y,
                current_cost,
                current_algorithm_state,
            )
            payload["status"] = "running"
            payload["next_repeat_idx"] = repeat_idx
            payload["current_run"] = serialized
            payload["last_run"] = serialized
            last_train_x = current_x
            last_maximum_reward = maximum_reward
            last_cumulative_cost = cumulative_cost_t
            save_payload()

        if config.algorithm.name == "RMFBO":
            train_x, train_obj, cumulative_cost, algorithm_state = rmfbo(
                problem,
                fidelities,
                config,
                tkwargs,
                train_x=train_x,
                train_obj=train_obj,
                cumulative_cost=cumulative_cost,
                algorithm_state=algorithm_state,
                checkpoint_callback=checkpoint_callback,
                show_progress=show_progress,
            )

        elif config.algorithm.name == "RMFBO-RANDOM":
            train_x, train_obj, cumulative_cost, algorithm_state = rmfbo_random(
                problem,
                fidelities,
                config,
                tkwargs,
                train_x=train_x,
                train_obj=train_obj,
                cumulative_cost=cumulative_cost,
                algorithm_state=algorithm_state,
                checkpoint_callback=checkpoint_callback,
                show_progress=show_progress,
            )

        elif config.algorithm.name == "RMFBO-PSEUDO":
            train_x, train_obj, cumulative_cost, algorithm_state = rmfbo_pseudo(
                problem,
                fidelities,
                config,
                tkwargs,
                train_x=train_x,
                train_obj=train_obj,
                cumulative_cost=cumulative_cost,
                algorithm_state=algorithm_state,
                checkpoint_callback=checkpoint_callback,
                show_progress=show_progress,
            )

        elif config.algorithm.name == "AdaCent":
            if not resumed_this_repeat:
                checkpoint_callback(train_x, train_obj, cumulative_cost, algorithm_state)
            train_x, train_obj, cumulative_cost, algorithm_state = run_adacent(
                problem,
                fidelities,
                config,
                tkwargs,
                train_x=train_x,
                train_obj=train_obj,
                cumulative_cost=cumulative_cost,
                algorithm_state=algorithm_state,
                checkpoint_callback=checkpoint_callback,
                show_progress=show_progress,
            )

        elif config.algorithm.name == "EnhancedAdaCent":
            if not resumed_this_repeat:
                checkpoint_callback(train_x, train_obj, cumulative_cost, algorithm_state)
            train_x, train_obj, cumulative_cost, algorithm_state = run_enhanced_adacent(
                problem,
                fidelities,
                config,
                tkwargs,
                train_x=train_x,
                train_obj=train_obj,
                cumulative_cost=cumulative_cost,
                algorithm_state=algorithm_state,
                checkpoint_callback=checkpoint_callback,
                show_progress=show_progress,
            )

        else:
            if train_x is None or train_obj is None:
                raise RuntimeError("Missing optimization state for a standard BO method.")

            start_iteration = int(
                algorithm_state.get("next_iteration", len(cumulative_cost))
            )
            if not resumed_this_repeat:
                checkpoint_callback(
                    train_x,
                    train_obj,
                    cumulative_cost,
                    {"next_iteration": start_iteration},
                )

            already_finished = (
                stopping_criteria == "budget"
                and sum(cumulative_cost) > float(config.problem.budget)
            )
            iteration_range = (
                range(n_iter, n_iter)
                if already_finished
                else range(start_iteration, n_iter)
            )
            opt_iterator = tqdm(
                iteration_range,
                desc=config.algorithm.name,
                unit="iter",
                leave=False,
                disable=not show_progress,
            )

            for iteration_idx in opt_iterator:
                if dk:
                    mll, model = init_model_srdk(
                        train_x,
                        train_obj,
                        training_iter=max_opt_iter,
                        verbose=False,
                    )
                else:
                    mll, model = initialize_mf_model(
                        train_x,
                        train_obj,
                        data_fidelity=fid_idx,
                    )

                fit_gpytorch_mll(mll)

                if config.algorithm.name == "KG":
                    new_x, new_obj, cost = bo_step_kg(
                        model,
                        cost_aware_utility,
                        project,
                        fixed_features_list,
                        problem,
                        cost_model,
                        batch_size,
                        num_restarts,
                        raw_samples,
                    )

                elif config.algorithm.name == "MES":
                    new_x, new_obj, cost = bo_step_mes(
                        model,
                        cost_aware_utility,
                        project,
                        fidelities,
                        fixed_features_list,
                        problem,
                        cost_model,
                        batch_size,
                        num_restarts,
                        raw_samples,
                        tkwargs=tkwargs,
                        space_sample_num=config.algorithm.space_sample_num,
                    )

                elif config.algorithm.name == "RAND":
                    new_x, new_obj = generate_initial_data(
                        problem=problem,
                        fidelities=fidelities,
                        n=1,
                        **tkwargs,
                    )
                    cost = low_cost + float(new_x[..., -1].item())

                else:
                    raise NotImplementedError(config.algorithm.name)

                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj.reshape(-1, 1)])
                cost_value = _as_float(cost)
                cumulative_cost.append(cost_value)

                target_mask = torch.isclose(
                    train_x[:, -1],
                    torch.as_tensor(target_fidelity, **tkwargs),
                )
                best_obs = (
                    train_obj[target_mask].max().item()
                    if bool(target_mask.any().item())
                    else min_value
                )
                opt_iterator.set_postfix_str(
                    f"best={best_obs:.4f}, fid={new_x[0, -1].item():.2f}, "
                    f"budget={sum(cumulative_cost):.2f}/{float(config.problem.budget):g}"
                )

                algorithm_state = {"next_iteration": iteration_idx + 1}
                checkpoint_callback(
                    train_x,
                    train_obj,
                    cumulative_cost,
                    algorithm_state,
                )

                if (
                    stopping_criteria == "budget"
                    and sum(cumulative_cost) > float(config.problem.budget)
                ):
                    break

        if train_x is None or train_obj is None:
            raise RuntimeError("Algorithm returned no optimization state.")

        last_maximum_reward, last_cumulative_cost = _update_record_row(
            record,
            repeat_idx,
            train_x,
            train_obj,
            cumulative_cost,
            config,
        )
        final_run = serialize_run_state(
            repeat_idx,
            train_x,
            train_obj,
            cumulative_cost,
            algorithm_state,
        )
        payload["current_run"] = None
        payload["last_run"] = final_run
        payload["next_repeat_idx"] = repeat_idx + 1
        payload["status"] = "complete" if repeat_idx + 1 >= n_repeat else "running"
        last_train_x = train_x
        save_payload()
        current_run = None

    return record, last_train_x, last_maximum_reward, last_cumulative_cost


def _maybe_plot_results(
    config: DictConfig,
    file_name: str,
    train_x,
    maximum_reward,
    cumulative_cost,
):
    if not bool(config.problem.plot_results):
        return
    if train_x is None or maximum_reward is None or cumulative_cost is None:
        return

    output_dir = PROJECT_ROOT / "res" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(train_x.size(0))), maximum_reward.detach().cpu().numpy())
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title("Rewards")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        list(range(cumulative_cost.size(0))),
        cumulative_cost.detach().cpu().numpy(),
    )
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cumulative Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"test_{file_name}.png", dpi=200)
    plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="experiment")
def experiment(config: DictConfig):
    problem = _build_problem(config)

    sf = config.problem.SF if hasattr(config.problem, "SF") else False
    dk = bool(config.algorithm.dk) if hasattr(config.algorithm, "dk") else False

    low_cost = float(config.problem.low_cost)
    fid_idx = problem.dim - 1
    fidelities = torch.tensor(
        config.problem.fidelities if not sf else [config.problem.fidelities[-1]],
        **tkwargs,
    )
    target_fidelities = {fid_idx: config.problem.target_fidelity}

    cost_model = AffineFidelityCostModel(
        fidelity_weights={fid_idx: 1.0},
        fixed_cost=low_cost,
    )
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    project = get_project(target_fidelities)

    find_best = (
        bool(config.algorithm.find_best)
        if hasattr(config.algorithm, "find_best")
        else False
    )
    do_grid_search = find_best and config.algorithm.name in [
        "AdaCent",
        "EnhancedAdaCent",
    ]

    best_params = None
    best_score = None

    if do_grid_search:
        param_grid = _make_adacent_grid(config)
        best_result = None
        best_config = None
        best_score_value = -float("inf")

        grid_progress = tqdm(
            param_grid,
            desc=f"{config.algorithm.name} grid search",
            unit="setting",
            dynamic_ncols=True,
        )
        for params in grid_progress:
            trial_config = _clone_config_with_algorithm_params(config, params)
            result = _run_single_setting(
                trial_config,
                problem,
                fidelities,
                cost_model,
                cost_aware_utility,
                project,
                show_progress=False,
            )
            score = _mean_final_reward(result[0])

            if best_result is None or score > best_score_value:
                best_score_value = score
                best_result = result
                best_params = params
                best_config = trial_config

            grid_progress.set_postfix_str(f"best={best_score_value:.6f}")

        if best_result is None or best_config is None:
            raise RuntimeError("AdaCent grid search produced no result.")

        record, train_x, maximum_reward, cumulative_cost = best_result
        config_to_save = best_config
        best_score = best_score_value
        print(f"Best params for {config.algorithm.name}: {best_params}")
        print(f"Best mean final reward: {best_score_value:.8f}")

    else:
        record, train_x, maximum_reward, cumulative_cost = _run_single_setting(
            config,
            problem,
            fidelities,
            cost_model,
            cost_aware_utility,
            project,
            show_progress=True,
        )
        config_to_save = config

    file_name = _build_file_name(config_to_save, dk)
    _maybe_plot_results(
        config_to_save,
        file_name,
        train_x,
        maximum_reward,
        cumulative_cost,
    )

    hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    hydra_dir.mkdir(parents=True, exist_ok=True)
    np.save(hydra_dir / f"{file_name}.npy", record)

    records_dir = PROJECT_ROOT / "res" / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = records_dir / f"{file_name}_{now}.npy"
    np.save(save_path, record)

    if best_params is not None:
        metadata_path = records_dir / f"{file_name}_{now}_best_params.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "algorithm": config.algorithm.name,
                    "best_params": best_params,
                    "best_mean_final_reward": best_score,
                },
                indent=2,
            )
        )

    print(f"Results saved to: {save_path}")


if __name__ == "__main__":
    experiment()
