import torch
import gpytorch
import numpy as np
from tqdm import tqdm

from botorch import fit_gpytorch_mll
from torch.quasirandom import SobolEngine
from botorch.acquisition import PosteriorMean
from botorch.acquisition.max_value_entropy_search import qMultiFidelityMaxValueEntropy
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_discrete
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.utils.transforms import unnormalize

from src.utility import generate_initial_data, initialize_mf_model, get_project
from src import (
    convert_to_nn_and_base_kernel,
    monte_carlo_entropy,
    rbf_kernel_variance_reduction_rate,
    excessive_risk_reduction_rate,
)
from src.model import init_model_srdk
from src.acquisition import (
    qLowerConfidenceBound,
    qConstrainedCI_m_UCB,
    qConfidenceInterval,
    qUpperConfidenceBound,
)


def _count_true(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _has_any(mask: torch.Tensor) -> bool:
    return bool(mask.any().item())


def _fid_eq(values: torch.Tensor, target) -> torch.Tensor:
    target_tensor = torch.as_tensor(target, device=values.device, dtype=values.dtype)
    return torch.isclose(values, target_tensor)


def _fid_index(fidelities: torch.Tensor, fidelity_value) -> int:
    matches = _fid_eq(fidelities, fidelity_value).nonzero(as_tuple=False)
    if matches.numel() == 0:
        raise ValueError(
            f"Fidelity value {float(fidelity_value)} not found in fidelities="
            f"{[float(fid) for fid in fidelities.detach().cpu().reshape(-1)]}"
        )
    return int(matches[0].item())


def _append_obj(train_obj: torch.Tensor, new_obj: torch.Tensor) -> torch.Tensor:
    return torch.cat((train_obj, new_obj.reshape(-1, 1)), dim=0)


def _safe_max_target(train_x: torch.Tensor,
                     train_obj: torch.Tensor,
                     target_fidelity,
                     min_value: float) -> float:
    target_mask = _fid_eq(train_x[:, -1], target_fidelity)
    if _has_any(target_mask):
        return train_obj[target_mask].max().item()
    return min_value


def get_mfkg(model, problem, cost_aware_utility, project):

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=problem.dim,
        columns=[problem.dim - 1],
        values=[1],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=problem.bounds[:, :-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_mfacq_and_get_observation(
    mfkg_acqf,
    fixed_features_list,
    problem,
    cost_model,
    batch_size,
    num_restarts,
    raw_samples,
):

    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=problem.bounds,
        fixed_features_list=fixed_features_list,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )

    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).reshape(-1, 1)
    return new_x, new_obj, cost


def bo_step_kg(model,
               cost_aware_utility,
               project,
               fixed_features_list,
               problem,
               cost_model,
               batch_size,
               num_restarts,
               raw_samples):
    mfkg_acqf = get_mfkg(model, problem, cost_aware_utility, project)
    new_x, new_obj, cost = optimize_mfacq_and_get_observation(
        mfkg_acqf,
        fixed_features_list,
        problem,
        cost_model,
        batch_size,
        num_restarts,
        raw_samples,
    )
    return new_x, new_obj, cost


def get_mfMES(model,
              problem,
              fidelities,
              cost_aware_utility,
              project,
              sample_n: int = 40000,
              tkwargs: dict | None = None):
    tkwargs = {} if tkwargs is None else tkwargs
    cand_x_full, _ = generate_initial_data(
        problem=problem,
        fidelities=fidelities,
        n=sample_n,
        **tkwargs,
    )
    return qMultiFidelityMaxValueEntropy(
        model=model,
        candidate_set=cand_x_full,
        project=project,
        cost_aware_utility=cost_aware_utility,
    )


def bo_step_mes(model,
                cost_aware_utility,
                project,
                fidelities,
                fixed_features_list,
                problem,
                cost_model,
                batch_size,
                num_restarts,
                raw_samples,
                tkwargs: dict | None = None,
                **kwargs):
    tkwargs = {} if tkwargs is None else tkwargs
    mf_acqf = get_mfMES(
        model,
        problem,
        fidelities,
        cost_aware_utility,
        project,
        sample_n=kwargs.get('space_sample_num', 40000),
        tkwargs=tkwargs,
    )
    new_x, new_obj, cost = optimize_mfacq_and_get_observation(
        mf_acqf,
        fixed_features_list,
        problem,
        cost_model,
        batch_size,
        num_restarts,
        raw_samples,
    )
    return new_x, new_obj, cost


def rmfbo(
    problem,
    fidelities,
    config,
    tkwargs,
    *,
    train_x=None,
    train_obj=None,
    cumulative_cost=None,
    algorithm_state=None,
    checkpoint_callback=None,
    show_progress=True,
):

    max_cholesky_size = float("inf")
    stopping_criteria = config.problem.stopping_criteria
    n_iter = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    n_init, low_cost, \
    MIN_VALUE, \
    batch_size, target_fidelity = \
    config.problem.n_init, config.problem.low_cost, \
    config.problem.min_value, \
    config.problem.batch_size, float(config.problem.target_fidelity)

    max_opt_iter, batch_limit = config.algorithm.max_opt_iter, config.algorithm.batch_limit
    bo_beta, filter_beta, rate_beta, rbf_beta = (
        config.algorithm.bo_beta,
        config.algorithm.filter_beta,
        config.algorithm.rate_beta,
        config.algorithm.rbf_beta,
    )
    space_sample_num, model_sample_num, sample_train_num = (
        config.algorithm.space_sample_num,
        config.algorithm.model_sample_num,
        config.algorithm.sample_train_num,
    )

    costs = fidelities + low_cost

    state = algorithm_state or {}
    if train_x is None or train_obj is None:
        train_x, train_obj = generate_initial_data(
            problem=problem,
            fidelities=fidelities,
            n=n_init,
            **tkwargs,
        )
    cumulative_cost = [] if cumulative_cost is None else list(cumulative_cost)
    _max_lcb = float(state.get("max_lcb", float("-inf")))

    if "sobol_seed" in state:
        sobol_seed = int(state["sobol_seed"])
    else:
        sobol_seed = int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())
    sobol_drawn = int(state.get("sobol_drawn", 0))
    sobol_eng = SobolEngine(
        dimension=problem.dim - 1,
        scramble=True,
        seed=sobol_seed,
    )
    if sobol_drawn:
        sobol_eng.fast_forward(sobol_drawn)

    start_iteration = int(state.get("next_iteration", len(cumulative_cost)))
    if checkpoint_callback is not None and start_iteration == 0:
        checkpoint_callback(
            train_x,
            train_obj,
            cumulative_cost,
            {
                "next_iteration": 0,
                "max_lcb": _max_lcb,
                "sobol_seed": sobol_seed,
                "sobol_drawn": sobol_drawn,
            },
        )

    already_finished = (
        stopping_criteria == 'budget'
        and sum(cumulative_cost) > float(config.problem.budget)
    )
    iteration_range = range(n_iter, n_iter) if already_finished else range(start_iteration, n_iter)
    bo_iterator = tqdm(
        iteration_range,
        desc="RMFBO",
        leave=False,
        disable=not show_progress,
    )
    for r_idx in bo_iterator:

        _, model = init_model_srdk(train_x, train_obj, training_iter=max_opt_iter, verbose=False)

        x_candidates, _ = generate_initial_data(
            problem=problem,
            fidelities=fidelities,
            n=space_sample_num,
            **tkwargs,
        )
        target_fid_filter = _fid_eq(x_candidates[..., -1], target_fidelity)

        attempts = 0
        while _count_true(target_fid_filter) == 0:
            attempts += 1
            if attempts >= 10:
                raise RuntimeError(
                    f"No candidates generated at target_fidelity={target_fidelity}. "
                    f"Available fidelities={[float(fid) for fid in fidelities.detach().cpu().reshape(-1)]}"
                )
            x_candidates, _ = generate_initial_data(
                problem=problem,
                fidelities=fidelities,
                n=space_sample_num,
                **tkwargs,
            )
            target_fid_filter = _fid_eq(x_candidates[..., -1], target_fidelity)

        base_model, projected_x = convert_to_nn_and_base_kernel(model, x_candidates)
        mc_f_lcb = qLowerConfidenceBound(base_model, beta=bo_beta)
        mc_f_ci = qConfidenceInterval(base_model, beta=bo_beta)


        lengthscale_list = []
        for _ in range(model_sample_num):
            _, tmp_model = init_model_srdk(
                train_x,
                train_obj,
                training_iter=sample_train_num,
                lr=1e-2,
                power_iterations=10,
                verbose=False,
            )
            tmp_lengthscale = tmp_model.covar_module.base_kernel.lengthscale.detach().clone().mean()
            lengthscale_list.append(tmp_lengthscale)

        int_lengthscale_list = torch.stack(
            [torch.ceil(lengthscale * 1e3) / 1e3 for lengthscale in lengthscale_list],
            dim=0,
        )
        lengthscale_entropy = monte_carlo_entropy(int_lengthscale_list)


        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            _, _tmp_max_lcb = optimize_acqf_discrete(
                acq_function=mc_f_lcb,
                q=1,
                choices=projected_x[target_fid_filter],
            )
            _, _tmp_max_ci = optimize_acqf_discrete(
                acq_function=mc_f_ci,
                q=1,
                choices=projected_x[target_fid_filter],
            )

        _max_lcb = max(_max_lcb, _tmp_max_lcb.item())
        _tmp_max_ci = torch.maximum(_tmp_max_ci, 1e-1 * torch.ones(1, **tkwargs))

        target_mask = _fid_eq(train_x[..., -1], target_fidelity)
        if _has_any(target_mask):
            _max_lcb = min(_max_lcb, train_obj[target_mask].max().item())

        model_list = [base_model]
        threshold_list = [_max_lcb]

        mc_f = qConstrainedCI_m_UCB(
            model_list,
            threshold_list,
            beta=bo_beta,
            filter_beta=filter_beta,
            return_UCB=True,
            constrained=False,
            sample_num=config.algorithm.sample_num,
        )


        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            if _has_any(target_mask):
                center_idx = train_obj[target_mask].argmax().item()
                center = train_x[target_mask][center_idx]
                x_distances = torch.linalg.norm(center - x_candidates, ord=2, dim=-1)
                diameter_filter = x_distances < torch.quantile(
                    x_distances,
                    config.algorithm.distance_quantile,
                )
                x_cand_filter = torch.logical_and(diameter_filter, target_fid_filter)
                if _count_true(x_cand_filter) == 0:
                    x_cand_filter = target_fid_filter
            else:
                x_cand_filter = target_fid_filter

            try:
                _choices = x_cand_filter if _count_true(x_cand_filter) > 0 else target_fid_filter
                f_X_next, _ = optimize_acqf_discrete(
                    acq_function=mc_f,
                    q=1,
                    choices=projected_x[_choices],
                    options={"batch_limit": batch_limit, "maxiter": max_opt_iter},
                )
            except Exception:
                _choices = target_fid_filter
                rand_idx = torch.randint(
                    0,
                    _count_true(_choices),
                    (1,),
                    device=projected_x.device,
                )
                f_X_next = projected_x[_choices][rand_idx]

            z_next = f_X_next.clone()
            match_mask = torch.abs(projected_x[_choices] - z_next).sum(dim=-1) == 0
            X_next = x_candidates[_choices][match_mask]
            _portion = _count_true(_choices) / space_sample_num


        _t = train_x.size(0)
        _t_fids = torch.tensor(
            [_count_true(_fid_eq(train_x[:, -1], fid)) for fid in fidelities],
            device=fidelities.device,
            dtype=tkwargs["dtype"],
        )
        rbf_rate = rbf_kernel_variance_reduction_rate(
            T=_t,
            dim=problem.dim,
            variance=_tmp_max_ci,
        ) * rbf_beta
        cost_aware_rbf_rate = rbf_rate / costs[-1]
        cost_aware_learning_rate = torch.sqrt(lengthscale_entropy) * torch.tensor(
            [
                excessive_risk_reduction_rate(
                    fidelity_counts=_t_fids,
                    fidelity_choosen=i,
                    beta=rate_beta,
                ) / costs[i]
                for i in range(len(fidelities))
            ],
            device=fidelities.device,
            dtype=tkwargs["dtype"],
        )

        if len(X_next) > batch_size:
            X_next = X_next[:batch_size]

        if cost_aware_rbf_rate > torch.max(cost_aware_learning_rate):
            X_next = X_next.reshape(batch_size, problem.dim)
        else:
            _fid_next = int(torch.argmax(cost_aware_learning_rate).item())
            _X_next = sobol_eng.draw(batch_size).to(**tkwargs)
            sobol_drawn += batch_size
            _X_next = unnormalize(_X_next, problem.bounds[:, :-1])
            _fid_col = fidelities[_fid_next].reshape(1, 1).repeat(batch_size, 1)
            X_next = torch.cat([_X_next, _fid_col], dim=-1)


        Y_next = problem(X_next).reshape(-1, 1)
        fid_idx = _fid_index(fidelities, X_next[0, -1])
        cost = costs[fid_idx]


        train_x = torch.cat((train_x, X_next), dim=0)
        train_obj = _append_obj(train_obj, Y_next)


        cumulative_cost.append(float(cost.item()))
        best_obs = _safe_max_target(train_x, train_obj, target_fidelity, MIN_VALUE)

        info = f"Max: {best_obs:.2e}"
        info += f" | Cost: {cost.item():.2e}"
        info += f" | new_fid: {X_next[0][-1].item():.2f}"
        info += f" | new_y: {Y_next[0].item():.2f}"
        info += f" | CI: {_tmp_max_ci.item():.2f}"
        info += f" | LCB: {_tmp_max_lcb.item():.2f}"
        info += f" | Portion: {_portion:.2e}"
        info += f" | RBF: {cost_aware_rbf_rate.item():.2f}"
        info += f" | RRate: {torch.max(cost_aware_learning_rate).item():.2e}"
        info += f" | Cost/Budget: {sum(cumulative_cost):.2f}/{config.problem.budget:d}"
        bo_iterator.set_postfix_str(info)

        current_state = {
            "next_iteration": r_idx + 1,
            "max_lcb": _max_lcb,
            "sobol_seed": sobol_seed,
            "sobol_drawn": sobol_drawn,
        }
        if checkpoint_callback is not None:
            checkpoint_callback(train_x, train_obj, cumulative_cost, current_state)


        if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
            break

    return train_x, train_obj, cumulative_cost, {
        "next_iteration": len(cumulative_cost),
        "max_lcb": _max_lcb,
        "sobol_seed": sobol_seed,
        "sobol_drawn": sobol_drawn,
    }


def rmfbo_random(
    problem,
    fidelities,
    config,
    tkwargs,
    *,
    train_x=None,
    train_obj=None,
    cumulative_cost=None,
    algorithm_state=None,
    checkpoint_callback=None,
    show_progress=True,
):

    max_cholesky_size = float("inf")
    stopping_criteria = config.problem.stopping_criteria
    n_iter = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    n_init, low_cost, \
    MIN_VALUE, \
    batch_size, target_fidelity = \
    config.problem.n_init, config.problem.low_cost, \
    config.problem.min_value, \
    config.problem.batch_size, config.problem.target_fidelity

    max_opt_iter, batch_limit = config.algorithm.max_opt_iter, config.algorithm.batch_limit
    bo_beta, filter_beta, rate_beta, rbf_beta = (
        config.algorithm.bo_beta,
        config.algorithm.filter_beta,
        config.algorithm.rate_beta,
        config.algorithm.rbf_beta,
    )
    space_sample_num, model_sample_num, sample_train_num = (
        config.algorithm.space_sample_num,
        config.algorithm.model_sample_num,
        config.algorithm.sample_train_num,
    )

    costs = fidelities + low_cost

    state = algorithm_state or {}
    if train_x is None or train_obj is None:
        train_x, train_obj = generate_initial_data(
            problem=problem,
            fidelities=fidelities,
            n=n_init,
            **tkwargs,
        )
    cumulative_cost = [] if cumulative_cost is None else list(cumulative_cost)
    start_iteration = int(state.get("next_iteration", len(cumulative_cost)))
    if checkpoint_callback is not None and start_iteration == 0:
        checkpoint_callback(train_x, train_obj, cumulative_cost, {"next_iteration": 0})

    already_finished = (
        stopping_criteria == 'budget'
        and sum(cumulative_cost) > float(config.problem.budget)
    )
    iteration_range = range(n_iter, n_iter) if already_finished else range(start_iteration, n_iter)
    bo_iterator = tqdm(
        iteration_range,
        desc="RMFBO-RANDOM",
        leave=False,
        disable=not show_progress,
    )
    for r_idx in bo_iterator:

        _, model = init_model_srdk(train_x, train_obj, training_iter=max_opt_iter, verbose=False)

        x_candidates, _ = generate_initial_data(
            problem=problem,
            fidelities=fidelities,
            n=space_sample_num,
            **tkwargs,
        )
        target_fid_filter = _fid_eq(x_candidates[..., -1], target_fidelity)

        attempts = 0
        while _count_true(target_fid_filter) == 0:
            attempts += 1
            if attempts >= 10:
                raise RuntimeError(
                    f"No candidates generated at target_fidelity={target_fidelity}. "
                    f"Available fidelities={[float(fid) for fid in fidelities.detach().cpu().reshape(-1)]}"
                )
            x_candidates, _ = generate_initial_data(
                problem=problem,
                fidelities=fidelities,
                n=space_sample_num,
                **tkwargs,
            )
            target_fid_filter = _fid_eq(x_candidates[..., -1], target_fidelity)

        base_model, projected_x = convert_to_nn_and_base_kernel(model, x_candidates)
        mc_f_ci = qConfidenceInterval(base_model, beta=bo_beta)
        mc_f_ucb = qUpperConfidenceBound(base_model, beta=bo_beta)
        mc_f_lcb = qLowerConfidenceBound(base_model, beta=bo_beta)


        lengthscale_list = []
        for _ in range(model_sample_num):
            _, tmp_model = init_model_srdk(
                train_x,
                train_obj,
                training_iter=sample_train_num,
                lr=1e-2,
                power_iterations=10,
                verbose=False,
            )
            tmp_lengthscale = tmp_model.covar_module.base_kernel.lengthscale.detach().clone().mean()
            lengthscale_list.append(tmp_lengthscale)

        int_lengthscale_list = torch.stack(
            [torch.ceil(lengthscale * 1e3) / 1e3 for lengthscale in lengthscale_list],
            dim=0,
        )
        lengthscale_entropy = monte_carlo_entropy(int_lengthscale_list)


        diameter_filter = torch.ones(
            space_sample_num,
            dtype=torch.bool,
            device=x_candidates.device,
        )

        target_mask = _fid_eq(train_x[..., -1], target_fidelity)
        if _has_any(target_mask):
            center_idx = train_obj[target_mask].argmax().item()
            center = train_x[target_mask][center_idx]
            x_distances = torch.linalg.norm(center - x_candidates, ord=2, dim=-1)
            diameter_filter = x_distances < torch.quantile(
                x_distances,
                config.algorithm.distance_quantile,
            )
            x_cand_filter = torch.logical_and(diameter_filter, target_fid_filter)
            if _count_true(x_cand_filter) == 0:
                x_cand_filter = target_fid_filter
        else:
            x_cand_filter = target_fid_filter
            if _count_true(x_cand_filter) == 0:
                x_cand_filter = target_fid_filter

        _portion = _count_true(x_cand_filter) / space_sample_num
        rand_idx = torch.randint(
            0,
            _count_true(x_cand_filter),
            (batch_size,),
            device=x_candidates.device,
        )
        X_next = x_candidates[x_cand_filter][rand_idx]


        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            _, _tmp_max_ci = optimize_acqf_discrete(
                acq_function=mc_f_ci,
                q=1,
                choices=projected_x[target_fid_filter],
            )
        _tmp_max_ci = torch.maximum(_tmp_max_ci, 1e-1 * torch.ones(1, **tkwargs))

        _t = train_x.size(0)
        _t_fids = torch.tensor(
            [_count_true(_fid_eq(train_x[:, -1], fid)) for fid in fidelities],
            device=fidelities.device,
            dtype=tkwargs["dtype"],
        )
        rbf_rate = rbf_kernel_variance_reduction_rate(
            T=_t,
            dim=problem.dim,
            variance=_tmp_max_ci,
        ) * rbf_beta
        cost_aware_rbf_rate = rbf_rate / costs[-1]
        cost_aware_learning_rate = torch.sqrt(lengthscale_entropy) * torch.tensor(
            [
                excessive_risk_reduction_rate(
                    fidelity_counts=_t_fids,
                    fidelity_choosen=i,
                    beta=rate_beta,
                ) / costs[i]
                for i in range(len(fidelities))
            ],
            device=fidelities.device,
            dtype=tkwargs["dtype"],
        )

        if cost_aware_rbf_rate > torch.max(cost_aware_learning_rate):
            X_next = X_next.reshape(batch_size, problem.dim)
        else:
            if _count_true(diameter_filter) == 0:
                rand_idx = torch.randint(0, space_sample_num, (batch_size,), device=x_candidates.device)
                X_next = x_candidates[rand_idx]
            else:
                rand_idx = torch.randint(
                    0,
                    _count_true(diameter_filter),
                    (batch_size,),
                    device=x_candidates.device,
                )
                X_next = x_candidates[diameter_filter][rand_idx]


        Y_next = problem(X_next).reshape(-1, 1)
        fid_idx = _fid_index(fidelities, X_next[0, -1])
        cost = costs[fid_idx]


        train_x = torch.cat((train_x, X_next), dim=0)
        train_obj = _append_obj(train_obj, Y_next)


        cumulative_cost.append(float(cost.item()))
        best_obs = _safe_max_target(train_x, train_obj, target_fidelity, MIN_VALUE)

        info = f"Max: {best_obs:.2e}"
        info += f" | Cost: {cost.item():.2e}"
        info += f" | new_fid: {X_next[0][-1].item():.2f}"
        info += f" | new_y: {Y_next[0].item():.2f}"
        info += f" | CI: {_tmp_max_ci.item():.2f}"
        info += f" | Entropy: {lengthscale_entropy.item():.2f}"
        info += f" | RBF: {cost_aware_rbf_rate.item():.2f}"
        info += f" | RRate: {torch.max(cost_aware_learning_rate).item():.2e}"
        info += f" | Valid portion: {_portion:.2f}"
        info += f" | Cost/Budget: {sum(cumulative_cost):.2f}/{config.problem.budget:d}"
        bo_iterator.set_postfix_str(info)

        current_state = {"next_iteration": r_idx + 1}
        if checkpoint_callback is not None:
            checkpoint_callback(train_x, train_obj, cumulative_cost, current_state)


        if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
            break

    return train_x, train_obj, cumulative_cost, {"next_iteration": len(cumulative_cost)}


def rmfbo_pseudo(
    problem,
    fidelities,
    config,
    tkwargs,
    *,
    train_x=None,
    train_obj=None,
    cumulative_cost=None,
    algorithm_state=None,
    checkpoint_callback=None,
    show_progress=True,
):

    stopping_criteria = config.problem.stopping_criteria
    n_iter = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    n_init, low_cost, \
    MIN_VALUE, \
    batch_size, target_fidelity = \
    config.problem.n_init, config.problem.low_cost, \
    config.problem.min_value, \
    config.problem.batch_size, config.problem.target_fidelity

    costs = fidelities + low_cost
    fid_idx = problem.dim - 1
    sf_fidelities = torch.tensor([config.problem.fidelities[-1]], **tkwargs)
    sf_fixed_features_list = [{fid_idx: fid} for fid in sf_fidelities]
    target_fidelity = config.problem.target_fidelity
    target_fidelities = {fid_idx: config.problem.target_fidelity}
    fixed_features_list = [{fid_idx: fid} for fid in fidelities]

    NEGATE = config.problem.negate
    N_INIT = config.problem.n_init
    N_ITER = config.problem.n_iter if stopping_criteria != 'budget' else config.problem.max_n_iter
    NUM_RESTARTS = config.problem.num_restarts
    RAW_SAMPLES = config.problem.raw_samples
    BATCH_SIZE = config.problem.batch_size
    SCALE = config.problem.scale
    MIN_VALUE = config.problem.min_value
    n_repeat = config.problem.n_repeat

    cost_model = AffineFidelityCostModel(fidelity_weights={fid_idx: 1.0}, fixed_cost=low_cost)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    project = get_project(target_fidelities)

    state = algorithm_state or {}
    if train_x is None or train_obj is None:
        train_x, train_obj = generate_initial_data(
            problem=problem,
            fidelities=fidelities,
            n=n_init,
            **tkwargs,
        )
    cumulative_cost = [] if cumulative_cost is None else list(cumulative_cost)
    start_iteration = int(state.get("next_iteration", len(cumulative_cost)))
    if checkpoint_callback is not None and start_iteration == 0:
        checkpoint_callback(train_x, train_obj, cumulative_cost, {"next_iteration": 0})

    already_finished = (
        stopping_criteria == 'budget'
        and sum(cumulative_cost) > float(config.problem.budget)
    )
    iteration_range = range(n_iter, n_iter) if already_finished else range(start_iteration, n_iter)
    bo_iterator = tqdm(
        iteration_range,
        desc="RMFBO-PSEUDO",
        leave=False,
        disable=not show_progress,
    )
    for r_idx in bo_iterator:

        mll, model = initialize_mf_model(train_x, train_obj, data_fidelity=fid_idx)
        fit_gpytorch_mll(mll)

        if config.algorithm.acq_name == 'KG':
            new_x, new_obj, cost = bo_step_kg(
                model,
                cost_aware_utility,
                project,
                fixed_features_list,
                problem,
                cost_model,
                BATCH_SIZE,
                NUM_RESTARTS,
                RAW_SAMPLES,
            )
        elif config.algorithm.acq_name == 'MES':
            new_x, new_obj, cost = bo_step_mes(
                model,
                cost_aware_utility,
                project,
                fidelities,
                fixed_features_list,
                problem,
                cost_model,
                BATCH_SIZE,
                NUM_RESTARTS,
                RAW_SAMPLES,
                tkwargs=tkwargs,
                space_sample_num=config.algorithm.space_sample_num,
            )
        elif config.algorithm.acq_name == 'RAND':
            new_x, new_obj = generate_initial_data(
                problem=problem,
                fidelities=fidelities,
                n=1,
                **tkwargs,
            )
            new_fid_val = new_x[..., -1].item()
            cost = low_cost + new_fid_val
        else:
            raise NotImplementedError


        cond_var = model.posterior(new_x).variance <= (0.5 ** 2)
        cond_is = qMultiFidelityMaxValueEntropy(model=model, candidate_set=new_x)(X=new_x) >= 0.5

        if not (bool(cond_var.all().item()) and bool(cond_is.all().item())):
            if config.algorithm.acq_name == 'KG':
                new_x, new_obj, cost = bo_step_kg(
                    model,
                    cost_aware_utility,
                    project,
                    sf_fixed_features_list,
                    problem,
                    cost_model,
                    BATCH_SIZE,
                    NUM_RESTARTS,
                    RAW_SAMPLES,
                )
            elif config.algorithm.acq_name == 'MES':
                new_x, new_obj, cost = bo_step_mes(
                    model,
                    cost_aware_utility,
                    project,
                    sf_fidelities,
                    sf_fixed_features_list,
                    problem,
                    cost_model,
                    BATCH_SIZE,
                    NUM_RESTARTS,
                    RAW_SAMPLES,
                    tkwargs=tkwargs,
                    space_sample_num=config.algorithm.space_sample_num,
                )
            elif config.algorithm.acq_name == 'RAND':
                new_x, new_obj = generate_initial_data(
                    problem=problem,
                    fidelities=sf_fidelities,
                    n=1,
                    **tkwargs,
                )
                new_fid_val = new_x[..., -1].item()
                cost = low_cost + new_fid_val
            else:
                raise NotImplementedError

        train_x = torch.cat([train_x, new_x], dim=0)
        train_obj = _append_obj(train_obj, new_obj)
        best_obs = _safe_max_target(train_x, train_obj, target_fidelity, MIN_VALUE)
        cumulative_cost.append(float(cost.item()) if torch.is_tensor(cost) else float(cost))

        info = f"Max: {best_obs:.2e}"
        info += f" | Cost: {float(cost.item()) if torch.is_tensor(cost) else float(cost):.2e}"
        info += f" | new_fid: {new_x[0][-1].item():.2f}"
        info += f" | new_y: {new_obj.reshape(-1)[0].item():.2f}"
        info += f" | Cost/Budget: {sum(cumulative_cost):.2f}/{config.problem.budget:d}"
        bo_iterator.set_postfix_str(info)

        current_state = {"next_iteration": r_idx + 1}
        if checkpoint_callback is not None:
            checkpoint_callback(train_x, train_obj, cumulative_cost, current_state)


        if stopping_criteria == 'budget' and sum(cumulative_cost) > config.problem.budget:
            break

    return train_x, train_obj, cumulative_cost, {"next_iteration": len(cumulative_cost)}
