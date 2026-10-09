from tqdm import tqdm
from math import log

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
from torch.quasirandom import SobolEngine

from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.acquisition.utils import project_to_target_fidelity

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval

import warnings

from .acquisition import *
warnings.filterwarnings("ignore")


def optimize_acqf_and_get_acq(acq_func, batch_size=1, bounds=None, num_sample: int = 512):
    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_sample,
        raw_samples=num_sample,
        options={"batch_limit": 100, "maxiter": 100},
    )
    return candidates, acq_value


def get_fitted_mlp(model, train_x, train_y, num_iter, optimizer):

    train_iter = tqdm(range(num_iter), desc="Training MLP")
    model.train()
    for _ in train_iter:
        model.zero_grad()
        pred = model(train_x)
        loss = F.mse_loss(pred, train_y)
        loss.backward()
        optimizer.step()
        train_iter.set_postfix(loss=loss.item())


def get_fitted_model(X, Y, training=True, normalize=False, max_cholesky_size=4096, **kwargs):
    dim = X.shape[-1]
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )

    if normalize:
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
    else:
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if training:
        try:
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                fit_gpytorch_mll(mll)
        except Exception:
            pass

    return model


def generate_initial_data(problem, fidelities, n=16, **kwargs):
    dim = problem.dim

    if "device" not in kwargs:
        kwargs["device"] = problem.bounds.device
    if "dtype" not in kwargs:
        kwargs["dtype"] = problem.bounds.dtype

    if hasattr(problem, "candidates"):
        if n <= len(problem.candidates):
            choice = np.random.choice(len(problem.candidates), n, replace=False)
        else:
            choice = np.random.choice(len(problem.candidates), n, replace=True)

        train_x_full = problem.candidates[choice].to(
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )
        train_obj = problem.objectives[choice].reshape([n, 1]).to(
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        )
    else:
        train_x = unnormalize(
            torch.rand(n, dim - 1, **kwargs),
            bounds=problem.bounds[:, :-1],
        )
        train_f = fidelities[
            torch.randint(
                len(fidelities),
                (n, 1),
                device=fidelities.device,
            )
        ]
        train_x_full = torch.cat((train_x, train_f), dim=1)
        train_obj = problem(train_x_full).reshape(-1, 1)

    return train_x_full, train_obj


def get_project(target_fidelities):
    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
    return project


def initialize_mf_model(train_x, train_obj, data_fidelity: int):

    model = SingleTaskMultiFidelityGP(
        train_x,
        train_obj,
        outcome_transform=Standardize(m=1),
        data_fidelity=data_fidelity,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def compute_maximum_posterior_variance(model, beta: Tensor, bounds=None) -> float:
    _acq_ci = qConfidenceInterval(model, beta)
    _, ci_width = optimize_acqf_and_get_acq(
        _acq_ci,
        batch_size=1,
        bounds=bounds,
        num_sample=512,
    )
    return ci_width.item()


def compute_information_bp_fast_classification(model, x_tr, y_tr, batch_size=200, no_bp=False):
    def one_hot_transform(y, num_class=100):
        one_hot_y = F.one_hot(y, num_classes=model.num_classes)
        return one_hot_y.float()

    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)

    num_all_batch = int(np.ceil(len(x_tr) / batch_size))

    param_keys = [p[0] for p in model.named_parameters()]
    delta_w_dict = dict().fromkeys(param_keys)
    for pa in model.named_parameters():
        if "weight" in pa[0]:
            w0 = model.w0_dict[pa[0]]
            delta_w = pa[1] - w0
            delta_w_dict[pa[0]] = delta_w

    info_dict = dict()
    gw_dict = dict().fromkeys(param_keys)

    for _ in range(10):
        sub_idx = np.random.choice(all_tr_idx, batch_size)
        x_batch = x_tr[sub_idx]
        y_batch = y_tr[sub_idx]

        y_oh_batch = one_hot_transform(y_batch, model.num_class)
        pred = model.forward(x_batch)
        loss = F.cross_entropy(pred, y_batch, reduction="mean")

        gradients = grad(loss, model.parameters())

        for i, gw in enumerate(gradients):
            gw_ = gw.flatten()
            if gw_dict[param_keys[i]] is None:
                gw_dict[param_keys[i]] = gw_
            else:
                gw_dict[param_keys[i]] += gw_

    for k in gw_dict.keys():
        if "weight" in k:
            gw_dict[k] *= 1 / num_all_batch
            delta_w = delta_w_dict[k]
            info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
            if no_bp:
                info_dict[k] = info_.item()
            else:
                info_dict[k] = info_

    return info_dict


def compute_information_bp_fast_regression(model, loss, x_tr, y_tr, batch_size=200, no_bp=False):
    all_tr_idx = np.arange(len(x_tr))
    np.random.shuffle(all_tr_idx)

    num_all_batch = int(np.ceil(len(x_tr) / batch_size))

    param_keys = [p[0] for p in model.named_parameters()]
    delta_w_dict = dict().fromkeys(param_keys)
    for pa in model.named_parameters():
        if "weight" in pa[0]:
            w0 = model.w0_dict[pa[0]]
            delta_w = pa[1] - w0
            delta_w_dict[pa[0]] = delta_w

    info_dict = dict()
    gw_dict = dict().fromkeys(param_keys)

    for _ in range(10):
        sub_idx = np.random.choice(all_tr_idx, batch_size)
        x_batch = x_tr[sub_idx]
        y_batch = y_tr[sub_idx]

        pred = model.forward(x_batch)
        gradients = grad(loss, model.parameters())

        for i, gw in enumerate(gradients):
            gw_ = gw.flatten()
            if gw_dict[param_keys[i]] is None:
                gw_dict[param_keys[i]] = gw_
            else:
                gw_dict[param_keys[i]] += gw_

    for k in gw_dict.keys():
        if "weight" in k:
            gw_dict[k] *= 1 / num_all_batch
            delta_w = delta_w_dict[k]
            info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
            if no_bp:
                info_dict[k] = info_.item()
            else:
                info_dict[k] = info_

    return info_dict


def monte_carlo_entropy(samples: torch.Tensor) -> torch.Tensor:
    if samples.numel() == 0:
        return torch.zeros(1, device=samples.device, dtype=samples.dtype).squeeze()

    if samples.dim() == 1:
        _, counts = torch.unique(samples, return_counts=True)
    else:
        _, counts = torch.unique(samples, return_counts=True, dim=0)

    counts = counts.to(device=samples.device, dtype=samples.dtype)
    total_num = counts.sum()

    probs = counts / total_num
    entropy_estimate = -(probs * torch.log(probs)).sum()

    return entropy_estimate


def monte_carlo_excessive_risk(fidelity_counts: torch.Tensor, beta) -> torch.Tensor:
    device = fidelity_counts.device
    dtype = torch.float64 if fidelity_counts.dtype not in (torch.float32, torch.float64) else fidelity_counts.dtype

    counts = fidelity_counts.to(device=device, dtype=dtype)
    beta_t = torch.as_tensor(beta, device=device, dtype=dtype)

    excessive_risk_estimate = torch.ones_like(counts, dtype=dtype, device=device)

    for i, c in enumerate(counts):
        c_safe = torch.clamp(c, min=torch.tensor(0.5, device=device, dtype=dtype))

        if i == 0:
            excessive_risk_estimate[i] = beta_t / torch.sqrt(c_safe)
            continue

        previous_risk = excessive_risk_estimate[i - 1]
        previous_risk_safe = torch.clamp(
            previous_risk,
            min=torch.tensor(1e-12, device=device, dtype=dtype),
        )

        if previous_risk_safe.item() < 1.0:
            excessive_risk_estimate[i] = previous_risk_safe
        else:
            excessive_risk_estimate[i] = beta_t * (
                previous_risk_safe + torch.sqrt(torch.log(previous_risk_safe)) / c_safe
            )

    return excessive_risk_estimate[-1]


def excessive_risk_reduction_rate(fidelity_counts: torch.Tensor, fidelity_choosen: int, beta) -> torch.Tensor:
    new_fidelity_counts = fidelity_counts.clone()
    new_fidelity_counts[fidelity_choosen] += 1

    rate = (
        monte_carlo_excessive_risk(fidelity_counts, beta)
        - monte_carlo_excessive_risk(new_fidelity_counts, beta)
    )
    return rate


def rbf_kernel_variance_reduction_rate(T: int, dim: int, variance: torch.Tensor) -> torch.Tensor:
    if T <= 1:
        return variance

    rate = 1 - log(T) ** (dim + 1) / log(T + 1) ** (dim + 1)
    return variance * rate
