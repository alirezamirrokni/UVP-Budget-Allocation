'''
Personalized toy example for testing the multi-fidelity optimization algorithms.
'''
from __future__ import annotations

import math
from typing import Optional

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor
import numpy as np


def _to_python_scalar(v):
    if torch.is_tensor(v):
        return v.detach().cpu().item()
    return v


class AugmentedRastrigin(SyntheticTestFunction):
    r"""  
    Augmented Rastrigin test function for multi-fidelity optimization.

    1-dimensional function with domain `[-5, 5] * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = -(x**2 - 10 * cos(2 * pi * x) + 10)
    low fids:
        y1 = y + np.random.normal(1 * np.abs(x))
        y2 = y + np.random.normal(2 * x**2, 8)
        y3 = y + np.random.normal(-.2 * x**4, 8)

    B_min ~= 40.2 which are on the sides
    B_max = 0 which is in the middle
    Discrete fidelities:
        fid = [0.1, 0.5, 0.75, 1]
    """

    dim = 2  # last is fidelity
    _bounds = [(-10.0, 10.0), (0, 1)]
    _optimal_value = 0
    _optimizers = [
        (0.0, 1),
    ]
    _scale = 1.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        single_fid_x = X[..., :-1].squeeze()
        base = (single_fid_x**2 - 10 * torch.cos(2 * math.pi * single_fid_x) + 10)
        fid = X[..., -1]
        fid_correction_mu = ((1 - fid) > 1e-10) * (2 ** ((fid) <= 0.5)) * ((-.1) ** ((fid) <= 0.25))
        fid_correction_mu = fid_correction_mu * (torch.abs(single_fid_x) ** ((1 - fid) / 0.25))
        fidelity_correction = torch.normal(mean=fid_correction_mu, std=8) * ((1 - fid) > 1e-10)
        return (-base - fidelity_correction) * self._scale


class FixedProtein(SyntheticTestFunction):
    r'''
    Fixed Protein test function for multi-fidelity optimization.
    '''
    dim = 87
    _protein_data = None

    @classmethod
    def _get_protein_data(cls):
        if cls._protein_data is None:
            cls._protein_data = torch.load("data/fixed_protein.pt").to(torch.float64)
        return cls._protein_data

    def __init__(self, negate: Optional[bool] = False) -> None:
        protein_data = self._get_protein_data()
        self._bounds = [(protein_data[:, d].min(), protein_data[:, d].max() + 1e-6) for d in range(self.dim)]
        self.candidates = protein_data[:, :-1]
        self.objectives = protein_data[:, -1].reshape(-1, 1)
        self._optimal_value = protein_data[protein_data[:, -2] == 2, -1].min()
        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        indices = []
        protein_data = self._get_protein_data().to(device=X.device, dtype=X.dtype)
        base = protein_data[:, :-1]

        for x in X:
            dist = torch.linalg.norm(base - x, ord=1, dim=-1)
            idx = torch.argmin(dist)
            indices.append(idx)

        return protein_data[indices, -1].to(device=X.device, dtype=X.dtype)


class AugmentedRastrigin20D(SyntheticTestFunction):
    r"""  
    Augmented Rastrigin test function for multi-fidelity optimization.

    1-dimensional function with domain `[-5, 5]^dim * [0,1]`, where
    the last dimension of is the fidelity parameter:

        B(x) = -(x**2 - 10 * cos(2 * pi * x) + 10)
    low fids:
        y1 = y + np.random.normal(1 * np.abs(x))
        y2 = y + np.random.normal(2 * x**2, 8)
        y3 = y + np.random.normal(-.2 * x**4, 8)

    B_max = 0 which is in the middle
    Discrete fidelities:
        fid = [0.1, 0.5, 0.75, 1]
    """

    dim = 21  # last is fidelity
    _bounds = [(-5.0, 5.0) for _ in range(dim - 1)]
    _bounds.append((0, 1))
    _optimal_value = 0
    _optimizers = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
    ]
    _scale = 1.0

    def evaluate_true(self, X: Tensor) -> Tensor:
        single_fid_x = X[..., :-1].squeeze()
        base = (torch.linalg.norm(single_fid_x, ord=2)**2 - 10 * torch.cos(2 * math.pi * single_fid_x.max()) + 10)
        fid = X[..., -1]
        fid_correction_mu = ((1 - fid) > 1e-10) * (2 ** ((fid) <= 0.5)) * ((-.1) ** ((fid) <= 0.25))
        fid_correction_mu = fid_correction_mu * (torch.linalg.norm(single_fid_x) ** ((1 - fid) / 0.25))
        fidelity_correction = torch.normal(mean=fid_correction_mu, std=8) * ((1 - fid) > 1e-10)
        return (-base - fidelity_correction) * self._scale


class YahooGYM(SyntheticTestFunction):
    '''
    Paper found in https://arxiv.org/abs/2109.03670
    Github: https://github.com/slds-lmu/yahpo_gym
    '''
    pass


class LCBench(YahooGYM):
    '''
    Part of YahooGYM, 7D Numeric, 34 instnces, 6 objectives, Fidelity defined by the number of epochs
    '''
    dim = 8
    _bounds = [
        (16, 512),
        (0.0001001, 0.1),
        (0.0, 1.0),
        (64, 1024),
        (0.1, 0.99),
        (1, 5),
        (0.0000101, 0.1),
        (1, 52)
    ]

    def __init__(self, negate: Optional[bool] = False, instance: str = '3945') -> None:
        from yahpo_gym import local_config
        from yahpo_gym import benchmark_set

        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")

        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("lcbench")
        self.input_keys = ['batch_size', 'learning_rate', 'max_dropout', 'max_units', 'momentum', 'num_layers', 'weight_decay', 'epoch']
        self.int_keys = ['batch_size', 'max_units', 'num_layers', 'epoch']

        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x)}
            for k in config.keys():
                val = _to_python_scalar(config[k])
                if k in self.int_keys:
                    config[k] = np.int64(val)
                else:
                    config[k] = np.float64(val)
            config['OpenML_task_id'] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['test_balanced_accuracy'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_rpart(YahooGYM):
    '''
    Part of YahooGYM, 4D Numeric, 4 instances, 12 objectives, Fidelity defined by fraction of training data
    '''
    dim = 5
    _bounds = [(0.001, 1.0), (1, 30), (1, 100), (1, 100), (0.03, 1.0)]

    def __init__(self, negate: Optional[bool] = False, instance: str = '1489') -> None:
        from yahpo_gym import local_config
        from yahpo_gym import benchmark_set

        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")

        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("iaml_rpart")
        self.input_keys = ['cp', 'maxdepth', 'minbucket', 'minsplit', 'trainsize']
        self.int_keys = ['maxdepth', 'minbucket', 'minsplit']

        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x)}
            for k in config.keys():
                val = _to_python_scalar(config[k])
                if k in self.int_keys:
                    config[k] = np.int64(val)
                else:
                    config[k] = np.float64(val)
            config['task_id'] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_xgboost(YahooGYM):
    '''
    Part of YahooGYM, 13D Numeric, 4 instnces, 12 objectives, Fidelity defined by the fraction of training data
    '''
    dim = 13
    _bounds = [
        (0.0001001, 999.9999999999998),
        (0.01001, 1.0),
        (0.01001, 1.0),
        (0.0001001, 1.0),
        (0.0001001, 6.999999999999999),
        (0.0001001, 999.9999999999998),
        (1.0, 15.0),
        (2.7183, 149.99999999999997),
        (3.0, 2000.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.1001, 1.0),
        (0.03, 1.0)
    ]

    def __init__(self, negate: Optional[bool] = False, instance: str = '1489', booster: str = 'dart') -> None:
        from yahpo_gym import local_config
        from yahpo_gym import benchmark_set

        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")

        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("iaml_xgboost")
        self.booster = booster
        self.input_keys = [
            'alpha', 'colsample_bylevel', 'colsample_bytree', 'eta', 'gamma', 'lambda',
            'max_depth', 'min_child_weight',
            'nrounds', 'rate_drop', 'skip_drop', 'subsample', 'trainsize'
        ]
        self.int_keys = ['max_depth', 'nrounds']

        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x)}
            for k in config.keys():
                val = _to_python_scalar(config[k])
                if k in self.int_keys:
                    config[k] = np.int64(val)
                else:
                    config[k] = np.float64(val)
            config['task_id'] = self.instance_id
            config['booster'] = self.booster
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst['auc'] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class HPO_Benchmark(SyntheticTestFunction):
    '''
    Paper found in https://arxiv.org/abs/2109.06716
    Github: https://github.com/automl/HPOBench
    '''
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from botorch.utils.transforms import unnormalize

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    problem = AugmentedRastrigin(negate=False).to(**tkwargs)
    fidelities = torch.tensor([0, 0.5, 0.75, 1.0], **tkwargs)
    fidelity_color = ['red', 'green', 'blue', 'black']
    n = 1000

    train_x = unnormalize(torch.rand(n, 1, **tkwargs).reshape(n, 1), bounds=problem.bounds[:, :-1])
    train_f = fidelities[torch.randint(4, (n, 1), device=tkwargs["device"])]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full).unsqueeze(-1)

    plt.figure()
    for x, y in zip(train_x_full, train_obj):
        if x[-1] == 0:
            label = 0
        elif x[-1] == 0.5:
            label = 1
        elif x[-1] == 0.75:
            label = 2
        else:
            label = 3
        plt.scatter(x[0].detach().cpu().numpy(), y.detach().cpu().numpy(), color=fidelity_color[label], s=5)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'One-Dimensional Rastrigin with Different Fidelities Max {train_obj[train_f == 1].max(dim=0).values:.2e}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'rastrigin_fidelity_min{train_obj[train_f == 1].min():.2e}_max{train_obj[train_f == 1].max():.2e}.png')