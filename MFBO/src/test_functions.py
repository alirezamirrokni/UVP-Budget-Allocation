from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


def _to_python_scalar(v):
    if torch.is_tensor(v):
        return v.detach().cpu().item()
    return v


def _snap_float_to_declared_bound(value: float, lower: float, upper: float) -> float:
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    scale = max(1.0, abs(lower), abs(upper))
    tol = 1e-12 * scale

    if value < lower and lower - value <= tol:
        return lower
    if value > upper and value - upper <= tol:
        return upper
    return value


class YahooGYM(SyntheticTestFunction):
    pass


class LCBench(YahooGYM):
    dim = 8
    _bounds = [
        (16, 512),
        (0.0001001, 0.1),
        (0.0, 1.0),
        (64, 1024),
        (0.1, 0.99),
        (1, 5),
        (0.0000101, 0.1),
        (1, 52),
    ]

    def __init__(self, negate: Optional[bool] = False, instance: str = "3945") -> None:
        from yahpo_gym import benchmark_set, local_config

        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")

        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("lcbench")
        self.input_keys = [
            "batch_size",
            "learning_rate",
            "max_dropout",
            "max_units",
            "momentum",
            "num_layers",
            "weight_decay",
            "epoch",
        ]
        self.int_keys = ["batch_size", "max_units", "num_layers", "epoch"]

        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x)}
            for k in config:
                val = _to_python_scalar(config[k])
                if k in self.int_keys:
                    config[k] = np.int64(val)
                else:
                    config[k] = np.float64(val)
            config["OpenML_task_id"] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst["test_balanced_accuracy"] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_rpart(YahooGYM):
    dim = 5
    _bounds = [(0.001, 1.0), (1, 30), (1, 100), (1, 100), (0.03, 1.0)]

    def __init__(self, negate: Optional[bool] = False, instance: str = "1489") -> None:
        from yahpo_gym import benchmark_set, local_config

        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")

        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("iaml_rpart")
        self.input_keys = ["cp", "maxdepth", "minbucket", "minsplit", "trainsize"]
        self.int_keys = ["maxdepth", "minbucket", "minsplit"]

        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x)}
            for k in config:
                val = _to_python_scalar(config[k])
                if k in self.int_keys:
                    config[k] = np.int64(val)
                else:
                    config[k] = np.float64(val)
            config["task_id"] = self.instance_id
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst["auc"] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)


class iaml_xgboost(YahooGYM):
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
        (0.03, 1.0),
    ]

    def __init__(
        self,
        negate: Optional[bool] = False,
        instance: str = "1489",
        booster: str = "dart",
    ) -> None:
        from yahpo_gym import benchmark_set, local_config

        local_config.init_config()
        local_config.set_data_path("./data/yahpo_data")

        self.local_config = local_config
        self.bench = benchmark_set.BenchmarkSet("iaml_xgboost")
        self.booster = booster
        self.input_keys = [
            "alpha",
            "colsample_bylevel",
            "colsample_bytree",
            "eta",
            "gamma",
            "lambda",
            "max_depth",
            "min_child_weight",
            "nrounds",
            "rate_drop",
            "skip_drop",
            "subsample",
            "trainsize",
        ]
        self.int_keys = ["max_depth", "nrounds"]

        assert instance in self.bench.instances
        self.bench.set_instance(instance)
        self.instance_id = instance

        super().__init__(negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        configs = []
        for x in X:
            config = {k: v for k, v in zip(self.input_keys, x)}
            for idx, k in enumerate(config):
                val = _to_python_scalar(config[k])
                lower, upper = self._bounds[idx]
                val = _snap_float_to_declared_bound(val, lower, upper)
                if k in self.int_keys:
                    config[k] = np.int64(val)
                else:
                    config[k] = np.float64(val)
            config["task_id"] = self.instance_id
            config["booster"] = self.booster
            configs.append(config)

        outputs = self.bench.objective_function(configs)
        objectives = [inst["auc"] for inst in outputs]
        return torch.tensor(objectives, dtype=X.dtype, device=X.device)
