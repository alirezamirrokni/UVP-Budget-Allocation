from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

CHECKPOINT_VERSION = 1


def _plain_config(config: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Expected the resolved Hydra configuration to be a mapping.")

    # Runtime-only checkpoint controls should not change experiment identity.
    payload = copy.deepcopy(payload)
    payload.pop("checkpoint", None)
    if isinstance(payload.get("problem"), dict):
        payload["problem"].pop("plot_results", None)
    return payload


def config_signature(config: DictConfig) -> str:
    encoded = json.dumps(
        _plain_config(config),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_checkpoint_path(
    project_root: Path,
    file_name: str,
    config: DictConfig,
    directory: str = "res/checkpoints",
) -> Path:
    signature = config_signature(config)
    path = project_root / directory / f"{file_name}_{signature[:12]}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().numpy(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu().numpy() for item in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in cuda_state]
        )


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one .npy checkpoint file.

    Writing through a temporary file prevents a killed process from corrupting
    the last valid checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, payload, allow_pickle=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = np.load(path, allow_pickle=True)
    if not isinstance(loaded, np.ndarray) or loaded.shape != ():
        raise ValueError(f"Invalid checkpoint format: {path}")
    payload = loaded.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version in {path}: "
            f"{payload.get('version')!r}"
        )
    return payload


def serialize_run_state(
    repeat_idx: int,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    cumulative_cost: list[float],
    algorithm_state: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "repeat_idx": int(repeat_idx),
        "train_x": train_x.detach().cpu().numpy(),
        "train_obj": train_obj.detach().cpu().numpy(),
        "cumulative_cost": [float(value) for value in cumulative_cost],
        "algorithm_state": algorithm_state or {},
    }


def deserialize_run_state(
    state: dict[str, Any],
    tkwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[float], dict[str, Any]]:
    train_x = torch.as_tensor(state["train_x"], **tkwargs)
    train_obj = torch.as_tensor(state["train_obj"], **tkwargs)
    cumulative_cost = [float(value) for value in state.get("cumulative_cost", [])]
    algorithm_state = state.get("algorithm_state") or {}
    return train_x, train_obj, cumulative_cost, algorithm_state
