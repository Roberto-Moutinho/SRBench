from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
import torch
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]

DYSYMNET_ROOT = ROOT / "algorithms" / "vendor" / "DySymNet-main"

if not DYSYMNET_ROOT.exists():
    raise FileNotFoundError(f"DySymNet original não encontrado: {DYSYMNET_ROOT}")

if str(DYSYMNET_ROOT) not in sys.path:
    sys.path.insert(0, str(DYSYMNET_ROOT))

from DySymNet.SymbolicRegression import SymboliRegression
from DySymNet.scripts.params import Params
from DySymNet.scripts.functions import (
    BaseFunction2, Plus, Sub, Product, Div, Exp, Log, Sqrt,
)


class Power(BaseFunction2):
    """Potência binária protegida: |x|^clip(y, -5, 5)."""

    def __init__(self):
        super().__init__(norm=1.0)
        self.name = "^"

    def torch(self, x, y):
        base = torch.abs(x) + 1e-6
        exponent = torch.clamp(y, min=-5.0, max=5.0)
        return torch.pow(base, exponent)

    def sp(self, x, y):
        return sp.Abs(x) ** y


def get_operators(scenario: str):
    if scenario == "C1":
        return [Plus(), Sub(), Product(norm=1.0), Div()]
    if scenario == "C2":
        return [
            Plus(), Sub(), Product(norm=1.0), Div(),
            Exp(), Log(), Sqrt(), Power()
        ]
    raise ValueError(f"Cenário inválido: {scenario}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_config(
    scenario: str,
    max_evaluations: int = 1000,
    log_dir: Path | None = None,
    batch_size: int = 10,
    max_time_seconds: float | None = 120.0,
    warmup_epochs: int = 500,
):
    if max_evaluations <= 0:
        raise ValueError("max_evaluations precisa ser positivo.")
    if batch_size <= 0:
        raise ValueError("batch_size precisa ser positivo.")
    if max_evaluations % batch_size != 0:
        raise ValueError("max_evaluations deve ser múltiplo de batch_size.")
    if max_time_seconds is not None and max_time_seconds <= 0:
        raise ValueError("max_time_seconds precisa ser positivo ou None.")

    config = Params()
    funcs = get_operators(scenario)

    config.funcs_avail = funcs
    config.input_size = len(funcs)

    config.n_layers = 3
    config.num_func_layer = 4
    config.hidden_size = 32
    config.embedding_size = 16

    config.optimizer = "Adam"
    config.learning_rate1 = 0.0006
    config.num_epochs = max_evaluations // batch_size
    config.batch_size = batch_size
    config.risk_seeking = True
    config.risk_factor = 0.5
    config.entropy_weight = 0.005

    config.learning_rate2 = 0.01
    config.n_epochs1 = 50
    config.n_epochs2 = 50
    config.summary_step = 10
    config.reg_weight = 0.005
    config.threshold = 0.05
    config.trials = 1

    config.clip_grad = True
    config.max_norm = 1.0
    config.window_size = 50
    config.refine_constants = True
    config.n_restarts = 1
    config.add_bias = False

    config.reward_type = "mse"
    config.use_gpu = False
    config.verbose = False
    config.plot_reward = False
    config.NOISE = 0

    config.max_time_seconds = max_time_seconds
    config.warmup_epochs = warmup_epochs
    config.max_bfgs_constants = 30
    config.bfgs_max_time = 20

    if log_dir is None:
        log_dir = ROOT / "results_tuning" / "DySymNet" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir = str(log_dir)

    return config


def select_features(
    train_csv_path: Path,
    max_features: int | None,
    seed: int,
):
   
    df = pd.read_csv(train_csv_path)
    n_features = df.shape[1] - 1
    all_idx = list(range(n_features))

    if max_features is None or n_features <= max_features:
        return train_csv_path, all_idx

    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    mi = mutual_info_regression(X, y, random_state=seed)
    selected = sorted(np.argsort(mi)[::-1][:max_features].tolist())

    reduced = df.iloc[:, selected + [n_features]]
    reduced_path = train_csv_path.with_name(
        train_csv_path.stem + "_selected.csv"
    )
    reduced.to_csv(reduced_path, index=False)

    print(
        f"[DySymNet] Seleção de variáveis: {n_features} -> {len(selected)} "
        f"(índices originais: {[i + 1 for i in selected]})",
        flush=True,
    )
    return reduced_path, selected


def run_dysymnet(
    train_csv_path: Path,
    scenario: str,
    seed: int,
    max_evaluations: int = 1000,
    log_dir: Path | None = None,
    run_name: str | None = None,
    max_time_seconds: float | None = 120.0,
    max_features: int | None = 12,
    warmup_epochs: int = 500,
) -> dict[str, Any]:

    train_csv_path = Path(train_csv_path)
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"Dataset de treinamento não encontrado: {train_csv_path}"
        )

    if max_time_seconds is not None and max_time_seconds <= 0:
        raise ValueError("max_time_seconds precisa ser positivo ou None.")

    set_seed(seed)

    config = build_config(
        scenario=scenario,
        max_evaluations=max_evaluations,
        log_dir=log_dir,
        max_time_seconds=max_time_seconds,
        warmup_epochs=warmup_epochs,
    )

    func_name = (
        run_name
        if run_name is not None
        else f"{scenario}_seed_{seed}"
    )

    used_csv, selected = select_features(
        train_csv_path,
        max_features,
        seed,
    )

    model = SymboliRegression(
        config=config,
        func_name=func_name,
        data_path=str(used_csv),
    )


    counter = {"evaluations": 0}
    original_play_episodes = model.play_episodes

    def counted_play_episodes():
        counter["evaluations"] += 1
        return original_play_episodes()

    model.play_episodes = counted_play_episodes

    expression, controller_r2, controller_mse, relative_error = (
        model.solve_environment()
    )

    if expression is None:
        raise RuntimeError(
            "DySymNet não conseguiu produzir uma expressão."
        )

    expression = sp.sympify(expression)

    mapping = {
        sp.Symbol(f"x_{k + 1}"): sp.Symbol(f"x_{orig + 1}")
        for k, orig in enumerate(selected)
    }
    expression = expression.xreplace(mapping)

    return {
        "expression": expression,
        "expression_string": str(expression),
        "evaluations_used": counter["evaluations"],
        "max_evaluations": max_evaluations,
        "max_time_seconds": max_time_seconds,
        "controller_r2": float(controller_r2),
        "controller_mse": float(controller_mse),
        "relative_error": float(relative_error),
        "n_layers": config.n_layers,
        "num_func_layer": config.num_func_layer,
        "n_epochs1": config.n_epochs1,
        "n_epochs2": config.n_epochs2,
        "funcs_avail": [
            func.name for func in config.funcs_avail
        ],
        "selected_features": [
            i + 1 for i in selected
        ],
        "scenario": scenario,
        "seed": seed,
    }
