
from __future__ import annotations

import random
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import sympy
from sympy.printing.str import StrPrinter
from sklearn.metrics import mean_squared_error, r2_score

try:
    import feyn
except ImportError as exc:
    raise ImportError(
        "Feyn não está instalado neste ambiente. Execute "
        "`python -m pip install -r requirements-qlattice.txt`."
    ) from exc


FEYN_INTERACTIONS = {
    
    "C1": ["add", "multiply", "inverse"],
    "C2": ["add", "multiply", "squared", "exp", "inverse", "log", "sqrt"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _finite_predictions(model: Any, frame: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(model.predict(frame), dtype=float).reshape(-1)
    if pred.shape[0] != len(frame):
        raise ValueError(
            f"QLattice retornou {pred.shape[0]} predições para {len(frame)} linhas."
        )
    return pred


def _expression(model: Any) -> str:
    try:
        expr = model.sympify()
        return str(StrPrinter().doprint(expr))
    except Exception:
        try:
            return str(model)
        except Exception:
            return repr(model)


def _complexity(model: Any) -> int:
    try:
        expr = model.sympify()
        return int(sum(1 for _ in sympy.preorder_traversal(expr)))
    except Exception:
        
        return -1


def _validate_feyn_data(df: pd.DataFrame) -> None:
    """Compatibilidade com as assinaturas de validate_data do Feyn."""
    try:
        feyn.validate_data(df, "regression", "target", stypes={})
    except TypeError:
        feyn.validate_data(df, "regression", "target")


def run_qlattice(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_names: list[str],
    seed: int,
    scenario: str,
    max_evaluations: int = 200_000,
    max_time_seconds: int = 7_200,
    max_edges_per_model: int = 10,
    threads: int = 1,
) -> list[dict[str, Any]]:
  
    if scenario not in {"C1", "C2"}:
        raise ValueError("scenario deve ser C1 ou C2.")
    if max_evaluations <= 0:
        raise ValueError("max_evaluations deve ser positivo.")

    set_seed(seed)
    columns = list(feature_names) + ["target"]
    train_df = pd.DataFrame(
        np.column_stack([np.asarray(X_train), np.asarray(y_train).reshape(-1)]),
        columns=columns,
    )
    validation_df = pd.DataFrame(
        np.column_stack([
            np.asarray(X_validation),
            np.asarray(y_validation).reshape(-1),
        ]),
        columns=columns,
    )
    train_df = train_df.astype(np.float64)
    validation_df = validation_df.astype(np.float64)
    _validate_feyn_data(train_df)

    qlattice = feyn.QLattice()
    
    if hasattr(qlattice, "reset"):
        try:
            qlattice.reset(seed)
        except TypeError:
            qlattice.reset(random_state=seed)

    priors = feyn.tools.estimate_priors(train_df, "target")
    qlattice.update_priors(priors)

    models: list[Any] = []
    sampled = 0
    started = time.monotonic()
    time_guard_hit = False

    while sampled < max_evaluations:
        if time.monotonic() - started >= max_time_seconds:
            time_guard_hit = True
            break

        # Mantém o ciclo nativo do QLattice: sample -> fit -> prune -> update.
        new_models = qlattice.sample_models(
            train_df,
            "target",
            "regression",
            {},
            max_edges_per_model,
            None,
            FEYN_INTERACTIONS[scenario],
        )
        if not new_models:
            break

        sampled += len(new_models)
        models.extend(new_models)
        models = feyn.fit_models(
            models,
            data=train_df,
            threads=threads,
        )
        models = feyn.prune_models(models)
        qlattice.update(models)

    elapsed = time.monotonic() - started
    if not models:
        raise RuntimeError(
            "QLattice não retornou modelos. Verifique versão do Feyn, "
            "compatibilidade da API e dados do fold."
        )

    try:
        diverse = feyn.get_diverse_models(models)
    except Exception:
        diverse = models

    if not diverse:
        diverse = models

    results: list[dict[str, Any]] = []
    for model_id, model in enumerate(diverse):
        try:
            pred_train = _finite_predictions(model, train_df)
            pred_val = _finite_predictions(model, validation_df)
            finite_train = np.all(np.isfinite(pred_train))
            finite_val = np.all(np.isfinite(pred_val))
            mse_train = float(mean_squared_error(y_train, pred_train)) if finite_train else float("inf")
            mse_val = float(mean_squared_error(y_validation, pred_val)) if finite_val else float("inf")
            r2_train = float(r2_score(y_train, pred_train)) if finite_train else float("-inf")
            r2_val = float(r2_score(y_validation, pred_val)) if finite_val else float("-inf")
            status = "ok"
        except Exception as exc:
            mse_train = mse_val = float("inf")
            r2_train = r2_val = float("-inf")
            status = f"prediction_error: {type(exc).__name__}: {exc}"

        results.append({
            "model_id": model_id,
            "expression": _expression(model),
            "complexity": _complexity(model),
            "mse_train": mse_train,
            "mse_test": mse_val,  
            "r2_train": r2_train,
            "r2_test": r2_val,    
            "seed": seed,
            "status": status,
            "algorithm": "QLattice",
            "sampled_models": sampled,
            "max_evaluations": max_evaluations,
            "elapsed_seconds": elapsed,
            "time_guard_hit": time_guard_hit,
            "threads": threads,
            "max_edges_per_model": max_edges_per_model,
            "requested_scenario": scenario,
            "requested_operators": (
                ["+", "-", "*", "/"] if scenario == "C1"
                else ["+", "-", "*", "/", "^", "exp", "log", "sqrt"]
            ),
            "effective_feyn_interactions": FEYN_INTERACTIONS[scenario],
            "operator_constraints_exact": False,
            "operator_constraints_note": (
                "C1 usa add/multiply/inverse, com subtração via pesos com sinal "
                 "e divisão via inverse*multiply. C2 usa squared e sqrt (não "
                 "potência/raiz arbitrária). Portanto, a equivalência é funcional "
                 "parcial, não identidade literal de primitivas."
            ),
            "evaluation_split": "validation_fold",
            "test_set_used": False,
        })

    
    return results

