from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
import traceback
import sys

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
import torch

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold




ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGORITHM = "DySymNet"
OUTPUT_FOLDER = "DySymNet"

OUTPUT_DIR = ROOT / "results_tuning" / OUTPUT_FOLDER

N_FOLDS = 5

SEEDS = [42, 43, 44, 45, 46]

MAX_EVALUATIONS = 1000

MAX_TIME_SECONDS = 120.0




DATASETS = {
    "chemical_2_competition":
        ROOT / "dados/datasets/chemical_2_competition_train.csv",

    "friction_dyn_one-hot":
        ROOT / "dados/datasets/friction_dyn_one-hot_train.csv",

    "nasa_battery_1_10min":
        ROOT / "dados/datasets/nasa_battery_1_10min_train.csv",

    "nikuradse_1":
        ROOT / "dados/datasets/nikuradse_1_train.csv",
}




SCENARIOS = {
    "C1": [
        "+",
        "-",
        "*",
        "/",
    ],

    "C2": [
        "+",
        "-",
        "*",
        "/",
        "^",
        "exp",
        "log",
        "sqrt",
    ],
}




from algorithms.wrapper_dysymnet import (
    run_dysymnet,
)




def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    if dataset not in DATASETS:
        raise ValueError(
            f"Dataset desconhecido: {dataset}"
        )

    path = DATASETS[dataset]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}"
        )

    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(
            f"Coluna target não encontrada: {path}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column != "target"
    ]

    X = df[feature_columns].to_numpy(
        dtype=np.float64
    )

    y = df["target"].to_numpy(
        dtype=np.float64
    ).reshape(-1)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Valores inválidos em X: {dataset}"
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"Valores inválidos em y: {dataset}"
        )

    if len(X) < N_FOLDS:
        raise ValueError(
            f"Dataset possui menos de {N_FOLDS} amostras."
        )

    return X, y, feature_columns



def predict_expression(
    expression,
    X: np.ndarray,
) -> np.ndarray:

    expression = sp.sympify(expression)

    n_features = X.shape[1]

    variables = [
        sp.Symbol(f"x_{i + 1}")
        for i in range(n_features)
    ]

    function = sp.lambdify(
        variables,
        expression,
        modules=["numpy"],
    )

    inputs = [
        X[:, i]
        for i in range(n_features)
    ]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        predictions = function(*inputs)

    predictions = np.asarray(predictions)

    if predictions.ndim == 0:
        predictions = np.full(
            len(X),
            predictions.item(),
        )

    predictions = predictions.reshape(-1)

    if np.iscomplexobj(predictions):
        predictions = np.real_if_close(
            predictions,
            tol=1000,
        )

        if np.iscomplexobj(predictions):
            raise ValueError(
                "A expressão produziu valores complexos."
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    if predictions.shape != (len(X),):
        raise ValueError(
            "Dimensão das predições incompatível."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "A expressão produziu NaN ou infinito."
        )

    return predictions




def expression_complexity(expression) -> int:

    expression = sp.sympify(expression)

    return len(
        list(
            sp.preorder_traversal(expression)
        )
    )




def save_result(
    result: dict[str, Any],
    output: Path,
):

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_suffix(
        output.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            [result],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(output)


def is_successful_result(output: Path) -> bool:

    if not output.exists():
        return False

    try:
        data = json.loads(
            output.read_text(encoding="utf-8")
        )

        return (
            isinstance(data, list)
            and len(data) > 0
            and data[0].get("status") == "success"
        )

    except Exception:
        return False


def save_failure(
    dataset: str,
    scenario: str,
    seed: int,
    fold: int,
    output_dir: Path,
    error: str,
    max_evaluations: int = MAX_EVALUATIONS,
    max_time_seconds: float = MAX_TIME_SECONDS,
):

    result = {
        "model_id": 0,

        "expression": None,

        "complexity": None,

        "mse_train": None,
        "mse_test": None,

        "r2_train": None,
        "r2_test": None,

        "seed": int(seed),

        "algorithm": ALGORITHM,

        "dataset": dataset,

        "scenario": scenario,

        "fold": int(fold),

        "max_evaluations": int(max_evaluations),

        "max_time_seconds": float(max_time_seconds),

        "status": "failed",

        "error": error,
    }

    output = output_dir / (
        f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
    )

    save_result(result, output)




def run_one(
    dataset: str,
    scenario: str,
    seed: int,
    fold: int,
    output_dir: Path,
    max_evaluations: int = MAX_EVALUATIONS,
    max_time_seconds: float = MAX_TIME_SECONDS,
    overwrite: bool = False,
) -> Path:

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Cenário inválido: {scenario}"
        )

    if dataset not in DATASETS:
        raise ValueError(
            f"Dataset inválido: {dataset}"
        )

    if not 1 <= fold <= N_FOLDS:
        raise ValueError(
            "Fold precisa estar entre 1 e 5."
        )

    if max_evaluations <= 0:
        raise ValueError(
            "max_evaluations precisa ser positivo."
        )

    if max_time_seconds <= 0:
        raise ValueError(
            "max_time_seconds precisa ser positivo."
        )

    output = output_dir / (
        f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
    )


    if (
        output.exists()
        and not overwrite
        and is_successful_result(output)
    ):

        print(
            f"Resultado existente; pulando: {output.name}",
            flush=True,
        )

        return output



    X, y, feature_columns = load_dataset(dataset)



    splitter = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    splits = list(
        splitter.split(X)
    )

    train_idx, validation_idx = splits[fold - 1]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[validation_idx]
    y_val = y[validation_idx]



    work_dir = output_dir / "_work"

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir = output_dir / "_logs"

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(seed)

    start = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="dysymnet_",
        dir=str(work_dir),
    ) as temporary_directory:

        train_path = (
            Path(temporary_directory)
            / "train.csv"
        )

        train_df = pd.DataFrame(
            X_train,
            columns=feature_columns,
        )

        train_df["target"] = y_train

        train_df.to_csv(
            train_path,
            index=False,
        )

   

        model_result = run_dysymnet(
            train_csv_path=train_path,

            scenario=scenario,

            seed=seed,

            max_evaluations=max_evaluations,

            log_dir=log_dir,

            run_name=(
                f"{scenario}_{dataset}_seed_{seed}_fold_{fold}"
            ),

            max_time_seconds=max_time_seconds,
        )


    expression = model_result["expression"]

    expression_string = str(expression)

    complexity = expression_complexity(
        expression
    )


    pred_train = predict_expression(
        expression,
        X_train,
    )

    pred_val = predict_expression(
        expression,
        X_val,
    )



    mse_train = float(
        mean_squared_error(
            y_train,
            pred_train,
        )
    )

    mse_test = float(
        mean_squared_error(
            y_val,
            pred_val,
        )
    )

    r2_train = float(
        r2_score(
            y_train,
            pred_train,
        )
    )

    r2_test = float(
        r2_score(
            y_val,
            pred_val,
        )
    )

    runtime = time.perf_counter() - start


    result = {
        "model_id": 0,

        "expression": expression_string,

        "complexity": complexity,

        "mse_train": mse_train,

        "mse_test": mse_test,

        "r2_train": r2_train,

        "r2_test": r2_test,

        "seed": int(seed),

        "algorithm": ALGORITHM,

        "dataset": dataset,

        "scenario": scenario,

        "fold": int(fold),

        "runtime_seconds": float(runtime),

        "max_evaluations": int(max_evaluations),

        "max_time_seconds": float(max_time_seconds),

        "evaluations_used": int(
            model_result["evaluations_used"]
        ),

        "controller_r2": model_result["controller_r2"],

        "controller_mse": model_result["controller_mse"],

        "relative_error": model_result["relative_error"],

        "n_samples": int(len(X)),

        "n_train": int(len(train_idx)),

        "n_test": int(len(validation_idx)),

        "n_features": int(X.shape[1]),

        "n_layers": model_result["n_layers"],

        "num_func_layer": model_result["num_func_layer"],

        "n_epochs1": model_result["n_epochs1"],

        "n_epochs2": model_result["n_epochs2"],

        "function_set": SCENARIOS[scenario],

        "effective_functions": model_result["funcs_avail"],

        "operator_constraints_exact": False,

        "operator_constraints_note": (
            "C1: divisão estabilizada por epsilon. "
            "C2: potência protegida Abs(x)^clip(y,-5,5), "
            "além da divisão estabilizada."
        ),

        "status": "success",

        "error": None,
    }

    save_result(
        result,
        output,
    )

    print(
        f"Salvo: {output.name} | "
        f"R²={r2_test:.6f} | "
        f"MSE={mse_test:.8g} | "
        f"Complexidade={complexity} | "
        f"Avaliações={result['evaluations_used']} | "
        f"Tempo={runtime:.1f}s",
        flush=True,
    )

    return output



def run_all(
    output_dir: Path,
    datasets_to_run: list[str],
    scenarios_to_run: list[str],
    seeds: list[int],
    max_evaluations: int,
    max_time_seconds: float,
    overwrite: bool,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = (
        len(datasets_to_run)
        * len(scenarios_to_run)
        * len(seeds)
        * N_FOLDS
    )

    current = 0

    completed = 0
    skipped = 0

    failures = []

    for scenario in scenarios_to_run:

        for dataset in datasets_to_run:

            for seed in seeds:

                for fold in range(1, N_FOLDS + 1):

                    current += 1

                    print(
                        f"\n[{current}/{total}] "
                        f"{ALGORITHM} | "
                        f"{scenario} | "
                        f"{dataset} | "
                        f"seed={seed} | "
                        f"fold={fold} | "
                        f"limite={max_time_seconds:.0f}s",
                        flush=True,
                    )

                    output = output_dir / (
                        f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
                    )

                    if (
                        output.exists()
                        and not overwrite
                        and is_successful_result(output)
                    ):

                        skipped += 1
                        completed += 1

                        print(
                            f"Já existe: {output.name}",
                            flush=True,
                        )

                        continue

                    try:

                        run_one(
                            dataset=dataset,

                            scenario=scenario,

                            seed=seed,

                            fold=fold,

                            output_dir=output_dir,

                            max_evaluations=max_evaluations,

                            max_time_seconds=max_time_seconds,

                            overwrite=True,
                        )

                        if is_successful_result(output):
                            completed += 1

                    except Exception as exc:

                        traceback.print_exc()

                        error = (
                            f"{type(exc).__name__}: {exc}"
                        )

                        failures.append({
                            "dataset": dataset,

                            "scenario": scenario,

                            "seed": seed,

                            "fold": fold,

                            "error": error,
                        })

                        save_failure(
                            dataset=dataset,
                            scenario=scenario,
                            seed=seed,
                            fold=fold,
                            output_dir=output_dir,
                            error=error,
                            max_evaluations=max_evaluations,
                            max_time_seconds=max_time_seconds,
                        )

                        print(
                            f"FALHA: {error}",
                            flush=True,
                        )

 

    expected_names = {

        f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"

        for scenario in scenarios_to_run

        for dataset in datasets_to_run

        for seed in seeds

        for fold in range(1, N_FOLDS + 1)
    }

    found = sum(
        1
        for name in expected_names
        if (output_dir / name).exists()
    )

    valid = sum(
        1
        for name in expected_names
        if is_successful_result(output_dir / name)
    )

    report = {
        "algorithm": ALGORITHM,

        "expected_experiments": total,

        "files_found": found,

        "valid_files": valid,

        "completed_or_skipped": completed,

        "skipped_existing_valid": skipped,

        "failed_executions": len(failures),

        "datasets": datasets_to_run,

        "scenarios": scenarios_to_run,

        "seeds": seeds,

        "folds": list(range(1, N_FOLDS + 1)),

        "max_evaluations_per_execution": max_evaluations,

        "max_time_seconds_per_execution": max_time_seconds,

        "failures": failures,
    }

    report_path = (
        output_dir
        / "relatorio_execucao_DySymNet.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)

    print("BENCHMARK DYSYMNET FINALIZADO")

    print("=" * 65)

    print(f"Experimentos esperados: {total}")

    print(f"Arquivos encontrados: {found}")

    print(f"Execuções válidas: {valid}")

    print(f"Falhas: {len(failures)}")

    print(f"Relatório: {report_path}")



def safe_float(value):

    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DySymNet: "
            "4 datasets x 5 seeds x 5 folds x 2 cenários."
        )
    )

    parser.add_argument(
        "--all-experiments",
        action="store_true",
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
    )

    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, N_FOLDS + 1),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=MAX_EVALUATIONS,
    )

    parser.add_argument(
        "--max-time-seconds",
        type=float,
        default=MAX_TIME_SECONDS,
        help=(
            "Tempo máximo em segundos por execução individual. "
            "Padrão: 300."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.max_evaluations <= 0:
        parser.error(
            "--max-evaluations deve ser positivo."
        )

    if args.max_evaluations % 10 != 0:
        parser.error(
            "--max-evaluations deve ser múltiplo de 10."
        )

    if args.max_time_seconds <= 0:
        parser.error(
            "--max-time-seconds deve ser positivo."
        )

    if not args.seeds:
        parser.error(
            "--seeds precisa conter pelo menos uma seed."
        )

    invalid_seeds = [
        seed
        for seed in args.seeds
        if seed not in SEEDS
    ]

    if invalid_seeds:
        parser.error(
            f"Seeds inválidas: {invalid_seeds}. "
            f"Valores permitidos: {SEEDS}."
        )

    if args.all_experiments:

        datasets_to_run = (
            [args.dataset]
            if args.dataset
            else list(DATASETS)
        )

        scenarios_to_run = (
            [args.scenario]
            if args.scenario
            else list(SCENARIOS)
        )

        run_all(
            output_dir=args.output_dir,

            datasets_to_run=datasets_to_run,

            scenarios_to_run=scenarios_to_run,

            seeds=args.seeds,

            max_evaluations=args.max_evaluations,

            max_time_seconds=args.max_time_seconds,

            overwrite=args.overwrite,
        )

        return

    required = (
        args.dataset,
        args.scenario,
        args.seed,
        args.fold,
    )

    if any(value is None for value in required):

        parser.error(
            "Use --all-experiments ou informe "
            "--dataset, --scenario, --seed e --fold."
        )

    run_one(
        dataset=args.dataset,

        scenario=args.scenario,

        seed=args.seed,

        fold=args.fold,

        output_dir=args.output_dir,

        max_evaluations=args.max_evaluations,

        max_time_seconds=args.max_time_seconds,

        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
