from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.wrapper_QLattice import run_qlattice

ALGORITHM = "QLattice"
CONFIG_PATH = ROOT / "configs" / "config_base_qlattice.json"
OUTPUT_DIR = ROOT / "results_tuning" / ALGORITHM


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuração não encontrada: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = (
        "experiment",
        "operator_sets",
        "parameters",
        "protocol_notes",
    )

    missing = [
        key for key in required
        if key not in config
    ]

    if missing:
        raise ValueError(
            f"Campos ausentes no config: {missing}"
        )

    cv = config["experiment"]["cross_validation"]
    seeds = cv["seeds"]
    n_folds = int(cv["n_folds"])

    if n_folds < 2:
        raise ValueError(
            "cross_validation.n_folds deve ser >= 2."
        )

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(
            "cross_validation.seeds deve ser não vazio e sem duplicatas."
        )

    budget = config["experiment"]["budget"]

    if int(budget["max_evaluations"]) <= 0:
        raise ValueError(
            "max_evaluations deve ser positivo."
        )

    if int(budget["max_time_seconds"]) <= 0:
        raise ValueError(
            "max_time_seconds deve ser positivo."
        )

    if set(config["operator_sets"]) != {"C1", "C2"}:
        raise ValueError(
            "operator_sets deve conter exatamente C1 e C2."
        )

    return config


def build_datasets(
    config: dict[str, Any]
) -> dict[str, Path]:

    filenames = config["protocol_notes"]["tuning_datasets"]

    datasets: dict[str, Path] = {}

    for filename in filenames:
        name = filename.removesuffix("_train.csv")

        datasets[name] = (
            ROOT / "dados" / "datasets" / filename
        )

    if not datasets:
        raise ValueError(
            "Nenhum dataset definido em protocol_notes.tuning_datasets."
        )

    return datasets


def scenario_names(
    config: dict[str, Any]
) -> list[str]:

    return list(config["operator_sets"].keys())


def output_name(
    scenario: str,
    dataset: str,
    seed: int,
    fold: int,
) -> str:

    return (
        f"{scenario}_{dataset}_seed_{seed}_fold_{fold}.csv"
    )


def read_result(
    path: Path,
) -> list[dict[str, Any]] | None:

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, list) or not data:
        return None

    if not all(isinstance(row, dict) for row in data):
        return None

    return data


def is_successful_result(path: Path) -> bool:

    rows = read_result(path)

    if not rows:
        return False

    return (
        not any(
            row.get("status") == "execution_error"
            for row in rows
        )
        and any(
            row.get("status") == "ok"
            for row in rows
        )
    )


def _json_safe(value: Any) -> Any:


    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]

    if isinstance(value, (float, np.floating)):
        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    if isinstance(value, (int, np.integer)):
        return int(value)

    return value


def write_result(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            _json_safe(rows),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


def load_dataset(
    dataset: str,
    datasets: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    path = datasets[dataset]

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}"
        )

    frame = pd.read_csv(path)

    if "target" not in frame.columns:
        raise ValueError(
            f"{path.name} não contém a coluna obrigatória 'target'."
        )

    if frame.empty:
        raise ValueError(
            f"{path.name} está vazio."
        )

    X_frame = frame.drop(columns=["target"])

    if X_frame.empty:
        raise ValueError(
            f"{path.name} não possui variáveis preditoras."
        )

    X = X_frame.to_numpy(dtype=np.float64)

    y = frame["target"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(
            f"{path.name} contém NaN ou infinito."
        )

    if len(frame) < 5:
        raise ValueError(
            f"{path.name} possui menos de 5 linhas; "
            "não permite 5-fold CV."
        )

    return X, y, list(X_frame.columns)


def run_one(
    dataset: str,
    scenario: str,
    seed: int,
    fold: int,
    output_dir: Path,
    datasets: dict[str, Path],
    config: dict[str, Any],
    max_evaluations: int,
    max_time_seconds: int,
    max_edges_per_model: int,
    overwrite: bool = False,
) -> Path:

    seeds = list(
        config["experiment"]["cross_validation"]["seeds"]
    )

    n_folds = int(
        config["experiment"]["cross_validation"]["n_folds"]
    )

    scenarios = scenario_names(config)

    if dataset not in datasets:
        raise ValueError(
            f"Dataset inválido: {dataset}. "
            f"Opções: {list(datasets)}"
        )

    if scenario not in scenarios:
        raise ValueError(
            f"Cenário inválido: {scenario}. "
            f"Opções: {scenarios}"
        )

    if seed not in seeds:
        raise ValueError(
            f"Seed inválida: {seed}. "
            f"Seeds do protocolo: {seeds}"
        )

    if not 1 <= fold <= n_folds:
        raise ValueError(
            f"fold deve estar entre 1 e {n_folds}."
        )

    if (
        max_evaluations <= 0
        or max_time_seconds <= 0
        or max_edges_per_model <= 0
    ):
        raise ValueError(
            "Orçamento, tempo e max_edges_per_model devem ser positivos."
        )

    output = output_dir / output_name(
        scenario,
        dataset,
        seed,
        fold,
    )

    if (
        output.exists()
        and not overwrite
        and is_successful_result(output)
    ):
        print(
            f"Resultado válido existente; pulando: {output.name}",
            flush=True,
        )
        return output

    X, y, feature_names = load_dataset(
        dataset,
        datasets,
    )

    cv = config["experiment"]["cross_validation"]

    splitter = KFold(
        n_splits=n_folds,
        shuffle=bool(cv.get("shuffle", True)),
        random_state=(
            seed
            if bool(cv.get("shuffle", True))
            else None
        ),
    )

    splits = list(splitter.split(X))

    train_idx, validation_idx = splits[fold - 1]

    print(
        f"\n{ALGORITHM} | {scenario} | {dataset} "
        f"| seed={seed} | fold={fold} "
        f"| budget={max_evaluations:,} "
        f"| time={max_time_seconds}s",
        flush=True,
    )

    started = time.monotonic()

    try:

        rows = run_qlattice(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_validation=X[validation_idx],
            y_validation=y[validation_idx],
            feature_names=feature_names,
            seed=seed,
            scenario=scenario,
            max_evaluations=max_evaluations,
            max_time_seconds=max_time_seconds,
            max_edges_per_model=max_edges_per_model,
            threads=int(
                config["parameters"].get("threads", 1)
            ),
        )

        if not rows:
            raise RuntimeError(
                "QLattice terminou sem retornar modelos."
            )

        for row in rows:

            row.update({
                "algorithm": ALGORITHM,
                "dataset": dataset,
                "scenario": scenario,
                "seed": seed,
                "fold": fold,
                "n_folds": n_folds,
                "max_evaluations": max_evaluations,
                "max_time_seconds": max_time_seconds,
                "max_edges_per_model": max_edges_per_model,
                "threads": int(
                    config["parameters"].get("threads", 1)
                ),
                "fold_train_rows": int(len(train_idx)),
                "fold_validation_rows": int(len(validation_idx)),
                "evaluation_split": "validation_fold",
                "test_set_used": False,
                "requested_binary_operators": (
                    config["operator_sets"][scenario].get(
                        "binary_operators", []
                    )
                ),
                "requested_unary_operators": (
                    config["operator_sets"][scenario].get(
                        "unary_operators", []
                    )
                ),
                "operator_constraints_exact": (
                    config.get(
                        "qlattice_compatibility", {}
                    ).get(
                        "operator_sets_exactly_enforced",
                        False,
                    )
                ),
            })

    except Exception as exc:

        rows = [{
            "algorithm": ALGORITHM,
            "dataset": dataset,
            "scenario": scenario,
            "seed": seed,
            "fold": fold,
            "status": "execution_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "max_evaluations": max_evaluations,
            "max_time_seconds": max_time_seconds,
            "test_set_used": False,
            "evaluation_split": "validation_fold",
        }]

        print(
            f"FALHA: {type(exc).__name__}: {exc}",
            flush=True,
        )

    # JSON com extensão .csv é mantido por compatibilidade
    # com o layout histórico deste repositório.

    write_result(output, rows)

    print(
        f"Salvo: {output} "
        f"({time.monotonic() - started:.1f}s)",
        flush=True,
    )

    return output


def run_all(
    output_dir: Path,
    datasets: dict[str, Path],
    config: dict[str, Any],
    seeds: list[int],
    scenarios_to_run: list[str],
    max_evaluations: int,
    max_time_seconds: int,
    max_edges_per_model: int,
    overwrite: bool,
) -> None:

    valid_scenarios = scenario_names(config)

    invalid = sorted(
        set(scenarios_to_run) - set(valid_scenarios)
    )

    if invalid:
        raise ValueError(
            f"Cenários inválidos: {invalid}"
        )

    protocol_seeds = list(
        config["experiment"]["cross_validation"]["seeds"]
    )

    invalid_seeds = sorted(
        set(seeds) - set(protocol_seeds)
    )

    if invalid_seeds:
        raise ValueError(
            f"Seeds fora do protocolo: {invalid_seeds}"
        )

    n_folds = int(
        config["experiment"]["cross_validation"]["n_folds"]
    )

    total = (
        len(datasets)
        * len(scenarios_to_run)
        * len(seeds)
        * n_folds
    )

    current = 0
    failures: list[dict[str, str]] = []

    skipped = 0
    completed = 0

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for scenario in scenarios_to_run:

        for dataset in datasets:

            for seed in seeds:

                for fold in range(1, n_folds + 1):

                    current += 1

                    output = output_dir / output_name(
                        scenario,
                        dataset,
                        seed,
                        fold,
                    )

                    print(
                        f"\n[{current}/{total}]",
                        flush=True,
                    )

                    if (
                        output.exists()
                        and not overwrite
                        and is_successful_result(output)
                    ):

                        print(
                            f"Resultado válido existente; "
                            f"pulando: {output.name}",
                            flush=True,
                        )

                        skipped += 1
                        completed += 1

                        continue

                    try:

                        run_one(
                            dataset=dataset,
                            scenario=scenario,
                            seed=seed,
                            fold=fold,
                            output_dir=output_dir,
                            datasets=datasets,
                            config=config,
                            max_evaluations=max_evaluations,
                            max_time_seconds=max_time_seconds,
                            max_edges_per_model=max_edges_per_model,
                            overwrite=True,
                        )

                        if is_successful_result(output):

                            completed += 1

                        else:

                            rows = read_result(output) or []

                            error = (
                                rows[0].get(
                                    "error",
                                    "Resultado sem modelo válido",
                                )
                                if rows
                                else "Arquivo vazio/inválido"
                            )

                            failures.append({
                                "file": output.name,
                                "error": str(error),
                            })

                    except Exception as exc:

                        failures.append({
                            "file": output.name,
                            "error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        })

                        print(
                            f"FALHA NO EXECUTOR: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )


    expected_names = {
        output_name(
            scenario,
            dataset,
            seed,
            fold,
        )
        for scenario in scenarios_to_run
        for dataset in datasets
        for seed in seeds
        for fold in range(1, n_folds + 1)
    }

    existing_names = {
        p.name
        for p in output_dir.glob("*.csv")
    }

    found = len(
        expected_names & existing_names
    )

    valid_count = sum(
        1
        for name in expected_names
        if (
            (output_dir / name).exists()
            and is_successful_result(output_dir / name)
        )
    )

    report = {
        "algorithm": ALGORITHM,
        "expected_files_in_this_run": total,
        "files_found_in_this_run": found,
        "valid_files_in_this_run": valid_count,
        "completed_or_skipped": completed,
        "skipped_existing_valid": skipped,
        "failed_executions": len(failures),
        "datasets": list(datasets),
        "scenarios": scenarios_to_run,
        "seeds": seeds,
        "folds": list(range(1, n_folds + 1)),
        "max_evaluations_per_run": max_evaluations,
        "max_time_seconds_per_run": max_time_seconds,
        "max_edges_per_model": max_edges_per_model,
        "threads": int(
            config["parameters"].get("threads", 1)
        ),
        "operator_constraints_exact": (
            config.get(
                "qlattice_compatibility", {}
            ).get(
                "operator_sets_exactly_enforced",
                False,
            )
        ),
        "test_set_used_during_search": False,
        "failures": failures,
    }

    report_path = (
        output_dir / "relatorio_execucao_qlattice.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nLote finalizado.")
    print(f"Arquivos encontrados: {found}/{total}")
    print(f"Arquivos válidos: {valid_count}/{total}")
    print(f"Execuções puladas (já válidas): {skipped}")
    print(f"Falhas: {len(failures)}")
    print(f"Relatório: {report_path}")


def main() -> None:

    pre_parser = argparse.ArgumentParser(
        add_help=False
    )

    pre_parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
    )

    pre_args, _ = pre_parser.parse_known_args()

    config = load_config(pre_args.config)

    datasets = build_datasets(config)

    cv = config["experiment"]["cross_validation"]
    budget = config["experiment"]["budget"]
    params = config["parameters"]

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark QLattice configurado "
            "por config_base_qlattice.json."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=pre_args.config,
    )

    parser.add_argument(
        "--all-experiments",
        action="store_true",
    )

    parser.add_argument(
        "--dataset",
        choices=list(datasets),
    )

    parser.add_argument(
        "--scenario",
        choices=scenario_names(config),
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=list(cv["seeds"]),
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(
            1,
            int(cv["n_folds"]) + 1,
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(cv["seeds"]),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=int(budget["max_evaluations"]),
    )

    parser.add_argument(
        "--max-time-seconds",
        type=int,
        default=int(budget["max_time_seconds"]),
    )

    parser.add_argument(
        "--max-edges-per-model",
        type=int,
        default=int(
            params.get("max_edges_per_model", 10)
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.all_experiments:

        selected_scenarios = (
            [args.scenario]
            if args.scenario
            else scenario_names(config)
        )

        run_all(
            output_dir=args.output_dir,
            datasets=datasets,
            config=config,
            seeds=args.seeds,
            scenarios_to_run=selected_scenarios,
            max_evaluations=args.max_evaluations,
            max_time_seconds=args.max_time_seconds,
            max_edges_per_model=args.max_edges_per_model,
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
        datasets=datasets,
        config=config,
        max_evaluations=args.max_evaluations,
        max_time_seconds=args.max_time_seconds,
        max_edges_per_model=args.max_edges_per_model,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
