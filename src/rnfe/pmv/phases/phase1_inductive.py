from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rnfe.pmv.phases.phase1_unimodal import (
    Phase1Config,
    Phase1RawMetrics,
    Phase1Result,
    build_phase1_results,
)


@dataclass(frozen=True)
class Phase1InductiveConfig:
    """
    Configuración para un experimento de Fase 1 con modo inductivo.

    Esta configuración es agnóstica del mini-mundo concreto: asume que
    algún componente externo construirá un Phase1InductiveDataset que
    respete window_size y embedding_dim.
    """
    window_size: int
    embedding_dim: int
    train_steps: int
    test_steps: int
    ood_steps: int
    lambda_cost: float = 0.1
    gamma_k: float = 1.0
    gamma_mdl: float = 1.0


@dataclass(frozen=True)
class Phase1InductiveDataset:
    """
    Dataset para F1-IND: ventanas de embeddings y targets escalares.

    Las matrices X_* deben tener forma (n_muestras, k * d), donde
    k = window_size y d = embedding_dim. Los vectores y_* tienen
    forma (n_muestras,).
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_ood: np.ndarray
    y_ood: np.ndarray


@dataclass(frozen=True)
class Phase1InductiveExperimentResult:
    """
    Resultado completo de un experimento F1-IND.

    Contiene tanto los parámetros aprendidos como las métricas crudas
    y normalizadas de F1.
    """
    config: Phase1InductiveConfig
    dataset: Phase1InductiveDataset
    w: np.ndarray
    b: float
    raw_metrics: Phase1RawMetrics
    phase1_result: Phase1Result


def train_inductive_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    l2_reg: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Entrena un regresor lineal sobre X_train -> y_train.

    Usa solución cerrada de mínimos cuadrados (ridge opcional).
    Devuelve (w, b) con w de forma (n_features,) y b escalar.
    """
    if X_train.ndim != 2:
        raise ValueError("X_train debe ser 2D (n_muestras, n_features).")
    if y_train.ndim not in (1, 2):
        raise ValueError("y_train debe ser 1D o 2D.")
    n_samples, n_features = X_train.shape
    if n_samples == 0:
        raise ValueError("Se requiere al menos una muestra para entrenar.")

    y = y_train.reshape(n_samples, 1)
    X_aug = np.concatenate(
        [X_train, np.ones((n_samples, 1), dtype=X_train.dtype)],
        axis=1,
    )

    A = X_aug.T @ X_aug
    if l2_reg > 0.0:
        A = A + l2_reg * np.eye(n_features + 1, dtype=X_train.dtype)
    b_vec = X_aug.T @ y

    w_full = np.linalg.solve(A, b_vec)
    w_full = w_full.reshape(n_features + 1)
    w = w_full[:-1]
    b_bias = float(w_full[-1])
    return w, b_bias


def predict_inductive_model(
    X: np.ndarray,
    w: np.ndarray,
    b: float,
) -> np.ndarray:
    """
    Aplica el modelo lineal a una matriz de entradas X.
    """
    if X.ndim != 2:
        raise ValueError("X debe ser 2D (n_muestras, n_features).")
    return X @ w + b


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Error cuadrático medio entre y_true y y_pred.

    Ambos deben tener la misma forma y no estar vacíos.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Las formas de y_true y y_pred deben coincidir.")
    if y_true.size == 0:
        raise ValueError("No se puede calcular error con vectores vacíos.")
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def compute_errors_for_model_and_baseline(
    dataset: Phase1InductiveDataset,
    w: np.ndarray,
    b: float,
) -> Tuple[float, float, float, float]:
    """
    Calcula errores in-distribution y OOD para el modelo y el baseline.

    El baseline es un predictor constante igual a la media de y_train.
    """
    y_train = dataset.y_train
    if y_train.size == 0:
        raise ValueError("El dataset de entrenamiento está vacío.")
    y_mean_train = float(np.mean(y_train))

    # In-distribution
    y_pred_test = predict_inductive_model(dataset.X_test, w, b)
    e_in = mean_squared_error(dataset.y_test, y_pred_test)
    y_pred_test_base = np.full_like(dataset.y_test, fill_value=y_mean_train)
    e_in_base = mean_squared_error(dataset.y_test, y_pred_test_base)

    # Fuera de distribución
    y_pred_ood = predict_inductive_model(dataset.X_ood, w, b)
    e_ood = mean_squared_error(dataset.y_ood, y_pred_ood)
    y_pred_ood_base = np.full_like(dataset.y_ood, fill_value=y_mean_train)
    e_ood_base = mean_squared_error(dataset.y_ood, y_pred_ood_base)

    return e_in, e_in_base, e_ood, e_ood_base


def estimate_mdl_k_c(
    config: Phase1InductiveConfig,
    dataset: Phase1InductiveDataset,
    w: np.ndarray,
    b: float,
) -> Tuple[float, int, float]:
    """
    Estima MDL, número de interacciones K y coste C para F1-IND.

    MDL se aproxima como el número de parámetros (w + b).
    K se toma como el número total de ejemplos usados.
    C combina K y MDL con pesos gamma_k y gamma_mdl.
    """
    n_features = dataset.X_train.shape[1]
    mdl = float(n_features + 1)

    total_examples = (
        dataset.X_train.shape[0]
        + dataset.X_test.shape[0]
        + dataset.X_ood.shape[0]
    )
    k = int(total_examples)

    cost = config.gamma_k * float(k) + config.gamma_mdl * mdl
    return mdl, k, cost


def run_phase1_inductive_experiment(
    config: Phase1InductiveConfig,
    dataset: Phase1InductiveDataset,
    l2_reg: float = 0.0,
) -> Phase1InductiveExperimentResult:
    """
    Ejecuta un experimento completo F1-IND sobre un dataset ya construido.

    Este nivel no sabe cómo se generó el dataset; solo asume que las
    dimensiones son coherentes con window_size y embedding_dim.
    """
    n_features = dataset.X_train.shape[1]
    expected_features = config.window_size * config.embedding_dim
    if n_features != expected_features:
        raise ValueError(
            "Número de features incompatible con window_size * embedding_dim: "
            f"{n_features} != {expected_features}."
        )

    w, b_bias = train_inductive_model(dataset.X_train, dataset.y_train, l2_reg=l2_reg)
    e_in, e_in_base, e_ood, e_ood_base = compute_errors_for_model_and_baseline(
        dataset, w, b_bias
    )
    mdl, k, cost = estimate_mdl_k_c(config, dataset, w, b_bias)

    raw = Phase1RawMetrics(
        e_in=e_in,
        e_in_base=e_in_base,
        e_ood=e_ood,
        e_ood_base=e_ood_base,
        mdl=mdl,
        k_interactions=k,
        cost=cost,
    )

    f1_config = Phase1Config(mode_name="IND", lambda_cost=config.lambda_cost)
    results_by_mode = build_phase1_results(f1_config, {"IND": raw})
    phase1_result = results_by_mode["IND"]

    experiment_result = Phase1InductiveExperimentResult(
        config=config,
        dataset=dataset,
        w=w,
        b=b_bias,
        raw_metrics=raw,
        phase1_result=phase1_result,
    )
    return experiment_result
