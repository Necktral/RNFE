from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Importar métrica y score estándar
from rnfe.pmv.phases.phase1_unimodal import (
    Phase1Config,
    Phase1RawMetrics,
    Phase1Result,
    build_phase1_results,
)

@dataclass(frozen=True)
class Phase1DeductiveConfig:
    """
    Configuración para experimento Fase 1 modo deductivo.
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
class Phase1DeductiveDataset:
    """
    Dataset para F1-DED: ventanas de embeddings y targets escalares.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_ood: np.ndarray
    y_ood: np.ndarray

@dataclass(frozen=True)
class Phase1DeductiveExperimentResult:
    """
    Resultado completo de experimento F1-DED.
    """
    config: Phase1DeductiveConfig
    raw_metrics: Phase1RawMetrics
    result: Phase1Result


def run_phase1_deductive_experiment(
    config: Phase1DeductiveConfig,
    dataset: Phase1DeductiveDataset,
) -> Phase1DeductiveExperimentResult:
    """
    Ejecuta experimento deductivo y calcula métricas + score S_F1.
    """
    # Aquí iría la lógica deductiva real (placeholder)
    # Para ejemplo, usamos una regresión lineal simple sobre X_train/y_train
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(dataset.X_train, dataset.y_train)
    y_pred_test = model.predict(dataset.X_test)
    y_pred_ood = model.predict(dataset.X_ood)

    # Métricas crudas
    e_in = np.mean((y_pred_test - dataset.y_test) ** 2)
    e_ood = np.mean((y_pred_ood - dataset.y_ood) ** 2)
    mdl_proxy = np.sum(model.coef_ ** 2)
    k_interactions = dataset.X_train.shape[1]
    cost_total = np.abs(model.coef_).sum()

    raw_metrics = Phase1RawMetrics(
        e_in=e_in,
        e_ood=e_ood,
        mdl_proxy=mdl_proxy,
        k_interactions=k_interactions,
        cost_total=cost_total,
    )
    result = build_phase1_results(
        config=Phase1Config(
            window_size=config.window_size,
            embedding_dim=config.embedding_dim,
            train_steps=config.train_steps,
            test_steps=config.test_steps,
            ood_steps=config.ood_steps,
            lambda_cost=config.lambda_cost,
            gamma_k=config.gamma_k,
            gamma_mdl=config.gamma_mdl,
        ),
        raw_metrics=raw_metrics,
    )
    return Phase1DeductiveExperimentResult(
        config=config,
        raw_metrics=raw_metrics,
        result=result,
    )
