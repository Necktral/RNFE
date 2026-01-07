import numpy as np
import pytest

from rnfe.pmv.phases.phase1_inductive import (
    Phase1InductiveConfig,
    Phase1InductiveDataset,
    train_inductive_model,
    predict_inductive_model,
    mean_squared_error,
    compute_errors_for_model_and_baseline,
    estimate_mdl_k_c,
    run_phase1_inductive_experiment,
)


def _make_synthetic_dataset(
    n_train: int = 64,
    n_test: int = 32,
    n_ood: int = 32,
    n_features: int = 4,
    noise_std: float = 0.01,
) -> tuple[Phase1InductiveDataset, np.ndarray, float]:
    """
    Genera un dataset sintético donde la relación es lineal y el modelo
    lineal debería superar claramente al baseline.
    """
    rng = np.random.default_rng(1234)
    w_true = rng.normal(size=(n_features,))
    b_true = 0.5

    def make_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        X = rng.normal(size=(n_samples, n_features))
        y = X @ w_true + b_true + rng.normal(
            scale=noise_std,
            size=(n_samples,),
        )
        return X, y

    X_train, y_train = make_split(n_train)
    X_test, y_test = make_split(n_test)

    # Para OOD cambiamos la escala de las entradas
    X_ood = rng.normal(loc=0.0, scale=2.0, size=(n_ood, n_features))
    y_ood = X_ood @ w_true + b_true + rng.normal(
        scale=noise_std,
        size=(n_ood,),
    )

    dataset = Phase1InductiveDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_ood=X_ood,
        y_ood=y_ood,
    )
    return dataset, w_true, b_true


def test_train_and_predict_inductive_model_recovers_weights():
    dataset, w_true, b_true = _make_synthetic_dataset()
    w_hat, b_hat = train_inductive_model(
        dataset.X_train,
        dataset.y_train,
        l2_reg=1e-3,
    )

    y_pred = predict_inductive_model(dataset.X_test, w_hat, b_hat)
    y_true = dataset.X_test @ w_true + b_true
    mse = mean_squared_error(y_true, y_pred)

    assert mse < 0.05


def test_train_inductive_model_raises_on_empty_dataset():
    X_empty = np.zeros((0, 3), dtype=float)
    y_empty = np.zeros((0,), dtype=float)

    with pytest.raises(ValueError):
        train_inductive_model(X_empty, y_empty)


def test_mean_squared_error_basic_and_errors():
    y_true = np.array([0.0, 1.0, 2.0], dtype=float)
    y_pred = np.array([0.0, 1.0, 3.0], dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    # Un error claro: solo la última posición difiere en 1 unidad
    # MSE = (0^2 + 0^2 + 1^2) / 3 = 1/3
    assert np.isclose(mse, 1.0 / 3.0)

    # Formas distintas -> error
    with pytest.raises(ValueError):
        mean_squared_error(y_true, y_pred[:-1])

    # Vectores vacíos -> error
    empty = np.zeros((0,), dtype=float)
    with pytest.raises(ValueError):
        mean_squared_error(empty, empty)


def test_compute_errors_model_better_than_baseline():
    dataset, _, _ = _make_synthetic_dataset()
    w_hat, b_hat = train_inductive_model(
        dataset.X_train,
        dataset.y_train,
        l2_reg=1e-3,
    )

    e_in, e_in_base, e_ood, e_ood_base = compute_errors_for_model_and_baseline(
        dataset,
        w_hat,
        b_hat,
    )

    assert e_in < e_in_base
    assert e_ood < e_ood_base


def test_estimate_mdl_k_c_values():
    dataset, _, _ = _make_synthetic_dataset()
    config = Phase1InductiveConfig(
        window_size=1,
        embedding_dim=dataset.X_train.shape[1],
        train_steps=dataset.X_train.shape[0],
        test_steps=dataset.X_test.shape[0],
        ood_steps=dataset.X_ood.shape[0],
        lambda_cost=0.1,
        gamma_k=2.0,
        gamma_mdl=3.0,
    )

    w_hat, b_hat = train_inductive_model(
        dataset.X_train,
        dataset.y_train,
        l2_reg=1e-3,
    )
    mdl, k, cost = estimate_mdl_k_c(config, dataset, w_hat, b_hat)

    expected_mdl = dataset.X_train.shape[1] + 1
    expected_k = (
        dataset.X_train.shape[0]
        + dataset.X_test.shape[0]
        + dataset.X_ood.shape[0]
    )
    expected_cost = config.gamma_k * float(expected_k) + config.gamma_mdl * float(expected_mdl)

    assert mdl == expected_mdl
    assert k == expected_k
    assert cost == expected_cost


def test_run_phase1_inductive_experiment_integrates_with_f1_metrics():
    dataset, _, _ = _make_synthetic_dataset()
    config = Phase1InductiveConfig(
        window_size=1,
        embedding_dim=dataset.X_train.shape[1],
        train_steps=dataset.X_train.shape[0],
        test_steps=dataset.X_test.shape[0],
        ood_steps=dataset.X_ood.shape[0],
        lambda_cost=0.1,
        gamma_k=1.0,
        gamma_mdl=1.0,
    )

    result = run_phase1_inductive_experiment(
        config,
        dataset,
        l2_reg=1e-3,
    )

    assert result.phase1_result.mode_name == "IND"
    m = result.phase1_result.metrics

    # Comprensión no trivial: el modelo supera al baseline
    assert m.s_law > 0.0
    assert m.s_ood > 0.0

    # Métrica global en rango
    assert 0.0 <= m.s_f1 <= 1.0


def test_run_phase1_inductive_experiment_raises_on_feature_mismatch():
    dataset, _, _ = _make_synthetic_dataset()
    # Config mal configurado a propósito: embedding_dim distinto
    config = Phase1InductiveConfig(
        window_size=1,
        embedding_dim=dataset.X_train.shape[1] + 1,
        train_steps=dataset.X_train.shape[0],
        test_steps=dataset.X_test.shape[0],
        ood_steps=dataset.X_ood.shape[0],
        lambda_cost=0.1,
        gamma_k=1.0,
        gamma_mdl=1.0,
    )

    with pytest.raises(ValueError):
        run_phase1_inductive_experiment(
            config,
            dataset,
            l2_reg=1e-3,
        )
