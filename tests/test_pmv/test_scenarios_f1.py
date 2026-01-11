from rnfe.pmv.experiments.scenarios_f1 import (
    run_f1_inductive_scenario_multi_seed,
    F1ScenarioBatchSummary,
)
import numpy as np

from rnfe.pmv.experiments.scenarios_f1 import (
    F1InductiveScenarioSpec,
    F1InductiveScenarioResult,
    _compute_fractal_observables_for_sequence,
    _build_phase1_inductive_dataset_from_sequences_pair,
    run_f1_inductive_scenario,
)


def test_compute_fractal_observables_for_sequence_basic():
    # Secuencia simple de estados 2x2:
    # X_t[0, 0] = t, X_t[0, 1] = 2t, resto ceros.
    T = 5
    H, W = 2, 2
    X_seq = np.zeros((T, H, W), dtype=float)
    for t in range(T):
        X_seq[t, 0, 0] = float(t)
        X_seq[t, 0, 1] = float(2 * t)

    # Nivel 0: celda (0,0); nivel 1: celda (0,1)
    mask0 = np.zeros((H, W), dtype=bool)
    mask0[0, 0] = True
    mask1 = np.zeros((H, W), dtype=bool)
    mask1[0, 1] = True

    g_all = _compute_fractal_observables_for_sequence(X_seq, [mask0, mask1])

    assert g_all.shape == (T, 2)
    for t in range(T):
        assert np.isclose(g_all[t, 0], float(t))
        assert np.isclose(g_all[t, 1], float(2 * t))


def test_build_phase1_inductive_dataset_from_sequences_pair_shapes():
    rng = np.random.default_rng(42)
    T_main = 40
    T_ood = 30
    H, W = 2, 2
    d = 3

    X_main = rng.normal(size=(T_main, H, W))
    X_ood = rng.normal(size=(T_ood, H, W))
    E_main = rng.normal(size=(T_main, d))
    E_ood = rng.normal(size=(T_ood, d))

    # Dos niveles que cubren todo el grid, por simplicidad
    mask0 = np.ones((H, W), dtype=bool)
    mask1 = np.zeros((H, W), dtype=bool)
    mask1[0, :] = True  # primera fila

    law_coeffs = [0.7, 0.3]
    window_size = 3
    n_train = 20
    n_test = 10
    n_ood = 10

    dataset = _build_phase1_inductive_dataset_from_sequences_pair(
        X_main=X_main,
        E_main=E_main,
        X_ood=X_ood,
        E_ood=E_ood,
        level_masks=[mask0, mask1],
        law_coeffs=law_coeffs,
        window_size=window_size,
        n_train=n_train,
        n_test=n_test,
        n_ood=n_ood,
    )

    assert dataset.X_train.shape == (n_train, window_size * d)
    assert dataset.y_train.shape == (n_train,)
    assert dataset.X_test.shape == (n_test, window_size * d)
    assert dataset.y_test.shape == (n_test,)
    assert dataset.X_ood.shape == (n_ood, window_size * d)
    assert dataset.y_ood.shape == (n_ood,)


def _stub_world_provider(
    condition: str,
    n_steps: int,
    rng: np.random.Generator,
):
    """
    Proveedor de mundo de juguete:

    - Genera una secuencia X_seq de forma (T, 2, 2) donde los valores
      siguen patrones simples en función de t.
    - Genera embeddings E_seq = [t, t^2, ruido] de dimensión d=3.

    La condición modifica ligeramente la escala para simular
    diferencias fractal / no fractal.
    """
    T = n_steps
    H, W = 2, 2
    d = 3
    X_seq = np.zeros((T, H, W), dtype=float)

    if "fractal" in condition:
        scale = 1.0
    else:
        scale = 1.5

    for t in range(T):
        val = scale * float(t)
        X_seq[t, 0, 0] = val
        X_seq[t, 0, 1] = 2.0 * val

    # Embeddings: [t, t^2, ruido]
    t_vals = np.arange(T, dtype=float)
    E_seq = np.zeros((T, d), dtype=float)
    E_seq[:, 0] = t_vals
    E_seq[:, 1] = t_vals ** 2
    E_seq[:, 2] = rng.normal(scale=0.1, size=(T,))

    return X_seq, E_seq


def test_run_f1_inductive_scenario_end_to_end():
    # Máscaras fractales sobre el grid 2x2
    H, W = 2, 2
    mask0 = np.zeros((H, W), dtype=bool)
    mask0[0, 0] = True
    mask1 = np.zeros((H, W), dtype=bool)
    mask1[0, 1] = True
    level_masks = [mask0, mask1]

    spec = F1InductiveScenarioSpec(
        scenario_id="F1_IND_boxworld_stub_v0",
        window_size=3,
        n_train=24,
        n_test=8,
        n_ood=8,
        law_coeffs=[1.0, 0.0],  # sólo usa el primer nivel
        lambda_cost=0.1,
        gamma_k=1.0,
        gamma_mdl=1.0,
    )

    rng = np.random.default_rng(123)

    result = run_f1_inductive_scenario(
        spec=spec,
        world_provider=_stub_world_provider,
        level_masks=level_masks,
        rng=rng,
    )

    assert isinstance(result, F1InductiveScenarioResult)
    assert result.spec.scenario_id == "F1_IND_boxworld_stub_v0"

    m_f = result.result_fractal.phase1_result.metrics
    m_n = result.result_nonfractal.phase1_result.metrics

    # Las métricas deben estar en rangos válidos.
    for m in (m_f, m_n):
        assert 0.0 <= m.s_law <= 1.0
        assert 0.0 <= m.s_ood <= 1.0
        assert 0.0 <= m.s_f1 <= 1.0

    # Las diferencias deben ser números finitos
    assert np.isfinite(result.delta_s_f1)
    assert np.isfinite(result.delta_e_in)
    assert np.isfinite(result.delta_e_ood)
def test_run_f1_inductive_scenario_multi_seed_basic():
    H, W = 2, 2
    level_masks = [
        np.zeros((H, W), dtype=bool),
        np.zeros((H, W), dtype=bool),
    ]
    level_masks[0][0, 0] = True
    level_masks[1][1, 1] = True

    spec = F1InductiveScenarioSpec(
        scenario_id="F1_IND_multi_seed_stub_v0",
        window_size=3,
        n_train=16,
        n_test=4,
        n_ood=4,
        law_coeffs=[1.0, 0.0],
        lambda_cost=0.1,
        gamma_k=1.0,
        gamma_mdl=1.0,
    )

    seeds = [1, 2, 3]

    summary = run_f1_inductive_scenario_multi_seed(
        spec=spec,
        world_provider=_stub_world_provider,  # el stub ya definido en este test
        level_masks=level_masks,
        seeds=seeds,
    )

    assert isinstance(summary, F1ScenarioBatchSummary)
    assert len(summary.runs) == len(seeds)
    assert list(summary.seeds) == seeds

    # Métricas agregadas en rangos razonables
    assert 0.0 <= summary.mean_s_f1_fractal <= 1.0
    assert 0.0 <= summary.mean_s_f1_nonfractal <= 1.0
    assert np.isfinite(summary.mean_delta_s_f1)
    assert np.isfinite(summary.std_delta_s_f1)

    for run in summary.runs:
        assert 0.0 <= run.s_f1_fractal <= 1.0
        assert 0.0 <= run.s_f1_nonfractal <= 1.0
        assert np.isfinite(run.delta_s_f1)
