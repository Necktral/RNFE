import math
import pytest

from rnfe.pmv.phases.phase1_unimodal import (
    _clip01,
    Phase1Config,
    Phase1RawMetrics,
    compute_s_law,
    compute_s_ood,
    compute_s_mdl,
    compute_s_eff,
    compute_s_cost,
    compute_phase1_metrics_for_all_modes,
    build_phase1_results,
)


# ---------------------------------------------------------------------------
# Tests de utilidades básicas
# ---------------------------------------------------------------------------

def test_clip01_extremos():
    # Menor que 0 se recorta a 0
    assert _clip01(-0.5) == 0.0
    # Mayor que 1 se recorta a 1
    assert _clip01(1.5) == 1.0
    # Dentro de rango se mantiene
    assert _clip01(0.3) == 0.3


def test_compute_phase1_metrics_empty_raises():
    # Diccionario vacío -> error controlado
    with pytest.raises(ValueError):
        compute_phase1_metrics_for_all_modes({})


# ---------------------------------------------------------------------------
# Tests de normalización elemental
# ---------------------------------------------------------------------------

def test_s_law_normalization_basic_cases():
    e_base = 1.0

    # Igual al baseline -> S_law ~ 0
    s_same = compute_s_law(e_in=e_base, e_in_base=e_base)
    assert 0.0 <= s_same <= 0.1

    # Error nulo -> S_law ~ 1
    s_zero = compute_s_law(e_in=0.0, e_in_base=e_base)
    assert 0.9 <= s_zero <= 1.0

    # Peor que el baseline -> S_law recortado a 0
    s_worse = compute_s_law(e_in=2.0, e_in_base=e_base)
    assert s_worse == 0.0


def test_s_ood_normalization_basic_cases():
    e_base = 2.0

    s_same = compute_s_ood(e_ood=e_base, e_ood_base=e_base)
    assert 0.0 <= s_same <= 0.1

    s_zero = compute_s_ood(e_ood=0.0, e_ood_base=e_base)
    assert 0.9 <= s_zero <= 1.0

    s_worse = compute_s_ood(e_ood=3.0, e_ood_base=e_base)
    assert s_worse == 0.0


def test_s_mdl_relative_min_max():
    mdl_min = 10.0
    mdl_max = 110.0

    # El más compacto debe estar cerca de 1
    s_min = compute_s_mdl(mdl=mdl_min, mdl_min=mdl_min, mdl_max=mdl_max)
    assert 0.9 <= s_min <= 1.0

    # El más complejo debe estar cerca de 0
    s_max = compute_s_mdl(mdl=mdl_max, mdl_min=mdl_min, mdl_max=mdl_max)
    assert 0.0 <= s_max <= 0.1

    # Un valor intermedio debe quedar aproximadamente en el medio
    mdl_mid = (mdl_min + mdl_max) / 2.0
    s_mid = compute_s_mdl(mdl=mdl_mid, mdl_min=mdl_min, mdl_max=mdl_max)
    assert 0.4 <= s_mid <= 0.6


def test_s_eff_relative_min_max():
    k_min = 10
    k_max = 110

    s_min = compute_s_eff(k=k_min, k_min=k_min, k_max=k_max)
    assert 0.9 <= s_min <= 1.0

    s_max = compute_s_eff(k=k_max, k_min=k_min, k_max=k_max)
    assert 0.0 <= s_max <= 0.1

    k_mid = (k_min + k_max) // 2
    s_mid = compute_s_eff(k=k_mid, k_min=k_min, k_max=k_max)
    assert 0.4 <= s_mid <= 0.6


def test_s_cost_relative_min_max():
    c_min = 100.0
    c_max = 200.0

    s_min = compute_s_cost(cost=c_min, cost_min=c_min, cost_max=c_max)
    assert 0.0 <= s_min <= 0.1

    s_max = compute_s_cost(cost=c_max, cost_min=c_min, cost_max=c_max)
    assert 0.9 <= s_max <= 1.0

    c_mid = (c_min + c_max) / 2.0
    s_mid = compute_s_cost(cost=c_mid, cost_min=c_min, cost_max=c_max)
    assert 0.4 <= s_mid <= 0.6


# ---------------------------------------------------------------------------
# Tests sobre la cascada y la métrica global S_F1
# ---------------------------------------------------------------------------

def test_compute_phase1_metrics_cascade_comprehension_dominates():
    """
    La comprensión (S_law, S_ood) debe dominar sobre eficiencia
    y simplicidad en la métrica final S_F1.
    """
    # Baseline común
    e_in_base = 1.0
    e_ood_base = 1.0

    # Modo "bueno": comprensión alta, aunque sea algo más caro y complejo
    raw_good = Phase1RawMetrics(
        e_in=0.1,
        e_in_base=e_in_base,
        e_ood=0.1,
        e_ood_base=e_ood_base,
        mdl=100.0,
        k_interactions=80,
        cost=150.0,
    )

    # Modo "malo": no mejora al baseline, aunque sea muy eficiente
    raw_bad = Phase1RawMetrics(
        e_in=1.0,
        e_in_base=e_in_base,
        e_ood=1.0,
        e_ood_base=e_ood_base,
        mdl=10.0,
        k_interactions=20,
        cost=80.0,
    )

    raw_by_mode = {"good": raw_good, "bad": raw_bad}
    metrics_by_mode = compute_phase1_metrics_for_all_modes(
        raw_by_mode=raw_by_mode,
        lambda_cost=0.1,
    )

    m_good = metrics_by_mode["good"]
    m_bad = metrics_by_mode["bad"]

    # El modo "malo" no debe tener comprensión
    assert m_bad.s_law == 0.0
    assert m_bad.s_ood == 0.0
    assert m_bad.s_core == 0.0

    # El modo "bueno" debe superar claramente al "malo" en S_F1
    assert m_good.s_f1 > m_bad.s_f1


def _make_two_raw_modes():
    # Modo A: un poco mejor, un poco más caro
    raw_a = Phase1RawMetrics(
        e_in=0.2,
        e_in_base=1.0,
        e_ood=0.2,
        e_ood_base=1.0,
        mdl=20.0,
        k_interactions=50,
        cost=100.0,
    )
    # Modo B: un poco peor, más barato, etc.
    raw_b = Phase1RawMetrics(
        e_in=0.3,
        e_in_base=1.0,
        e_ood=0.3,
        e_ood_base=1.0,
        mdl=80.0,
        k_interactions=120,
        cost=10.0,
    )
    return {"A": raw_a, "B": raw_b}


def test_factor_cost_floor_and_non_negative_s_f1():
    """
    Con lambda_cost muy grande se fuerza factor_cost < 0,
    que debe ser clipeado a 0, y aún así S_F1 no debe ser negativo.
    """
    raw_by_mode = _make_two_raw_modes()

    # lambda_cost alto -> factor_cost se hace negativo antes de clip
    metrics = compute_phase1_metrics_for_all_modes(
        raw_by_mode=raw_by_mode,
        lambda_cost=2.0,
    )

    for m in metrics.values():
        # s_f1 no debe ser negativo incluso si factor_cost se recortó a 0
        assert m.s_f1 >= 0.0


def test_s_f1_clipping_upper_bound_with_negative_lambda_cost():
    """
    Usando un lambda_cost negativo (caso patológico) podemos forzar
    raw_s_f1 > 1. La implementación debe recortar S_F1 a 1.0.
    """
    raw_by_mode = _make_two_raw_modes()

    # lambda_cost negativo -> factor_cost > 1, raw_s_f1 puede exceder 1
    metrics = compute_phase1_metrics_for_all_modes(
        raw_by_mode=raw_by_mode,
        lambda_cost=-10.0,
    )

    for m in metrics.values():
        assert 0.0 <= m.s_f1 <= 1.0


def test_build_phase1_results_single_mode_degenerate_spans():
    """
    Caso extremo PMV v0: solo un modo.

    Los mínimos y máximos coinciden, pero la normalización no debe
    explotar ni producir NaN. El modo único debería quedar con
    métricas coherentes.
    """
    config = Phase1Config(mode_name="IND", lambda_cost=0.1)

    raw = Phase1RawMetrics(
        e_in=0.5,
        e_in_base=1.0,
        e_ood=0.7,
        e_ood_base=1.0,
        mdl=42.0,
        k_interactions=100,
        cost=123.0,
    )

    results = build_phase1_results(config, {"IND": raw})
    assert "IND" in results

    res = results["IND"]
    assert res.mode_name == "IND"
    assert res.config.mode_name == "IND"

    # Métricas en rangos válidos
    m = res.metrics
    assert 0.0 <= m.s_law <= 1.0
    assert 0.0 <= m.s_ood <= 1.0
    assert 0.0 <= m.s_mdl <= 1.0
    assert 0.0 <= m.s_eff <= 1.0
    assert 0.0 <= m.s_cost <= 1.0
    assert 0.0 <= m.s_core <= 1.0
    assert 0.0 <= m.s_aux <= 1.0
    assert not math.isnan(m.s_f1)
