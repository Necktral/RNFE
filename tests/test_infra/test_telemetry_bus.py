# tests/test_infra/test_telemetry_bus.py

import numpy as np
import pytest

from rnfe.infra.telemetry_bus import (
    TelemetryConfig,
    TelemetryBus,
    TelemetrySeries,
)


def test_log_scalar_and_retrieve_series():
    """
    Registrar una métrica escalar varias veces y recuperarla como
    serie NumPy debe producir formas coherentes.
    """
    bus = TelemetryBus()

    for t in range(5):
        bus.log_scalar("rho_J", step=t, value=0.9 + 0.01 * t)

    assert bus.has_metric("rho_J")
    names = bus.list_metrics()
    assert "rho_J" in names

    series = bus.get_series("rho_J")
    assert isinstance(series, TelemetrySeries)
    assert series.name == "rho_J"
    assert series.is_vector is False
    assert series.dim == 1
    assert len(series.steps) == 5
    assert len(series.values) == 5

    steps, values = series.to_numpy()
    assert steps.shape == (5,)
    assert values.shape == (5,)
    assert np.all(steps == np.arange(5))
    assert np.allclose(values, 0.9 + 0.01 * np.arange(5))


def test_log_vector_and_schema_consistency():
    """
    Una métrica vectorial debe mantener dimensión fija. Intentar
    registrar otro vector con dimensión diferente debe fallar.
    """
    bus = TelemetryBus()

    v1 = np.array([1.0, 2.0], dtype=np.float32)
    v2 = np.array([3.0, 4.0], dtype=np.float32)
    v_wrong = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    bus.log_vector("edge_state", step=0, value=v1)
    bus.log_vector("edge_state", step=1, value=v2)

    # Dimensión debe ser 2 (como v1 y v2)
    series = bus.get_series("edge_state")
    assert series.is_vector
    assert series.dim == 2

    # Registrar vector con dimensión incompatible debe dar error
    with pytest.raises(ValueError):
        bus.log_vector("edge_state", step=2, value=v_wrong)


def test_incompatible_scalar_vs_vector():
    """
    Si una métrica se registró como escalar, no se permite luego
    registrar valores vectoriales con el mismo nombre, y viceversa.
    """
    bus = TelemetryBus()

    bus.log_scalar("Io", step=0, value=1.23)

    with pytest.raises(TypeError):
        bus.log_vector("Io", step=1, value=np.array([1.0, 2.0], dtype=np.float32))

    bus2 = TelemetryBus()
    bus2.log_vector("fitness", step=0, value=np.array([0.1, 0.2], dtype=np.float32))

    with pytest.raises(TypeError):
        bus2.log_scalar("fitness", step=1, value=0.5)


def test_freeze_schema_blocks_new_metrics():
    """
    Cuando el esquema está congelado, no se pueden añadir métricas nuevas,
    solo actualizar las métricas existentes.
    """
    bus = TelemetryBus()

    bus.log_scalar("rho_J", step=0, value=0.95)
    bus.log_scalar("vram_usage", step=0, value=0.5)

    bus.freeze_schema()

    # Estas métricas ya existen, se pueden seguir registrando
    bus.log_scalar("rho_J", step=1, value=0.96)

    # Métrica nueva debe disparar KeyError
    with pytest.raises(KeyError):
        bus.log_scalar("new_metric", step=0, value=1.0)


def test_as_dict_numpy_returns_all_series():
    """
    as_dict_numpy debe agrupar todas las métricas registradas en
    un diccionario nombre -> (steps, values).
    """
    bus = TelemetryBus()

    # Métrica escalar
    for t in range(3):
        bus.log_scalar("rho_J", step=t, value=1.0 + 0.1 * t)

    # Métrica vectorial
    for t in range(2):
        v = np.array([t, t + 1], dtype=np.float32)
        bus.log_vector("edge_vec", step=t, value=v)

    d = bus.as_dict_numpy()
    assert "rho_J" in d
    assert "edge_vec" in d

    steps_rho, vals_rho = d["rho_J"]
    assert steps_rho.shape == (3,)
    assert vals_rho.shape == (3,)

    steps_edge, vals_edge = d["edge_vec"]
    assert steps_edge.shape == (2,)
    assert vals_edge.shape == (2, 2)


def test_log_scalar_rejects_negative_step():
    bus = TelemetryBus()
    with pytest.raises(ValueError):
        bus.log_scalar("test_metric", step=-1, value=0.0)


def test_log_vector_rejects_empty_vector():
    bus = TelemetryBus()
    empty = np.array([], dtype=np.float32)
    with pytest.raises(ValueError):
        bus.log_vector("vec_metric", step=0, value=empty)


def test_get_series_raises_for_unknown_metric():
    bus = TelemetryBus()
    with pytest.raises(KeyError):
        _ = bus.get_series("unknown_metric")


def test_unfreeze_schema_allows_new_metrics_after_freeze():
    bus = TelemetryBus()

    bus.log_scalar("rho_J", step=0, value=0.9)
    bus.freeze_schema()

    # Aquí no se puede crear una métrica nueva
    with pytest.raises(KeyError):
        bus.log_scalar("new_metric", step=0, value=1.0)

    # Al des-congelar, sí se puede
    bus.unfreeze_schema()
    bus.log_scalar("new_metric", step=1, value=1.23)
    assert bus.has_metric("new_metric")
