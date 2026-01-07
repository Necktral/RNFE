# tests/test_mini_world_boxes.py

import numpy as np
import pytest

from rnfe.fmse.world.mini_world_boxes import (
    BoxWorldConfig,
    BoxWorld,
)


def test_level_shapes_monotonic_and_positive():
    """
    Las formas de los niveles deben ser positivas y no crecer
    al subir niveles (coarse → tamaños menores o iguales).
    """
    cfg = BoxWorldConfig(base_height=32, base_width=64, n_levels=4)
    shapes = cfg.level_shape(0), cfg.level_shape(1), cfg.level_shape(2), cfg.level_shape(3)

    prev_h, prev_w = shapes[0]
    assert prev_h > 0 and prev_w > 0

    for (h, w) in shapes[1:]:
        assert h > 0 and w > 0
        # Cada nivel no debe ser más grande que el anterior
        assert h <= prev_h
        assert w <= prev_w
        prev_h, prev_w = h, w


def test_reset_is_deterministic_with_same_seed():
    """
    Con la misma semilla, reset() debe producir estados iniciales idénticos.
    """
    cfg = BoxWorldConfig(base_height=16, base_width=16, n_levels=3, seed=123)

    env1 = BoxWorld(cfg)
    env2 = BoxWorld(cfg)

    s1 = env1.reset(seed=999)
    s2 = env2.reset(seed=999)

    assert s1.t == 0
    assert s2.t == 0
    assert len(s1.levels) == len(s2.levels)

    for lvl1, lvl2 in zip(s1.levels, s2.levels):
        assert lvl1.shape == lvl2.shape
        assert np.allclose(lvl1, lvl2)


def test_step_changes_state_but_preserves_shape():
    """
    step() debe cambiar los valores del estado (en promedio),
    pero mantener las formas de los niveles.
    """
    cfg = BoxWorldConfig(
        base_height=16,
        base_width=16,
        n_levels=2,
        temporal_alpha=0.8,
        seed=42,
    )
    env = BoxWorld(cfg)
    s0 = env.reset()
    s1 = env.step()

    assert s0.t == 0
    assert s1.t == 1

    assert len(s0.levels) == len(s1.levels)

    for lvl0, lvl1 in zip(s0.levels, s1.levels):
        # La forma debe ser la misma
        assert lvl0.shape == lvl1.shape

        # Debe haber cambio significativo (no todo igual)
        diff = np.mean(np.abs(lvl1 - lvl0))
        assert diff > 1e-3


def test_observe_flat_concatenates_all_levels():
    """
    observe_flat() debe concatenar todas las grillas de todos
    los niveles en un único vector 1D.
    """
    cfg = BoxWorldConfig(base_height=8, base_width=8, n_levels=3, seed=7)
    env = BoxWorld(cfg)
    state = env.reset()

    obs = env.observe_flat()
    assert obs.ndim == 1

    expected_len = 0
    for lvl in state.levels:
        expected_len += lvl.size

    assert obs.shape[0] == expected_len

    # Valores dentro de un rango razonable tras normalización y tanh
    assert np.all(np.isfinite(obs))


def test_step_without_reset_raises():
    """
    Llamar a step() sin reset() debe lanzar un error claro.
    """
    cfg = BoxWorldConfig()
    env = BoxWorld(cfg)

    with pytest.raises(RuntimeError):
        _ = env.step()


def test_current_state_without_reset_raises():
    """
    Llamar a current_state() sin reset() debe lanzar un error claro.
    """
    cfg = BoxWorldConfig()
    env = BoxWorld(cfg)

    with pytest.raises(RuntimeError):
        _ = env.current_state()

from rnfe.fmse.world.mini_world_boxes import BoxWorldConfig, BoxWorld


def test_level_shape_raises_for_invalid_level():
    cfg = BoxWorldConfig(base_height=8, base_width=8, n_levels=2)

    with pytest.raises(ValueError):
        cfg.level_shape(-1)

    with pytest.raises(ValueError):
        cfg.level_shape(2)


def test_boxworld_step_rejects_temporal_alpha_out_of_range():
    # temporal_alpha menor que 0
    cfg1 = BoxWorldConfig(
        base_height=8,
        base_width=8,
        n_levels=1,
        temporal_alpha=-0.1,
    )
    env1 = BoxWorld(cfg1)
    env1.reset()
    with pytest.raises(ValueError):
        env1.step()

    # temporal_alpha mayor que 1
    cfg2 = BoxWorldConfig(
        base_height=8,
        base_width=8,
        n_levels=1,
        temporal_alpha=1.5,
    )
    env2 = BoxWorld(cfg2)
    env2.reset()
    with pytest.raises(ValueError):
        env2.step()


def test_boxworld_minimal_shapes_do_not_crash():
    # Caso extremo: base_height=1, base_width=1, varios niveles
    cfg = BoxWorldConfig(base_height=1, base_width=1, n_levels=3, seed=123)
    env = BoxWorld(cfg)
    state0 = env.reset()

    # Todas las formas deben ser (1, 1) por el mínimo
    shapes = env.level_shapes()
    assert len(shapes) == 3
    for h, w in shapes:
        assert h == 1
        assert w == 1

    # Avanzar algunos pasos no debe lanzar errores
    for _ in range(5):
        state = env.step()
        assert state.t >= state0.t
