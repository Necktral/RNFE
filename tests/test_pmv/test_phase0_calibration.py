# tests/test_pmv/test_phase0_calibration.py

import numpy as np

from rnfe.fmse.world.mini_world_boxes import BoxWorldConfig
from rnfe.core.mfm.mfm_state import MFMConfig
from rnfe.pmv.phases.phase0_calibration import (
    Phase0Config,
    Phase0CalibrationRunner,
)


def _make_phase0_config() -> Phase0Config:
    """
    Crea una configuración pequeña pero no trivial para pruebas de F0.
    """
    world_cfg = BoxWorldConfig(
        base_height=8,
        base_width=8,
        n_levels=2,
        seed=123,
    )

    mfm_cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=16,
        ttl_steps=32,
    )

    cfg = Phase0Config(
        world_config=world_cfg,
        mfm_config=mfm_cfg,
        embedding_dim=8,
        n_steps=10,
        seed=999,
        log_prefix="F0test",
    )
    return cfg


def test_phase0_runs_and_logs_basic_metrics():
    """
    F0 debe ejecutarse sin errores y registrar métricas básicas
    en el bus de telemetría.
    """
    cfg = _make_phase0_config()
    runner = Phase0CalibrationRunner(cfg)
    result = runner.run()

    # Estado final consistente con n_steps
    assert result.final_state_step == cfg.n_steps - 1
    assert result.world_state.t == cfg.n_steps - 1

    telemetry = result.telemetry
    metrics = telemetry.list_metrics()

    # Métricas esperadas
    assert f"{cfg.log_prefix}.n_levels" in metrics
    assert f"{cfg.log_prefix}.level_sizes" in metrics
    assert f"{cfg.log_prefix}.obs_norm_l2" in metrics
    assert f"{cfg.log_prefix}.obs_std" in metrics
    assert f"{cfg.log_prefix}.emb_norm_l2" in metrics
    assert f"{cfg.log_prefix}.mfm_occupancy_ratio" in metrics
    assert f"{cfg.log_prefix}.mfm_valid_ratio" in metrics

    # obs_norm_l2 debe tener un valor por paso
    steps, values = telemetry.get_series(
        f"{cfg.log_prefix}.obs_norm_l2"
    ).to_numpy()
    assert steps.shape == (cfg.n_steps,)
    assert values.shape == (cfg.n_steps,)
    assert np.all(steps == np.arange(cfg.n_steps))


def test_phase0_embedding_dimension_matches_config():
    """
    El embedding almacenado en MFM debe tener la dimensión
    especificada en embedding_dim.
    """
    cfg = _make_phase0_config()
    runner = Phase0CalibrationRunner(cfg)
    result = runner.run()

    emb = result.mfm.read_last(
        level=0,
        sector=0,
        channel="E",
        t_step=cfg.n_steps - 1,
    )
    assert emb is not None
    assert emb.shape == (cfg.embedding_dim,)


def test_phase0_is_deterministic_with_same_seed():
    """
    Con la misma configuración (incluida la semilla), dos ejecuciones
    independientes de F0 deben producir series de telemetría idénticas
    y embeddings finales idénticos.
    """
    cfg = _make_phase0_config()

    runner1 = Phase0CalibrationRunner(cfg)
    runner2 = Phase0CalibrationRunner(cfg)

    res1 = runner1.run()
    res2 = runner2.run()

    d1 = res1.telemetry.as_dict_numpy()
    d2 = res2.telemetry.as_dict_numpy()

    assert set(d1.keys()) == set(d2.keys())

    for name in d1.keys():
        steps1, vals1 = d1[name]
        steps2, vals2 = d2[name]

        assert np.all(steps1 == steps2)
        assert np.allclose(vals1, vals2)

    # Embeddings finales en MFM deben coincidir
    t_last = cfg.n_steps - 1
    emb1 = res1.mfm.read_last(0, 0, "E", t_step=t_last)
    emb2 = res2.mfm.read_last(0, 0, "E", t_step=t_last)

    assert emb1 is not None and emb2 is not None
    assert emb1.shape == emb2.shape
    assert np.allclose(emb1, emb2)
