# tests/test_core/test_mfm_state.py

import numpy as np
import pytest

from rnfe.core.mfm.mfm_state import (
    MFMConfig,
    MFM,
    CHANNELS_DEFAULT,
    MFMUsageStats,
)


def test_mfm_config_basic_validation():
    """
    Verifica que MFMConfig valide parámetros básicos.
    """
    cfg = MFMConfig(
        n_levels=3,
        n_sectors_per_level=4,
        capacity_per_sector_per_channel=8,
        ttl_steps=16,
    )
    assert cfg.n_levels == 3
    assert cfg.n_sectors_per_level == 4
    assert cfg.capacity_per_sector_per_channel == 8
    assert cfg.ttl_steps == 16
    assert cfg.channels == CHANNELS_DEFAULT

    with pytest.raises(ValueError):
        _ = MFMConfig(
            n_levels=0,
            n_sectors_per_level=4,
            capacity_per_sector_per_channel=8,
            ttl_steps=16,
        )

    with pytest.raises(ValueError):
        _ = MFMConfig(
            n_levels=3,
            n_sectors_per_level=0,
            capacity_per_sector_per_channel=8,
            ttl_steps=16,
        )

    with pytest.raises(ValueError):
        _ = MFMConfig(
            n_levels=3,
            n_sectors_per_level=4,
            capacity_per_sector_per_channel=0,
            ttl_steps=16,
        )

    with pytest.raises(ValueError):
        _ = MFMConfig(
            n_levels=3,
            n_sectors_per_level=4,
            capacity_per_sector_per_channel=8,
            ttl_steps=0,
        )


def test_mfm_write_and_read_last_basic():
    """
    Escribir en un slot y leerlo con read_last debe devolver exactamente
    el mismo vector.
    """
    cfg = MFMConfig(
        n_levels=2,
        n_sectors_per_level=3,
        capacity_per_sector_per_channel=4,
        ttl_steps=10,
    )
    embedding_dim = 5
    mfm = MFM(cfg, embedding_dim=embedding_dim)

    vec = np.arange(embedding_dim, dtype=np.float32)
    t_step = 0

    level = 1
    sector = 2
    channel = "E"

    lvl, sec, c_idx, slot_idx = mfm.write(
        level=level,
        sector=sector,
        channel=channel,
        value=vec,
        t_step=t_step,
    )

    assert lvl == level
    assert sec == sector
    assert c_idx == cfg.channels.index(channel)
    assert 0 <= slot_idx < cfg.capacity_per_sector_per_channel

    recovered = mfm.read_last(
        level=level,
        sector=sector,
        channel=channel,
        t_step=t_step,
    )

    assert recovered is not None
    assert recovered.shape == (embedding_dim,)
    assert np.allclose(recovered, vec)


def test_mfm_ttl_expiration():
    """
    Verifica que los slots expiren correctamente según ttl_steps.
    """
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=3,  # vida de 3 pasos
    )
    embedding_dim = 3
    mfm = MFM(cfg, embedding_dim=embedding_dim)

    v1 = np.ones(embedding_dim, dtype=np.float32)
    mfm.write(level=0, sector=0, channel="E", value=v1, t_step=0)

    # Antes de que venza el TTL, debe leerse
    assert mfm.read_last(0, 0, "E", t_step=2) is not None

    # Después de compactar y pasar el TTL, no debe leerse
    mfm.compact(t_step=4)
    assert mfm.read_last(0, 0, "E", t_step=4) is None


def test_mfm_ring_buffer_overwrite():
    """
    Verifica que el ring buffer sobrescriba valores antiguos cuando
    se excede la capacidad.
    """
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=10,
    )
    embedding_dim = 2
    mfm = MFM(cfg, embedding_dim=embedding_dim)

    v0 = np.array([0.0, 0.0], dtype=np.float32)
    v1 = np.array([1.0, 1.0], dtype=np.float32)
    v2 = np.array([2.0, 2.0], dtype=np.float32)

    mfm.write(0, 0, "E", v0, t_step=0)
    mfm.write(0, 0, "E", v1, t_step=1)

    # Ahora escribir el tercero debería sobrescribir el más antiguo (v0)
    mfm.write(0, 0, "E", v2, t_step=2)

    # El último valor leído debe ser v2
    last = mfm.read_last(0, 0, "E", t_step=2)
    assert last is not None
    assert np.allclose(last, v2)

    # La media debe estar entre v1 y v2 (v0 ya no cuenta)
    mean = mfm.read_mean(0, 0, "E", t_step=2, max_items=2)
    assert mean is not None
    assert np.all(mean >= v1)
    assert np.all(mean <= v2)


def test_mfm_multiresolution_read_stops_with_epsilon():
    """
    Simula una lectura multiresolución donde los niveles sucesivos
    convergen rápidamente, para probar el criterio epsilon.
    """
    cfg = MFMConfig(
        n_levels=3,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=4,
        ttl_steps=10,
    )
    embedding_dim = 3
    mfm = MFM(cfg, embedding_dim=embedding_dim)

    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t_step = 0

    # Escribimos valores cada vez más cercanos a "base" en niveles 0,1,2
    mfm.write(0, 0, "X", base, t_step=t_step)
    mfm.write(1, 0, "X", base + 0.01, t_step=t_step + 1)
    mfm.write(2, 0, "X", base + 0.001, t_step=t_step + 2)

    # Epsilon suficientemente pequeño para obligar a usar más de un nivel,
    # pero no todos necesariamente.
    epsilon = 0.05
    agg = mfm.read_multiresolution(
        sector=0,
        channel="X",
        t_step=t_step + 2,
        epsilon=epsilon,
        max_levels=None,
    )

    assert agg is not None
    assert agg.shape == (embedding_dim,)


def test_mfm_usage_stats():
    """
    Verifica que usage_stats devuelva valores coherentes.
    """
    cfg = MFMConfig(
        n_levels=2,
        n_sectors_per_level=2,
        capacity_per_sector_per_channel=2,
        ttl_steps=5,
    )
    embedding_dim = 4
    mfm = MFM(cfg, embedding_dim=embedding_dim)

    stats0 = mfm.usage_stats(t_step=0)
    assert isinstance(stats0, MFMUsageStats)
    assert stats0.total_slots == 2 * 2 * len(cfg.channels) * 2
    assert stats0.occupied_slots == 0
    assert stats0.valid_slots == 0

    # Escribimos en algunos slots
    v = np.ones(embedding_dim, dtype=np.float32)
    mfm.write(0, 0, "E", v, t_step=0)
    mfm.write(1, 1, "C", v, t_step=1)

    stats1 = mfm.usage_stats(t_step=1)
    assert stats1.occupied_slots >= 2
    assert stats1.valid_slots >= 2
    assert stats1.occupancy_ratio > 0.0
    assert stats1.valid_ratio > 0.0

def test_mfm_write_rejects_negative_t_step():
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=4,
    )
    mfm = MFM(cfg, embedding_dim=3)
    v = np.zeros(3, dtype=np.float32)

    with pytest.raises(ValueError):
        mfm.write(level=0, sector=0, channel="E", value=v, t_step=-1)


def test_mfm_write_rejects_wrong_embedding_dim():
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=4,
    )
    mfm = MFM(cfg, embedding_dim=3)

    v_wrong = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        mfm.write(level=0, sector=0, channel="E", value=v_wrong, t_step=0)


def test_mfm_read_last_returns_none_when_no_valid_slots():
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=2,
    )
    mfm = MFM(cfg, embedding_dim=2)

    # No se escribe nada: debe devolver None
    assert mfm.read_last(0, 0, "E", t_step=0) is None

    # Escribimos algo, luego dejamos que caduque
    v = np.ones(2, dtype=np.float32)
    mfm.write(0, 0, "E", v, t_step=0)
    mfm.compact(t_step=10)  # mucho más que ttl_steps

    assert mfm.read_last(0, 0, "E", t_step=10) is None


def test_mfm_read_mean_returns_none_when_no_valid_slots():
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=1,
    )
    mfm = MFM(cfg, embedding_dim=2)

    # Nada escrito: None
    assert mfm.read_mean(0, 0, "E", t_step=0) is None

    # Escribimos pero luego todo caduca
    v = np.zeros(2, dtype=np.float32)
    mfm.write(0, 0, "E", v, t_step=0)
    mfm.compact(t_step=5)
    assert mfm.read_mean(0, 0, "E", t_step=5) is None


def test_mfm_read_multiresolution_returns_none_without_data():
    cfg = MFMConfig(
        n_levels=3,
        n_sectors_per_level=2,
        capacity_per_sector_per_channel=2,
        ttl_steps=5,
    )
    mfm = MFM(cfg, embedding_dim=3)

    # No hay ningún write: multiresolution tiene que devolver None
    agg = mfm.read_multiresolution(
        sector=0,
        channel="E",
        t_step=0,
        epsilon=1e-3,
        max_levels=None,
    )
    assert agg is None


def test_mfm_read_multiresolution_respects_max_levels():
    cfg = MFMConfig(
        n_levels=3,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=4,
        ttl_steps=10,
    )
    mfm = MFM(cfg, embedding_dim=2)

    # Escribimos en nivel 0 y 1, pero dejamos vacío el 2
    v0 = np.array([1.0, 0.0], dtype=np.float32)
    v1 = np.array([0.0, 1.0], dtype=np.float32)

    mfm.write(0, 0, "E", v0, t_step=0)
    mfm.write(1, 0, "E", v1, t_step=1)

    # Si max_levels=1, solo puede ver nivel 0
    agg1 = mfm.read_multiresolution(
        sector=0,
        channel="E",
        t_step=1,
        epsilon=1.0,
        max_levels=1,
    )
    assert agg1 is not None
    assert np.allclose(agg1, v0)

    # Si max_levels=2, puede mezclar nivel 0 y 1
    agg2 = mfm.read_multiresolution(
        sector=0,
        channel="E",
        t_step=1,
        epsilon=0.0,
        max_levels=2,
    )
    assert agg2 is not None
    # Debe estar entre v0 y v1
    assert not np.allclose(agg2, v0)
    assert not np.allclose(agg2, v1)


def test_mfm_compact_rejects_negative_t_step():
    cfg = MFMConfig(
        n_levels=1,
        n_sectors_per_level=1,
        capacity_per_sector_per_channel=2,
        ttl_steps=5,
    )
    mfm = MFM(cfg, embedding_dim=2)

    with pytest.raises(ValueError):
        mfm.compact(t_step=-1)

