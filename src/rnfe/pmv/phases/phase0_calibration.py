# src/rnfe/pmv/phases/phase0_calibration.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

from rnfe.fmse.world.mini_world_boxes import (
    BoxWorldConfig,
    BoxWorld,
    BoxWorldState,
)
from rnfe.core.mfm.mfm_state import MFMConfig, MFM
from rnfe.infra.telemetry_bus import TelemetryBus


@dataclass
class Phase0Config:
    """
    Configuración de la Fase 0 (F0) de calibración.

    - world_config:
        Configuración del mini-mundo de cajitas fractales.
    - mfm_config:
        Configuración de la MFM mínima usada en F0.
    - embedding_dim:
        Dimensión del embedding al que se proyecta la observación plana.
    - n_steps:
        Número de pasos de simulación de F0.
    - seed:
        Semilla maestra para la parte interna de F0 (proyección, etc.).
    - log_prefix:
        Prefijo para nombres de métricas en el TelemetryBus.
    """

    world_config: BoxWorldConfig = field(default_factory=BoxWorldConfig)
    mfm_config: MFMConfig = field(
        default_factory=lambda: MFMConfig(
            n_levels=1,
            n_sectors_per_level=1,
            capacity_per_sector_per_channel=256,
            ttl_steps=256,
        )
    )
    embedding_dim: int = 64
    n_steps: int = 128
    seed: Optional[int] = None
    log_prefix: str = "F0"

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim debe ser > 0")
        if self.n_steps <= 0:
            raise ValueError("n_steps debe ser > 0")


@dataclass
class Phase0Result:
    """
    Resultado de una corrida completa de F0.

    - config:
        Configuración utilizada.
    - final_state_step:
        Paso temporal final alcanzado por el mini-mundo.
    - world_state:
        Estado del mini-mundo en el último paso.
    - mfm:
        Instancia de MFM después de todos los pasos.
    - telemetry:
        Bus de telemetría con todas las métricas registradas.
    """

    config: Phase0Config
    final_state_step: int
    world_state: BoxWorldState
    mfm: MFM
    telemetry: TelemetryBus


class Phase0CalibrationRunner:
    """
    Orquestador de la Fase 0 (F0).

    Conecta:

    - Mini-mundo fractal (BoxWorld).
    - MFM mínima (MFM).
    - TelemetryBus.

    Flujo:

    1. Inicializa mundo y memoria.
    2. Construye una proyección lineal aleatoria de la observación plana
       hacia un embedding de dimensión fija.
    3. Durante n_steps:
        - Avanza el mundo.
        - Observa (vector plano).
        - Proyecta a embedding.
        - Escribe en MFM.
        - Compacta slots expirados según TTL.
        - Registra métricas básicas en TelemetryBus.
    """

    def __init__(
        self,
        config: Phase0Config,
        telemetry: Optional[TelemetryBus] = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry or TelemetryBus()

        # RNG interno para F0 (no el del mini-mundo).
        self._rng = np.random.default_rng(config.seed)
        # Matriz de proyección [obs_dim, embedding_dim]
        self._proj_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Proyección a embedding
    # ------------------------------------------------------------------

    def _build_projection(self, obs_dim: int) -> None:
        """
        Construye una matriz de proyección [obs_dim, embedding_dim]
        con columnas normalizadas (norma L2 aproximadamente uno).

        Determinista para un Phase0Config dado (usa config.seed).
        """
        if self._proj_matrix is not None:
            return

        if obs_dim <= 0:
            raise ValueError("obs_dim debe ser > 0")

        d = self.config.embedding_dim
        raw = self._rng.normal(loc=0.0, scale=1.0, size=(obs_dim, d))
        col_norms = np.linalg.norm(raw, axis=0, keepdims=True)
        col_norms[col_norms == 0.0] = 1.0
        self._proj_matrix = (raw / col_norms).astype(np.float32)

    def _embed(self, obs: np.ndarray) -> np.ndarray:
        """
        Proyecta una observación plana a un embedding de dimensión fija.
        """
        if obs.ndim != 1:
            raise ValueError("obs debe ser vector 1D")
        if obs.size == 0:
            raise ValueError("obs no puede estar vacío")

        if self._proj_matrix is None:
            self._build_projection(obs_dim=obs.shape[0])

        assert self._proj_matrix is not None
        emb = obs.astype(np.float32) @ self._proj_matrix
        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    # Ejecución de F0
    # ------------------------------------------------------------------

    def run(self) -> Phase0Result:
        """
        Ejecuta la Fase 0 completa y devuelve el resultado.
        """

        # 1. Inicializar mini-mundo y MFM
        world = BoxWorld(self.config.world_config)
        mfm = MFM(self.config.mfm_config, embedding_dim=self.config.embedding_dim)

        # Semilla para el mundo:
        # prioriza world_config.seed, si no hay usa Phase0Config.seed.
        world_seed = self.config.world_config.seed
        if world_seed is None:
            world_seed = self.config.seed

        state = world.reset(seed=world_seed)

        # 2. Registrar información estructural del mundo
        shapes: List[Tuple[int, int]] = world.level_shapes()
        prefix = self.config.log_prefix

        self.telemetry.log_scalar(
            f"{prefix}.n_levels",
            step=0,
            value=len(shapes),
        )

        level_sizes = np.array(
            [h * w for (h, w) in shapes],
            dtype=np.float32,
        )
        self.telemetry.log_vector(
            f"{prefix}.level_sizes",
            step=0,
            value=level_sizes,
        )

        # 3. Primera observación y construcción de proyección
        obs = world.observe_flat()
        self._build_projection(obs_dim=obs.shape[0])

        # 4. Bucle principal de pasos
        for step in range(self.config.n_steps):
            if step > 0:
                state = world.step()
                obs = world.observe_flat()

            emb = self._embed(obs)

            # Escribimos embedding en MFM: nivel 0, sector 0, canal "E"
            mfm.write(
                level=0,
                sector=0,
                channel="E",
                value=emb,
                t_step=step,
            )

            # Compactar según TTL
            mfm.compact(t_step=step)

            # Métricas básicas sobre observación y embedding
            obs_norm = float(np.linalg.norm(obs))
            obs_std = float(np.std(obs))
            emb_norm = float(np.linalg.norm(emb))

            self.telemetry.log_scalar(
                f"{prefix}.obs_norm_l2",
                step=step,
                value=obs_norm,
            )
            self.telemetry.log_scalar(
                f"{prefix}.obs_std",
                step=step,
                value=obs_std,
            )
            self.telemetry.log_scalar(
                f"{prefix}.emb_norm_l2",
                step=step,
                value=emb_norm,
            )

            # Métricas de uso de MFM
            stats = mfm.usage_stats(t_step=step)
            self.telemetry.log_scalar(
                f"{prefix}.mfm_occupancy_ratio",
                step=step,
                value=stats.occupancy_ratio,
            )
            self.telemetry.log_scalar(
                f"{prefix}.mfm_valid_ratio",
                step=step,
                value=stats.valid_ratio,
            )

        # 5. Empaquetar resultado
        return Phase0Result(
            config=self.config,
            final_state_step=state.t,
            world_state=state,
            mfm=mfm,
            telemetry=self.telemetry,
        )
