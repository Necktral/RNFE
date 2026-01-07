# src/rnfe/core/mfm/mfm_state.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np


CHANNELS_DEFAULT: Tuple[str, ...] = ("E", "X", "C", "M")


@dataclass
class MFMConfig:
    """
    Configuración de la Memoria Fractal Multiescala (MFM) mínima para F0.

    Esta versión está alineada con la especificación conceptual de MFM, pero
    implementada como un almacenamiento denso en forma de tensor:

        values[l, s, c, k, d]

    donde:
    - l: nivel fractal   (0..n_levels-1, 0 = más grueso)
    - s: sector          (0..n_sectors_per_level-1)
    - c: canal           (índice dentro de `channels`)
    - k: slot            (0..capacity_per_sector_per_channel-1)
    - d: dimensión de embedding

    Propiedades:
    - TTL en pasos discretos: slots expirados se marcan como vacíos.
    - Escritura en modo "ring buffer" (sobrescribe lo más antiguo en el sector/canal).
    """

    n_levels: int
    n_sectors_per_level: int
    capacity_per_sector_per_channel: int
    ttl_steps: int = 128
    channels: Tuple[str, ...] = CHANNELS_DEFAULT

    def __post_init__(self) -> None:
        if self.n_levels <= 0:
            raise ValueError("n_levels debe ser > 0")
        if self.n_sectors_per_level <= 0:
            raise ValueError("n_sectors_per_level debe ser > 0")
        if self.capacity_per_sector_per_channel <= 0:
            raise ValueError("capacity_per_sector_per_channel debe ser > 0")
        if self.ttl_steps <= 0:
            raise ValueError("ttl_steps debe ser > 0")
        if len(self.channels) == 0:
            raise ValueError("Debe haber al menos un canal en channels")


@dataclass
class MFMUsageStats:
    """
    Estadísticas simples de uso de la MFM en un instante dado.
    """

    total_slots: int
    occupied_slots: int
    valid_slots: int  # slots dentro de TTL
    occupancy_ratio: float
    valid_ratio: float


class MFM:
    """
    Implementación mínima de la Memoria Fractal Multiescala (MFM) para F0.

    Esta MFM no implementa todavía el ruteo completo sobre la VFD, ni el
    QP de proyección a S_safe. Se centra en:

    - Almacenamiento multiescala y multicanal de embeddings.
    - TTL por slot (expiración temporal discreta).
    - Lectura:
        - por nivel/sector/canal (último o media),
        - multiresolución coarse → fine con criterio de parada simple.

    Esta es la capa que F0 puede usar para:
    - guardar observaciones comprimidas del mini-mundo,
    - leer "resúmenes" a distintas escalas,
    - medir estabilidad básica de la dinámica interna de memoria.
    """

    def __init__(self, config: MFMConfig, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim debe ser > 0")

        self.config = config
        self.embedding_dim = int(embedding_dim)

        # Mapear nombre de canal a índice entero.
        self._channel_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.config.channels)
        }

        L = self.config.n_levels
        S = self.config.n_sectors_per_level
        C = len(self.config.channels)
        K = self.config.capacity_per_sector_per_channel
        D = self.embedding_dim

        # Tensor principal de valores: [L, S, C, K, D]
        self._values = np.zeros((L, S, C, K, D), dtype=np.float32)

        # Tiempos de escritura (en pasos discretos). -1 indica slot vacío.
        self._ages = np.full((L, S, C, K), fill_value=-1, dtype=np.int64)

        # Puntero de escritura tipo "ring buffer": [L, S, C]
        self._write_ptrs = np.zeros((L, S, C), dtype=np.int64)

    # ------------------------------------------------------------------
    # Métodos internos auxiliares
    # ------------------------------------------------------------------

    def _validate_indices(self, level: int, sector: int, channel: str) -> Tuple[int, int, int]:
        if not (0 <= level < self.config.n_levels):
            raise IndexError(f"nivel fuera de rango: {level}")
        if not (0 <= sector < self.config.n_sectors_per_level):
            raise IndexError(f"sector fuera de rango: {sector}")
        if channel not in self._channel_to_idx:
            raise KeyError(f"canal desconocido: {channel}")
        c_idx = self._channel_to_idx[channel]
        return level, sector, c_idx

    # ------------------------------------------------------------------
    # API pública: escritura
    # ------------------------------------------------------------------

    def write(
        self,
        level: int,
        sector: int,
        channel: str,
        value: np.ndarray,
        t_step: int,
    ) -> Tuple[int, int, int, int]:
        """
        Escribe un embedding en la MFM.

        Argumentos:
        - level: nivel fractal (0..L-1).
        - sector: índice de sector dentro del nivel.
        - channel: nombre de canal, por ejemplo "E", "X", "C", "M".
        - value: vector 1D de tamaño embedding_dim.
        - t_step: paso discreto global en el que se escribe.

        Devuelve:
        - una tupla (level, sector, channel_index, slot_index) que identifica
          la celda donde quedó almacenado el valor.
        """
        if t_step < 0:
            raise ValueError("t_step debe ser >= 0")

        lvl, sec, c_idx = self._validate_indices(level, sector, channel)

        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("value debe ser un vector 1D")
        if value.shape[0] != self.embedding_dim:
            raise ValueError(
                f"dimensión de value ({value.shape[0]}) "
                f"no coincide con embedding_dim ({self.embedding_dim})"
            )

        k_ptr = int(self._write_ptrs[lvl, sec, c_idx])
        k_ptr = k_ptr % self.config.capacity_per_sector_per_channel

        self._values[lvl, sec, c_idx, k_ptr, :] = value
        self._ages[lvl, sec, c_idx, k_ptr] = int(t_step)

        # Avanzar puntero de escritura
        self._write_ptrs[lvl, sec, c_idx] = (k_ptr + 1) % self.config.capacity_per_sector_per_channel

        return lvl, sec, c_idx, k_ptr

    # ------------------------------------------------------------------
    # API pública: lectura local
    # ------------------------------------------------------------------

    def read_last(
        self,
        level: int,
        sector: int,
        channel: str,
        t_step: int,
        ignore_ttl: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Lee el último valor válido para (level, sector, channel).

        - Si `ignore_ttl` es False, solo devuelve slots dentro de TTL.
        - Si no encuentra ningún slot válido, devuelve None.
        """
        lvl, sec, c_idx = self._validate_indices(level, sector, channel)

        ages = self._ages[lvl, sec, c_idx, :]
        mask_valid = ages >= 0

        if not ignore_ttl:
            ttl = self.config.ttl_steps
            mask_valid &= (t_step - ages) <= ttl

        idx_valid = np.where(mask_valid)[0]
        if idx_valid.size == 0:
            return None

        # Elegimos el slot con mayor edad (último escrito efectivo)
        k_last = int(idx_valid[np.argmax(ages[idx_valid])])
        return self._values[lvl, sec, c_idx, k_last, :].copy()

    def read_mean(
        self,
        level: int,
        sector: int,
        channel: str,
        t_step: int,
        max_items: Optional[int] = None,
        ignore_ttl: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Devuelve la media de los valores válidos para (level, sector, channel).

        - max_items: si se especifica, solo usa los `max_items` slots más recientes.
        - Si no hay slots válidos, devuelve None.
        """
        lvl, sec, c_idx = self._validate_indices(level, sector, channel)

        ages = self._ages[lvl, sec, c_idx, :]
        mask_valid = ages >= 0

        if not ignore_ttl:
            ttl = self.config.ttl_steps
            mask_valid &= (t_step - ages) <= ttl

        idx_valid = np.where(mask_valid)[0]
        if idx_valid.size == 0:
            return None

        # Ordenamos por edad descendente (más reciente primero)
        idx_sorted = idx_valid[np.argsort(-ages[idx_valid])]

        if max_items is not None and max_items > 0:
            idx_sorted = idx_sorted[:max_items]

        vals = self._values[lvl, sec, c_idx, idx_sorted, :]
        return np.mean(vals, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # API pública: lectura multiresolución
    # ------------------------------------------------------------------

    def read_multiresolution(
        self,
        sector: int,
        channel: str,
        t_step: int,
        epsilon: float = 1e-3,
        max_levels: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Lectura multiresolución coarse → fine para un sector/canal.

        Estrategia:
        - Recorre niveles desde 0 (más grueso) hasta L-1 (más fino).
        - En cada nivel:
            - toma el último valor válido (si existe),
            - actualiza una representación agregada por promedio incremental.
        - Se detiene cuando el cambio entre iteraciones cae por debajo de epsilon
          (norma L2 de la diferencia), o cuando se alcanzan todos los niveles.

        Devuelve:
        - vector 1D de tamaño embedding_dim si se encontró al menos un valor,
        - None si no había ningún valor válido en ningún nivel.
        """
        if sector < 0 or sector >= self.config.n_sectors_per_level:
            raise IndexError(f"sector fuera de rango: {sector}")
        if channel not in self._channel_to_idx:
            raise KeyError(f"canal desconocido: {channel}")

        L = self.config.n_levels
        L_eff = L if max_levels is None else min(L, max_levels)

        agg: Optional[np.ndarray] = None

        for level in range(L_eff):
            v = self.read_last(level, sector, channel, t_step=t_step, ignore_ttl=False)
            if v is None:
                continue

            if agg is None:
                agg = v
            else:
                new_agg = 0.5 * agg + 0.5 * v
                diff = float(np.linalg.norm(new_agg - agg))
                agg = new_agg
                if diff < float(epsilon):
                    break

        if agg is None:
            return None

        return agg.astype(np.float32)

    # ------------------------------------------------------------------
    # Compactación / limpieza
    # ------------------------------------------------------------------

    def compact(self, t_step: int) -> None:
        """
        Marca como vacíos todos los slots cuya edad haya excedido el TTL.

        No reordena datos; simplemente invalida los slots expirados.
        """
        if t_step < 0:
            raise ValueError("t_step debe ser >= 0")

        ttl = self.config.ttl_steps
        ages = self._ages

        mask_expired = (ages >= 0) & ((t_step - ages) > ttl)
        ages[mask_expired] = -1

    # ------------------------------------------------------------------
    # Estadísticas de uso
    # ------------------------------------------------------------------

    def usage_stats(self, t_step: int) -> MFMUsageStats:
        """
        Calcula estadísticas básicas de uso de la memoria en un instante dado.
        """
        ages = self._ages
        ttl = self.config.ttl_steps

        total_slots = int(ages.size)
        occupied_slots = int(np.sum(ages >= 0))
        valid_slots = int(np.sum((ages >= 0) & ((t_step - ages) <= ttl)))

        occupancy_ratio = occupied_slots / total_slots if total_slots > 0 else 0.0
        valid_ratio = valid_slots / total_slots if total_slots > 0 else 0.0

        return MFMUsageStats(
            total_slots=total_slots,
            occupied_slots=occupied_slots,
            valid_slots=valid_slots,
            occupancy_ratio=occupancy_ratio,
            valid_ratio=valid_ratio,
        )
