# src/rnfe/infra/telemetry_bus.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class TelemetryConfig:
    """
    Configuración del bus de telemetría.

    - allow_new_metrics:
        Si es False, solo se podrán registrar métricas que hayan sido
        registradas previamente. Esto sirve para congelar el "esquema" de
        telemetría en fases más avanzadas.
    """

    allow_new_metrics: bool = True


@dataclass
class TelemetrySeries:
    """
    Serie de telemetría para una métrica concreta.

    - name:
        Nombre de la métrica, por ejemplo "rho_J", "vram_usage", "Io".
    - is_vector:
        True si la métrica es un vector (dim > 1), False si es escalar.
    - dim:
        Dimensión del vector en el caso is_vector=True. Para escalares es 1.
    - steps:
        Lista de pasos discretos (enteros no negativos).
    - values:
        Lista de arrays float32. Para escalares se usa array de forma (1,).
    """

    name: str
    is_vector: bool
    dim: int
    steps: List[int]
    values: List[np.ndarray]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (steps, values) como arrays NumPy.

        - steps: forma (T,)
        - values:
            - (T,) para métricas escalares,
            - (T, dim) para métricas vectoriales.
        """
        if not self.steps:
            return np.zeros((0,), dtype=np.int64), np.zeros(
                (0, self.dim), dtype=np.float32
            )

        steps_arr = np.asarray(self.steps, dtype=np.int64)
        vals_arr = np.stack(self.values, axis=0).astype(np.float32)

        if not self.is_vector:
            # Para escalares, devolvemos un vector 1D
            vals_arr = vals_arr.reshape(-1)
        return steps_arr, vals_arr


class TelemetryBus:
    """
    Bus de telemetría simple pero robusto para RNFE-PMV.

    Objetivo:
    - Registrar series temporales de métricas diversas (escalars o vectores).
    - Mantener coherencia de dimensiones.
    - Permitir extracción como arrays NumPy para análisis posterior.

    Este módulo está pensado para:

    - F0: registrar usos de VRAM, métricas de borde, estado de MFM.
    - F1/F2: registrar métricas de razonamiento, Io, fitness de linajes.

    No escribe directamente a disco; esa responsabilidad será de un módulo
    de almacenamiento superior.
    """

    def __init__(self, config: Optional[TelemetryConfig] = None) -> None:
        self.config = config or TelemetryConfig()
        self._series: Dict[str, TelemetrySeries] = {}

    # ------------------------------------------------------------------
    # Registro de métricas
    # ------------------------------------------------------------------

    def log_scalar(self, name: str, step: int, value: float) -> None:
        """
        Registra una métrica escalar en un paso dado.

        - name: nombre de la métrica.
        - step: paso discreto (entero no negativo).
        - value: valor escalar convertible a float32.
        """
        if step < 0:
            raise ValueError("step debe ser >= 0")

        v = np.array([float(value)], dtype=np.float32)
        self._log(name=name, step=step, value=v, is_vector=False)

    def log_vector(self, name: str, step: int, value: np.ndarray) -> None:
        """
        Registra una métrica vectorial en un paso dado.

        - name: nombre de la métrica.
        - step: paso discreto (entero no negativo).
        - value: array 1D de floats (dimensión del vector).
        """
        if step < 0:
            raise ValueError("step debe ser >= 0")

        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("value para log_vector debe ser 1D")
        if arr.size == 0:
            raise ValueError("value para log_vector no puede estar vacío")

        self._log(name=name, step=step, value=arr, is_vector=True)

    def _log(
        self, name: str, step: int, value: np.ndarray, is_vector: bool
    ) -> None:
        """
        Implementación interna compartida para log_scalar y log_vector.
        """
        if name in self._series:
            series = self._series[name]
            # Verificar consistencia de tipo escalar/vectorial
            if series.is_vector != is_vector:
                kind_new = "vectorial" if is_vector else "escalar"
                kind_old = "vectorial" if series.is_vector else "escalar"
                raise TypeError(
                    f"La métrica '{name}' ya estaba registrada como {kind_old} "
                    f"y se intenta registrar como {kind_new}."
                )

            # Verificar consistencia de dimensión
            if value.size != series.dim:
                raise ValueError(
                    f"La métrica '{name}' tiene dimensión fija {series.dim}, "
                    f"se intentó registrar con dimensión {value.size}."
                )

            series.steps.append(int(step))
            series.values.append(value.astype(np.float32))

        else:
            # Métrica nueva
            if not self.config.allow_new_metrics:
                raise KeyError(
                    f"No se permite registrar nuevas métricas: '{name}'. "
                    "Configura allow_new_metrics=True o declara la métrica de antemano."
                )

            dim = int(value.size)
            series = TelemetrySeries(
                name=name,
                is_vector=is_vector,
                dim=dim,
                steps=[int(step)],
                values=[value.astype(np.float32)],
            )
            self._series[name] = series

    # ------------------------------------------------------------------
    # Consulta de métricas
    # ------------------------------------------------------------------

    def get_series(self, name: str) -> TelemetrySeries:
        """
        Devuelve la serie de telemetría para una métrica dada.

        Lanza KeyError si la métrica no existe.
        """
        if name not in self._series:
            raise KeyError(f"Métrica desconocida: '{name}'")
        return self._series[name]

    def has_metric(self, name: str) -> bool:
        """
        Indica si la métrica dada ha sido registrada alguna vez.
        """
        return name in self._series

    def list_metrics(self) -> List[str]:
        """
        Lista los nombres de todas las métricas registradas.
        """
        return sorted(self._series.keys())

    def as_dict_numpy(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Devuelve un diccionario:
            nombre_métrica -> (steps, values)

        - steps: array int64 de forma (T,)
        - values: array float32 de forma:
            - (T,) para escalares,
            - (T, dim) para vectores.
        """
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, series in self._series.items():
            out[name] = series.to_numpy()
        return out

    # ------------------------------------------------------------------
    # Gestión de configuración
    # ------------------------------------------------------------------

    def freeze_schema(self) -> None:
        """
        A partir de este punto no se podrán añadir nuevas métricas.

        Se sigue permitiendo registrar valores de métricas ya existentes.
        """
        self.config.allow_new_metrics = False

    def unfreeze_schema(self) -> None:
        """
        Permite nuevamente registrar métricas nuevas.

        Se mantiene la información previa sin cambios.
        """
        self.config.allow_new_metrics = True
