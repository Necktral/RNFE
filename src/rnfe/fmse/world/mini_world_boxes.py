# src/rnfe/fmse/world/mini_world_boxes.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class BoxWorldConfig:
    """
    Configuración del mini-mundo de cajitas fractales para F0.

    Este mini-mundo está pensado como entorno base para F0/F1:
    - Varias "capas" o niveles fractales (coarse → fine).
    - Cada nivel es una grilla rectangular de cajitas.
    - La dinámica es ruido fractal simple + correlación temporal (AR(1)).

    La semilla se puede fijar para reproducibilidad.
    """

    base_height: int = 32
    base_width: int = 32
    n_levels: int = 3

    # Número de escalas utilizadas para componer el ruido fractal
    n_scales_noise: int = 4

    # Factor de persistencia entre escalas (0 < persistence < 1)
    # Valores más altos → más peso a escalas groseras.
    persistence: float = 0.5

    # Coeficiente de correlación temporal en [0, 1].
    # alpha cerca de 1 → cambios suaves en el tiempo.
    temporal_alpha: float = 0.9

    # Semilla para RNG; si None, se usa una aleatoria.
    seed: Optional[int] = None

    def level_shape(self, level: int) -> Tuple[int, int]:
        """
        Devuelve (height, width) para un nivel dado.

        Usamos un escalado simple: cada nivel superior reduce las dimensiones
        a la mitad (redondeando hacia abajo, con mínimo 1).
        """
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Nivel fuera de rango: {level}")

        h = max(1, self.base_height // (2 ** level))
        w = max(1, self.base_width // (2 ** level))
        return h, w


@dataclass
class BoxWorldState:
    """
    Estado del mini-mundo en un instante de tiempo.

    - t: paso temporal.
    - levels: lista de arrays 2D (float32) con los valores de las cajitas
      en cada nivel. level 0 es el más fino, level n_levels-1 el más grosero.
    """

    t: int
    levels: List[np.ndarray]

    def copy(self) -> "BoxWorldState":
        """
        Copia profunda del estado (para comparación o logging).
        """
        return BoxWorldState(
            t=self.t,
            levels=[np.array(l, copy=True) for l in self.levels],
        )


class BoxWorld:
    """
    Mini-mundo de cajitas fractales para Fase 0 de RNFE.

    Características:
    - Cada nivel es una grilla de valores reales en [-1, 1] aproximadamente.
    - La inicialización genera campos fractales espaciales mediante composición
      multi-escala de ruido gaussiano.
    - La dinámica temporal es un proceso AR(1) por nivel:
        field_{t+1} = alpha * field_t + (1 - alpha) * noise_fractal_nuevo
    - El sistema NO modela aún leyes ocultas complejas ni razonamiento alto.
      Es un entorno "biosfera" donde el núcleo RNFE puede medir:
        - estabilidad numérica,
        - propiedades fractales básicas,
        - comportamiento de telemetría.
    """

    def __init__(self, config: BoxWorldConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._state: Optional[BoxWorldState] = None

        # Precomputamos el factor de normalización del ruido multi-escala
        weights = np.array(
            [config.persistence ** s for s in range(config.n_scales_noise)],
            dtype=np.float64,
        )
        # Evita división por cero en caso degenerado
        norm = float(np.sqrt(np.sum(weights ** 2))) if np.any(weights) else 1.0
        self._scale_weights = weights / norm

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> BoxWorldState:
        """
        Reinicia el mini-mundo y devuelve el estado inicial.

        Si se pasa `seed`, se re-inicializa el RNG, permitiendo
        reproducibilidad exacta de episodios.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        levels: List[np.ndarray] = []
        for level in range(self.config.n_levels):
            field = self._generate_fractal_field_for_level(level)
            levels.append(field.astype(np.float32))

        self._state = BoxWorldState(t=0, levels=levels)
        return self._state.copy()

    def step(self) -> BoxWorldState:
        """
        Avanza un paso temporal.

        En F0 no modelamos acciones del agente; la dinámica es puramente
        interna (ruido fractal + correlación temporal).
        """
        if self._state is None:
            raise RuntimeError("Debe llamarse reset() antes de step().")

        alpha = float(self.config.temporal_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"temporal_alpha debe estar en [0, 1], recibido {alpha}"
            )

        new_levels: List[np.ndarray] = []
        for level_idx, prev_field in enumerate(self._state.levels):
            noise = self._generate_fractal_field_for_level(level_idx)
            # AR(1) por nivel
            new_field = alpha * prev_field + (1.0 - alpha) * noise
            # Compactamos a float32 y controlamos saturación suave con tanh
            new_field = np.tanh(new_field).astype(np.float32)
            new_levels.append(new_field)

        self._state = BoxWorldState(t=self._state.t + 1, levels=new_levels)
        return self._state.copy()

    def current_state(self) -> BoxWorldState:
        """
        Devuelve una copia del estado actual.

        Lanza un error si aún no se ha llamado a reset().
        """
        if self._state is None:
            raise RuntimeError("Mini-mundo no inicializado. Llama a reset().")
        return self._state.copy()

    def observe_flat(self) -> np.ndarray:
        """
        Devuelve una observación en forma de vector 1D.

        Concatenamos los valores de todas las cajitas de todos los niveles,
        en el siguiente orden:
            [nivel 0 (fine), nivel 1, ..., nivel n_levels-1 (coarse)].

        Esta API es la que luego consumirá la MFM mínima en F0.
        """
        if self._state is None:
            raise RuntimeError("Mini-mundo no inicializado. Llama a reset().")

        flat_list: List[np.ndarray] = []
        for level in self._state.levels:
            flat_list.append(level.ravel())

        if not flat_list:
            return np.empty((0,), dtype=np.float32)

        obs = np.concatenate(flat_list).astype(np.float32)
        return obs

    def level_shapes(self) -> List[Tuple[int, int]]:
        """
        Devuelve las formas (height, width) de cada nivel.
        """
        return [self.config.level_shape(lvl) for lvl in range(self.config.n_levels)]

    # ------------------------------------------------------------------
    # Implementación interna
    # ------------------------------------------------------------------

    def _generate_fractal_field_for_level(self, level: int) -> np.ndarray:
        """
        Genera un campo fractal 2D para un nivel determinado.

        Estrategia:
        - Para cada escala s = 0..n_scales_noise-1:
            - Se genera ruido gaussiano en una grilla más pequeña.
            - Se hace upsample por repetición a la forma del nivel.
            - Se combina con pesos de persistencia.
        - Se normaliza para tener varianza aproximadamente constante.
        - No se aplica tanh aquí; eso se aplica en la dinámica temporal.

        Esta construcción es ligera (solo NumPy) y suficiente para F0,
        sin recurrir a librerías externas adicionales.
        """
        h, w = self.config.level_shape(level)
        base_shape = (h, w)

        acc = np.zeros(base_shape, dtype=np.float64)

        for s, weight in enumerate(self._scale_weights):
            # Resolución de esta escala (más grosera a medida que s crece)
            scale_div = 2**s
            small_h = max(1, h // scale_div)
            small_w = max(1, w // scale_div)

            noise_small = self._rng.normal(loc=0.0, scale=1.0, size=(small_h, small_w))

            # Upsample por repetición (nearest neighbor) para volver a (h, w)
            # Repetimos filas y columnas para rellenar la grilla grande.
            rep_h = int(np.ceil(h / small_h))
            rep_w = int(np.ceil(w / small_w))
            noise_large = np.repeat(
                np.repeat(noise_small, rep_h, axis=0), rep_w, axis=1
            )
            noise_large = noise_large[:h, :w]  # recorte exacto

            acc += float(weight) * noise_large

        # Normalización suave para evitar valores demasiado grandes
        # y hacer que la escala entre niveles sea coherente.
        std = float(np.std(acc)) if np.std(acc) > 0 else 1.0
        acc /= std

        return acc.astype(np.float32)
