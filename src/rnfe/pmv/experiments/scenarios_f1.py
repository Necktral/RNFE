
from __future__ import annotations
# ---------------------------------------------------------------------------
# Especificación de escenario F1-IND
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Sequence, Tuple, Protocol, Callable

@dataclass(frozen=True)
class F1InductiveScenarioSpec:
    scenario_id: str
    window_size: int
    n_train: int
    n_test: int
    n_ood: int
    law_coeffs: Sequence[float]
    lambda_cost: float = 0.1
    gamma_k: float = 1.0
    gamma_mdl: float = 1.0

import numpy as np

from rnfe.pmv.phases.phase1_inductive import (
    Phase1InductiveConfig,
    Phase1InductiveDataset,
    Phase1InductiveExperimentResult,
    run_phase1_inductive_experiment,
)


# ---------------------------------------------------------------------------
# Protocolo de proveedor de mundo: F0 + MFM → secuencias (X_seq, E_seq)
# ---------------------------------------------------------------------------


class WorldSequenceProvider(Protocol):
    # Protocolo para un generador de secuencias de mundo + embeddings.
    #
    # Un implementador típico llamará al mini-mundo real (por ejemplo,
    # mini_world_boxes + MFM) para producir:
    #
    # - X_seq: estados del mundo de forma (T, H, W)
    # - E_seq: embeddings de forma (T, d)
    #
    # El parámetro `condition` permite distinguir entre:
    #
    # - "fractal_main":   condición fractal in-distribution (train + test)
    # - "fractal_ood":    condición fractal OOD (otra configuración)
    # - "nonfr_main":     condición no fractal in-distribution
    # BoxWorld fractal vs no fractal para F1 (toy 12x12 y 150x150)
    # ----------------------------------------------------------------------
    pass


@dataclass
class GenericWorldSequenceProvider:
    init_fn: Callable[[str, np.random.Generator], np.ndarray]
    step_fn: Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]
    grid_shape: tuple[int, int]
    embedding_dim: int

    def __call__(self, condition: str, n_steps: int, rng: np.random.Generator):
        # Produce sequences X_seq (T,H,W) and E_seq (T,d) by repeatedly
        # applying the provided init and step functions.
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        state = self.init_fn(condition, rng)
        state = np.asarray(state, dtype=float)
        if state.shape != self.grid_shape:
            raise ValueError(f"Initial state has shape {state.shape}, expected {self.grid_shape}")

        T = int(n_steps)
        H, W = self.grid_shape
        d = int(self.embedding_dim)
        X_seq = np.zeros((T, H, W), dtype=float)
        E_seq = np.zeros((T, d), dtype=float)

        for t in range(T):
            # embedding corresponds to the embedding produced when
            # stepping from the current `state` to the next one
            next_state, emb = self.step_fn(state, rng)
            X_seq[t] = state
            emb = np.asarray(emb, dtype=float)
            if emb.shape != (d,):
                # allow embeddings that are longer/shorter but coerce when possible
                emb = np.resize(emb, (d,))
            E_seq[t] = emb
            state = np.asarray(next_state, dtype=float)

        return X_seq, E_seq

def make_boxworld_provider(
    grid_shape: tuple[int, int],
    embedding_dim: int,
) -> GenericWorldSequenceProvider:
    # WorldSequenceProvider toy para F1 con dos mundos:
    # - fractal_main / fractal_ood: con latentes jerárquicos G (global),
    #   Q_k (macro-bloques 3x3) y R_m (micro-bloques dentro de cada macro).
    # - nonfr_main / nonfr_ood: AR(1) plano sobre la grilla, sin estructura latente.
    # Soporta explícitamente grillas (12, 12) y (150, 150).
    import numpy as np  # por si este módulo no lo tiene ya importado

    H, W = grid_shape
    d = embedding_dim

    if (H, W) not in [(12, 12), (150, 150)]:
        raise ValueError(
            f"make_boxworld_provider toy solo está definido para (12,12) y (150,150); "
            f"recibido grid_shape={grid_shape}."
        )

    # 1) Topología: macros 3x3 y micro-bloques dentro de cada macro
    macro_size = 3 if (H, W) == (12, 12) else 10
    n_macro_h = H // macro_size
    n_macro_w = W // macro_size
    n_macro = n_macro_h * n_macro_w
    n_micro_per_macro = macro_size * macro_size
    n_micro_total = n_macro * n_micro_per_macro
    macro_index = np.zeros((H, W), dtype=int)
    micro_index = np.zeros((H, W), dtype=int)
    micro_parent_macro = np.zeros(n_micro_total, dtype=int)
    idx = 0
    for i in range(n_macro_h):
        for j in range(n_macro_w):
            for mi in range(macro_size):
                for mj in range(macro_size):
                    hi = i * macro_size + mi
                    wj = j * macro_size + mj
                    macro = i * n_macro_w + j
                    macro_index[hi, wj] = macro
                    micro = idx
                    micro_index[hi, wj] = micro
                    micro_parent_macro[micro] = macro
                    idx += 1

    def _build_embedding(X):
        # Simplemente un flatten + media para toy
        return X.flatten()[:d].mean() * np.ones(d)

    # Parámetros de dinámica
    fractal_main = dict(
        a_g=0.95,
        sigma_g=0.05,
        a_q=0.90,
        b_q=0.20,
        sigma_q=0.06,
        a_r=0.90,
        b_r=0.30,
        sigma_r=0.08,
        sigma_cell=0.05,
    )
    fractal_ood = dict(
        a_g=0.90,
        sigma_g=0.07,
        a_q=0.88,
        b_q=0.22,
        sigma_q=0.08,
        a_r=0.86,
        b_r=0.32,
        sigma_r=0.10,
        sigma_cell=0.06,
    )
    nonfr_main = dict(alpha=0.94, sigma=0.10)
    nonfr_ood = dict(alpha=0.88, sigma=0.14)

    # Latentes / estado de mundo (se mantienen en el closure)
    G = None
    Q = None
    R = None
    world_kind = None
    params_current = None

    def init_fn(condition: str, rng: np.random.Generator) -> np.ndarray:
        nonlocal G, Q, R, world_kind, params_current
        if condition == "fractal_main":
            world_kind = "fractal"
            params_current = fractal_main
        elif condition == "fractal_ood":
            world_kind = "fractal"
            params_current = fractal_ood
        elif condition == "nonfr_main":
            world_kind = "nonfr"
            params_current = nonfr_main
        elif condition == "nonfr_ood":
            world_kind = "nonfr"
            params_current = nonfr_ood
        else:
            raise ValueError(f"Condición desconocida: {condition!r}")
        if world_kind == "fractal":
            G = float(rng.normal(scale=0.1))
            Q = rng.normal(scale=0.1, size=n_macro)
            R = rng.normal(scale=0.1, size=n_micro_total)
            base = G + Q[macro_index] + R[micro_index]
            X0 = base + rng.normal(
                scale=params_current["sigma_cell"],
                size=(H, W),
            )
        else:
            G = None
            Q = None
            R = None
            X0 = rng.normal(scale=params_current["sigma"], size=(H, W))
        X0 = np.asarray(X0, dtype=float)
        if X0.shape != (H, W):
            raise ValueError(
                f"El estado inicial tiene forma {X0.shape}, se esperaba {(H, W)}."
            )
        return X0

    def step_fn(state: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        nonlocal G, Q, R, world_kind, params_current
        if params_current is None or world_kind is None:
            raise RuntimeError(
                "init_fn debe llamarse antes de step_fn (params_current/world_kind vacíos)."
            )
        if world_kind == "fractal":
            pc = params_current
            G_new = pc["a_g"] * G + rng.normal(scale=pc["sigma_g"])
            Q_new = (
                pc["a_q"] * Q
                + pc["b_q"] * G_new
                + rng.normal(scale=pc["sigma_q"], size=n_macro)
            )
            Q_for_R = Q_new[micro_parent_macro]
            R_new = (
                pc["a_r"] * R
                + pc["b_r"] * Q_for_R
                + rng.normal(scale=pc["sigma_r"], size=n_micro_total)
            )
            base = G_new + Q_new[macro_index] + R_new[micro_index]
            X_next = base + rng.normal(
                scale=pc["sigma_cell"],
                size=(H, W),
            )
            G = float(G_new)
            Q = Q_new
            R = R_new
        else:
            pc = params_current
            X_next = pc["alpha"] * state + rng.normal(
                scale=pc["sigma"],
                size=(H, W),
            )
        X_next = np.asarray(X_next, dtype=float)
        if X_next.shape != (H, W):
            raise ValueError(
                f"X_next tiene forma {X_next.shape}, se esperaba {(H, W)}."
            )
        e_next = _build_embedding(X_next)
        if e_next.shape != (d,):
            raise ValueError(
                f"e_next tiene forma {e_next.shape}, se esperaba {(d,)}."
            )
        return X_next, e_next

    provider = GenericWorldSequenceProvider(
        init_fn=init_fn,
        step_fn=step_fn,
        grid_shape=(H, W),
        embedding_dim=d,
    )
    return provider

    # El resto del código de la función debe estar correctamente indentado aquí
    # --------------------------------------------------------------
    n_macro_rows = n_macro_cols = 3
    macro_h = H // n_macro_rows
    macro_w = W // n_macro_cols

    if (H, W) == (12, 12):
        # Cada macro 4x4 -> subdividimos en 2x2 micro-bloques de 2x2
        n_micro_rows = n_micro_cols = 2
    else:
        # (150,150): cada macro 50x50 -> subdividimos en 5x5 micro-bloques de 10x10
        n_micro_rows = n_micro_cols = 5

    micro_h = macro_h // n_micro_rows
    micro_w = macro_w // n_micro_cols

    n_macro = n_macro_rows * n_macro_cols
    n_micro_per_macro = n_micro_rows * n_micro_cols
    n_micro_total = n_macro * n_micro_per_macro

    # Índice de macro-bloque por celda
    macro_index = np.empty((H, W), dtype=int)
    # Índice de micro-bloque global por celda
    micro_index = np.empty((H, W), dtype=int)
    # Para cada micro-bloque m, qué macro k(m) le corresponde
    micro_parent_macro = np.empty(n_micro_total, dtype=int)

    m_counter = 0
    for iM in range(n_macro_rows):
        for jM in range(n_macro_cols):
            k = iM * n_macro_cols + jM
            rM0 = iM * macro_h
            cM0 = jM * macro_w
            macro_index[rM0 : rM0 + macro_h, cM0 : cM0 + macro_w] = k

            for iR in range(n_micro_rows):
                for jR in range(n_micro_cols):
                    m_local = iR * n_micro_cols + jR
                    m_global = k * n_micro_per_macro + m_local
                    micro_parent_macro[m_global] = k

                    r0 = rM0 + iR * micro_h
                    c0 = cM0 + jR * micro_w
                    micro_index[r0 : r0 + micro_h, c0 : c0 + micro_w] = m_global

                    m_counter += 1

    if m_counter != n_micro_total:
        raise RuntimeError(
            f"Construcción de micro-bloques inconsistente: esperados {n_micro_total}, "
            f"construidos {m_counter}."
        )

    # --------------------------------------------------------------
    # 2) Máscaras geométricas (para embedding)
    #    P0: global
    #    P1: macros en diagonal
    #    P2: macros fuera de diag
    #    P3: micro TL de cada macro
    #    P4: micro BR de cada macro
    # --------------------------------------------------------------
    mask_global = np.ones((H, W), dtype=bool)
    mask_macro_diag = np.zeros((H, W), dtype=bool)
    mask_macro_offdiag = np.zeros((H, W), dtype=bool)
    mask_micro_tl = np.zeros((H, W), dtype=bool)
    mask_micro_br = np.zeros((H, W), dtype=bool)

    for iM in range(n_macro_rows):
        for jM in range(n_macro_cols):
            rM0 = iM * macro_h
            cM0 = jM * macro_w

            if iM == jM:
                mask_macro_diag[rM0 : rM0 + macro_h, cM0 : cM0 + macro_w] = True
            else:
                mask_macro_offdiag[rM0 : rM0 + macro_h, cM0 : cM0 + macro_w] = True

            # micro TL dentro de la macro
            r0_tl = rM0
            c0_tl = cM0
            mask_micro_tl[r0_tl : r0_tl + micro_h, c0_tl : c0_tl + micro_w] = True

            # micro BR dentro de la macro
            r0_br = rM0 + (n_micro_rows - 1) * micro_h
            c0_br = cM0 + (n_micro_cols - 1) * micro_w
            mask_micro_br[r0_br : r0_br + micro_h, c0_br : c0_br + micro_w] = True

    pattern_masks = [
        mask_macro_diag,
        mask_macro_offdiag,
        mask_micro_tl,
        mask_micro_br,
    ]

    # --------------------------------------------------------------
    # 3) Embedding: resumen fractal de X_t
    # --------------------------------------------------------------
    def _build_embedding(X: np.ndarray) -> np.ndarray:
        features: list[float] = []

        # 3.1 global
        features.append(float(X.mean()))
        features.append(float(X.std()))
        features.append(float(X.min()))
        features.append(float(X.max()))

        # 3.2 medias por macro (orden k = 0..8)
        for k in range(n_macro):
            mask_k = macro_index == k
            features.append(float(X[mask_k].mean()))

        # 3.3 medias por patrones (diag, off-diag, micro TL, micro BR)
        for mask in pattern_masks:
            features.append(float(X[mask].mean()))

        f = np.asarray(features, dtype=float)

        if f.shape[0] >= d:
            # recortamos
            return f[:d]

        # rellenamos con celdas crudas si falta dimensión
        flat = X.ravel()
        needed = d - f.shape[0]
        if needed <= flat.size:
            extra = flat[:needed]
        else:
            extra = np.pad(flat, (0, needed - flat.size), mode="constant")
        return np.concatenate([f, extra])

    # --------------------------------------------------------------
    # 4) Parámetros de dinámica fractal vs no fractal
    # --------------------------------------------------------------
    fractal_main = dict(
        a_g=0.96,
        sigma_g=0.05,
        a_q=0.93,
        b_q=0.20,
        sigma_q=0.06,
        a_r=0.90,
        b_r=0.30,
        sigma_r=0.08,
        sigma_cell=0.05,
    )

    fractal_ood = dict(
        a_g=0.90,
        sigma_g=0.07,
        a_q=0.88,
        b_q=0.22,
        sigma_q=0.08,
        a_r=0.86,
        b_r=0.32,
        sigma_r=0.10,
        sigma_cell=0.06,
    )

    nonfr_main = dict(alpha=0.94, sigma=0.10)
    nonfr_ood = dict(alpha=0.88, sigma=0.14)

    # Latentes / estado de mundo (se mantienen en el closure)
    G: float | None = None          # global
    Q: np.ndarray | None = None     # (n_macro,)
    R: np.ndarray | None = None     # (n_micro_total,)
    world_kind: str | None = None   # "fractal" o "nonfr"
    params_current: dict | None = None

    # --------------------------------------------------------------
    # 5) init_fn: elige mundo y resetea latentes
    # --------------------------------------------------------------
    def init_fn(condition: str, rng: np.random.Generator) -> np.ndarray:
            nonlocal G, Q, R, world_kind, params_current

            if condition == "fractal_main":
                world_kind = "fractal"
                params_current = fractal_main
            elif condition == "fractal_ood":
                world_kind = "fractal"
                params_current = fractal_ood
            elif condition == "nonfr_main":
                world_kind = "nonfr"
                params_current = nonfr_main
            elif condition == "nonfr_ood":
                world_kind = "nonfr"
                params_current = nonfr_ood
            else:
                raise ValueError(f"Condición desconocida: {condition!r}")

            if world_kind == "fractal":
                # Inicializamos latentes G, Q, R y renderizamos X_0
                G = float(rng.normal(scale=0.1))
                Q = rng.normal(scale=0.1, size=n_macro)
                R = rng.normal(scale=0.1, size=n_micro_total)

                base = G + Q[macro_index] + R[micro_index]
                X0 = base + rng.normal(
                    scale=params_current["sigma_cell"],
                    size=(H, W),
                )
            else:
                # Mundo plano: sólo AR(1) sobre la grilla
                G = None
                Q = None
                R = None
                X0 = rng.normal(scale=params_current["sigma"], size=(H, W))



@dataclass(frozen=True)
class F1InductiveScenarioResult:
    # Resultado de ejecutar un escenario F1-IND fractal vs no fractal.

    spec: F1InductiveScenarioSpec
    result_fractal: Phase1InductiveExperimentResult
    result_nonfractal: Phase1InductiveExperimentResult
    delta_s_f1: float
    delta_e_in: float
    delta_e_ood: float


# ---------------------------------------------------------------------------
# Utilidades internas: observables fractales y datasets F1
# ---------------------------------------------------------------------------


def _compute_fractal_observables_for_sequence(
    X_seq: np.ndarray,
    level_masks: Sequence[np.ndarray],
) -> np.ndarray:
    # Calcula g_s(t) para cada nivel fractal s y cada tiempo t.
    #
    # Parámetros
    # ----------
    # X_seq:
    #     Estados del mundo con forma (T, H, W).
    # level_masks:
    #     Secuencia de máscaras booleanas con forma (H, W) cada una.
    #
    # Devuelve
    # --------
    # g_all : np.ndarray
    #     Array de forma (T, S) donde S = len(level_masks) y
    #     g_all[t, s] = promedio de X_seq[t] sobre P_s.
    if X_seq.ndim != 3:
        raise ValueError("X_seq debe tener forma (T, H, W).")
    T, H, W = X_seq.shape
    S = len(level_masks)
    if S == 0:
        raise ValueError("Se requiere al menos una máscara fractal.")

    g_all = np.zeros((T, S), dtype=float)

    for s, mask in enumerate(level_masks):
        if mask.shape != (H, W):
            raise ValueError(
                f"Máscara fractal {s} tiene forma incompatible: "
                f"{mask.shape} != {(H, W)}."
            )
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            raise ValueError(f"La máscara fractal {s} no contiene celdas activas.")
        vals = X_seq[:, mask_bool]
        g_all[:, s] = vals.mean(axis=1)

    return g_all


def _build_phase1_inductive_dataset_from_sequences_pair(
    X_main: np.ndarray,
    E_main: np.ndarray,
    X_ood: np.ndarray,
    E_ood: np.ndarray,
    level_masks: Sequence[np.ndarray],
    law_coeffs: Sequence[float],
    window_size: int,
    n_train: int,
    n_test: int,
    n_ood: int,
) -> Phase1InductiveDataset:
    """
    Construye un Phase1InductiveDataset a partir de dos secuencias:

    - Secuencia "main": usada para train + test (in-distribution).
    - Secuencia OOD:    usada para ejemplos fuera de distribución.

    X_* deben tener forma (T, H, W) y E_* forma (T, d).
    """
    if X_main.ndim != 3 or X_ood.ndim != 3:
        raise ValueError("X_main y X_ood deben tener forma (T, H, W).")
    if E_main.ndim != 2 or E_ood.ndim != 2:
        raise ValueError("E_main y E_ood deben tener forma (T, d).")

    T_main, H_main, W_main = X_main.shape
    T_ood, H_ood, W_ood = X_ood.shape
    if (H_main, W_main) != (H_ood, W_ood):
        raise ValueError("X_main y X_ood deben compartir (H, W).")

    if E_main.shape[0] != T_main or E_ood.shape[0] != T_ood:
        raise ValueError("Dimensión temporal de E_* debe coincidir con X_*.")

    d_main = E_main.shape[1]
    d_ood = E_ood.shape[1]
    if d_main != d_ood:
        raise ValueError("E_main y E_ood deben tener la misma dimensión de embedding.")

    S = len(level_masks)
    law_coeffs_arr = np.asarray(law_coeffs, dtype=float)
    if law_coeffs_arr.shape != (S,):
        raise ValueError(
            f"law_coeffs debe tener forma ({S},), pero tiene {law_coeffs_arr.shape}."
        )

    if window_size < 1:
        raise ValueError("window_size debe ser >= 1.")

    # Observables fractales g_s(t) para ambas secuencias.
    g_main = _compute_fractal_observables_for_sequence(X_main, level_masks)
    g_ood = _compute_fractal_observables_for_sequence(X_ood, level_masks)

    # Para cada t, usamos ventana sobre embeddings e_{t-k+1},...,e_t
    # y target y_t definido en t+1.
    L_main = T_main - window_size
    L_ood = T_ood - window_size
    if L_main <= 0 or L_ood <= 0:
        raise ValueError("Las secuencias son demasiado cortas para la ventana temporal.")

    if n_train + n_test > L_main:
        raise ValueError(
            f"No hay suficientes ejemplos main: n_train + n_test = {n_train + n_test} "
            f"> L_main = {L_main}."
        )
    if n_ood > L_ood:
        raise ValueError(
            f"No hay suficientes ejemplos OOD: n_ood = {n_ood} > L_ood = {L_ood}."
        )

    n_features = window_size * d_main
    X_train = np.zeros((n_train, n_features), dtype=float)
    y_train = np.zeros((n_train,), dtype=float)
    X_test = np.zeros((n_test, n_features), dtype=float)
    y_test = np.zeros((n_test,), dtype=float)
    X_ood_out = np.zeros((n_ood, n_features), dtype=float)
    y_ood = np.zeros((n_ood,), dtype=float)

    def build_examples(
        X_seq: np.ndarray,
        E_seq: np.ndarray,
        g_seq: np.ndarray,
        n_examples: int,
        offset_t: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Construye ventanas y targets a partir de una secuencia dada,
        # comenzando en un índice t = offset_t.
        T_local = X_seq.shape[0]
        max_examples = T_local - window_size - offset_t
        if n_examples > max_examples:
            raise ValueError(
                f"n_examples = {n_examples} excede los ejemplos posibles "
                f"desde offset_t={offset_t} (máx {max_examples})."
            )

        X_out = np.zeros((n_examples, n_features), dtype=float)
        y_out = np.zeros((n_examples,), dtype=float)

        for i in range(n_examples):
            # último índice de la ventana de embeddings
            t = offset_t + i + window_size - 1
            # target usa X_{t+1} → g_{t+1}
            t_target = t + 1
            if t_target >= T_local:
                raise ValueError(
                    "Índice temporal fuera de rango al construir ejemplos."
                )

            window = E_seq[t - window_size + 1 : t + 1]  # (k, d)
            X_out[i, :] = window.reshape(-1)

            g_t1 = g_seq[t_target, :]  # (S,)
            y_out[i] = float(g_t1 @ law_coeffs_arr)

        return X_out, y_out

    # Train y test desde la secuencia main.
    X_train[:, :], y_train[:] = build_examples(
        X_main,
        E_main,
        g_main,
        n_examples=n_train,
        offset_t=0,
    )
    X_test[:, :], y_test[:] = build_examples(
        X_main,
        E_main,
        g_main,
        n_examples=n_test,
        offset_t=n_train,
    )

    # OOD desde la secuencia OOD.
    X_ood_out[:, :], y_ood[:] = build_examples(
        X_ood,
        E_ood,
        g_ood,
        n_examples=n_ood,
        offset_t=0,
    )

    return Phase1InductiveDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_ood=X_ood_out,
        y_ood=y_ood,
    )


def _compute_deltas(
    result_fractal: Phase1InductiveExperimentResult,
    result_nonfractal: Phase1InductiveExperimentResult,
) -> Tuple[float, float, float]:
    # Calcula diferencias clave entre condiciones fractal y no fractal.
    m_f = result_fractal.phase1_result.metrics
    m_n = result_nonfractal.phase1_result.metrics

    delta_s_f1 = m_f.s_f1 - m_n.s_f1
    delta_e_in = result_fractal.raw_metrics.e_in - result_nonfractal.raw_metrics.e_in
    delta_e_ood = result_fractal.raw_metrics.e_ood - result_nonfractal.raw_metrics.e_ood

    return float(delta_s_f1), float(delta_e_in), float(delta_e_ood)


# ---------------------------------------------------------------------------
# Ejecución de escenarios F1-IND (fractal vs no fractal)
# ---------------------------------------------------------------------------


def run_f1_inductive_scenario(
    spec: F1InductiveScenarioSpec,
    world_provider: WorldSequenceProvider,
    level_masks: Sequence[np.ndarray],
    rng: np.random.Generator | None = None,
) -> F1InductiveScenarioResult:
    # Ejecución de escenario F1-IND (fractal vs no fractal)
    # Ejecuta un escenario F1-IND fractal vs no fractal usando un proveedor
    # de secuencias de mundo + embeddings.
    #
    # Parámetros
    # ----------
    # spec:
    #     Especificación del escenario (ventana, tamaños de datasets, ley y parámetros).
    # world_provider:
    #     Implementación concreta que llama al mini-mundo real y a la MFM.
    # level_masks:
    #     Máscaras fractales (H, W) compartidas entre todas las condiciones.
    # rng:
    #     Generador de números aleatorios compartido, para reproducibilidad.
    if rng is None:
        rng = np.random.default_rng()

    # Longitud mínima de secuencias para poder construir
    # n_train + n_test ejemplos (main) y n_ood ejemplos (OOD).
    min_T_main = spec.window_size + spec.n_train + spec.n_test + 1
    min_T_ood = spec.window_size + spec.n_ood + 1

    # --------------------------
    # Rama fractal
    # --------------------------
    X_main_f, E_main_f = world_provider("fractal_main", min_T_main, rng)
    X_ood_f, E_ood_f = world_provider("fractal_ood", min_T_ood, rng)

    dataset_fractal = _build_phase1_inductive_dataset_from_sequences_pair(
        X_main=X_main_f,
        E_main=E_main_f,
        X_ood=X_ood_f,
        E_ood=E_ood_f,
        level_masks=level_masks,
        law_coeffs=spec.law_coeffs,
        window_size=spec.window_size,
        n_train=spec.n_train,
        n_test=spec.n_test,
        n_ood=spec.n_ood,
    )

    embedding_dim = dataset_fractal.X_train.shape[1] // spec.window_size

    config_fractal = Phase1InductiveConfig(
        window_size=spec.window_size,
        embedding_dim=embedding_dim,
        train_steps=dataset_fractal.X_train.shape[0],
        test_steps=dataset_fractal.X_test.shape[0],
        ood_steps=dataset_fractal.X_ood.shape[0],
        lambda_cost=spec.lambda_cost,
        gamma_k=spec.gamma_k,
        gamma_mdl=spec.gamma_mdl,
    )

    result_fractal = run_phase1_inductive_experiment(
        config=config_fractal,
        dataset=dataset_fractal,
        l2_reg=0.0,
    )

    # --------------------------
    # Rama no fractal
    # --------------------------
    X_main_nf, E_main_nf = world_provider("nonfr_main", min_T_main, rng)
    X_ood_nf, E_ood_nf = world_provider("nonfr_ood", min_T_ood, rng)

    dataset_nonfr = _build_phase1_inductive_dataset_from_sequences_pair(
        X_main=X_main_nf,
        E_main=E_main_nf,
        X_ood=X_ood_nf,
        E_ood=E_ood_nf,
        level_masks=level_masks,
        law_coeffs=spec.law_coeffs,
        window_size=spec.window_size,
        n_train=spec.n_train,
        n_test=spec.n_test,
        n_ood=spec.n_ood,
    )

    config_nonfr = Phase1InductiveConfig(
        window_size=spec.window_size,
        embedding_dim=embedding_dim,
        train_steps=dataset_nonfr.X_train.shape[0],
        test_steps=dataset_nonfr.X_test.shape[0],
        ood_steps=dataset_nonfr.X_ood.shape[0],
        lambda_cost=spec.lambda_cost,
        gamma_k=spec.gamma_k,
        gamma_mdl=spec.gamma_mdl,
    )

    result_nonfr = run_phase1_inductive_experiment(
        config=config_nonfr,
        dataset=dataset_nonfr,
        l2_reg=0.0,
    )

    delta_s_f1, delta_e_in, delta_e_ood = _compute_deltas(
        result_fractal,
        result_nonfr,
    )

    return F1InductiveScenarioResult(
        spec=spec,
        result_fractal=result_fractal,
        result_nonfractal=result_nonfr,
        delta_s_f1=delta_s_f1,
        delta_e_in=delta_e_in,
        delta_e_ood=delta_e_ood,
    )


# ---------------------------------------------------------------------------
# Proveedor de mini-mundo de juguete para PMV v0
# (luego se reemplaza por mini_world_boxes + MFM reales)
# ---------------------------------------------------------------------------
    nonfr_main_params = dict(
        alpha=0.92,
        sigma=0.20,
    )
    nonfr_ood_params = dict(
        alpha=0.80,
        sigma=0.28,
    )

    world_kind: str | None = None  # "fractal" o "nonfr"
    params_current: dict[str, float] | None = None

    # Latentes para el modo fractal
    G: float = 0.0
    Q: np.ndarray = np.zeros(n_macro, dtype=float)

    def _render_fractal_grid(
        G_val: float,
        Q_vec: np.ndarray,
        sigma_cell: float,
        rng_local: np.random.Generator,
    ) -> np.ndarray:
        # Construye X_t a partir de G(t) y Q_k(t) sobre la partición en macro-bloques.
        # G_val es escalar, Q_vec[k] para cada macro-bloque k.
        base = G_val + Q_vec[macro_index]  # broadcast a (H, W)
        noise = rng_local.normal(loc=0.0, scale=sigma_cell, size=(H, W))
        return base + noise

    def _build_embedding_from_grid(X: np.ndarray) -> np.ndarray:
        # Embedding basado en estadísticas globales y por macro-bloque.
        # base_features = [mean_global, std_global, min_global, max_global,
        #                  mean_block_0, ..., mean_block_8]
        # Si embedding_dim <= len(base_features), se recortan.
        # Si embedding_dim > len(base_features), se completan con valores de la grilla aplanada.
        if X.shape != (H, W):
            raise ValueError(f"X tiene forma {X.shape}, se esperaba {(H, W)}.")

        mean_val = float(X.mean())
        std_val = float(X.std())
        min_val = float(X.min())
        max_val = float(X.max())

        block_means = []
        for k in range(n_macro):
            block_means.append(float(X[macro_index == k].mean()))
        block_means_arr = np.array(block_means, dtype=float)

        base_features = np.concatenate(
            [
                np.array([mean_val, std_val, min_val, max_val], dtype=float),
                block_means_arr,
            ],
            axis=0,
        )  # longitud 4 + 9 = 13

        d = embedding_dim
        if d <= base_features.size:
            emb = base_features[:d]
        else:
            flat = X.reshape(-1)
            extra_needed = d - base_features.size
            if extra_needed <= flat.size:
                extra = flat[:extra_needed]
            else:
                reps = (extra_needed + flat.size - 1) // flat.size
                extra = np.tile(flat, reps)[:extra_needed]
            emb = np.concatenate([base_features, extra], axis=0)

        assert emb.shape == (d,)
        return emb

    def init_fn(condition: str, rng: np.random.Generator) -> np.ndarray:
        nonlocal world_kind, params_current, G, Q

        if condition == "fractal_main":
            world_kind = "fractal"
            params_current = fractal_main_params
        elif condition == "fractal_ood":
            world_kind = "fractal"
            params_current = fractal_ood_params
        elif condition == "nonfr_main":
            world_kind = "nonfr"
            params_current = nonfr_main_params
        elif condition == "nonfr_ood":
            world_kind = "nonfr"
            params_current = nonfr_ood_params
        else:
            raise ValueError(f"Condición desconocida para BoxWorld toy: {condition!r}")

        if world_kind == "fractal":
            # Inicializar latentes G y Q
            G = rng.normal(loc=0.0, scale=1.0)
            Q = rng.normal(loc=0.0, scale=1.0, size=n_macro)
            sigma_cell = float(params_current["sigma_cell"])
            X0 = _render_fractal_grid(G, Q, sigma_cell, rng)
        else:
            # Mundo no fractal: grilla inicial con ruido suave
            X0 = rng.normal(loc=0.0, scale=1.0, size=(H, W))

        return X0

    def step_fn(state: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        nonlocal world_kind, params_current, G, Q

        if world_kind is None or params_current is None:
            raise RuntimeError("init_fn debe llamarse antes de step_fn en BoxWorld toy.")

        if state.shape != (H, W):
            raise ValueError(f"state tiene forma {state.shape}, se esperaba {(H, W)}.")

        if world_kind == "fractal":
            # Actualizar latentes fractales
            a_g = float(params_current["a_g"])
            sigma_g = float(params_current["sigma_g"])
            a_q = float(params_current["a_q"])
            b_q = float(params_current["b_q"])
            sigma_q = float(params_current["sigma_q"])
            sigma_cell = float(params_current["sigma_cell"])

            # G(t+1)
            G = a_g * G + rng.normal(loc=0.0, scale=sigma_g)

            # Q_k(t+1) para cada macro-bloque k
            noise_q = rng.normal(loc=0.0, scale=sigma_q, size=n_macro)
            Q = a_q * Q + b_q * G + noise_q

            X_next = _render_fractal_grid(G, Q, sigma_cell, rng)

        else:
            # Mundo no fractal: AR(1) independiente por celda
            alpha = float(params_current["alpha"])
            sigma = float(params_current["sigma"])
            noise = rng.normal(loc=0.0, scale=sigma, size=(H, W))
            X_next = alpha * state + noise

        emb = _build_embedding_from_grid(X_next)
        return X_next, emb

    provider = GenericWorldSequenceProvider(
        init_fn=init_fn,
        step_fn=step_fn,
        grid_shape=(H, W),
        embedding_dim=embedding_dim,
    )
    return provider

@dataclass(frozen=True)
class F1ScenarioRunSummary:
    # Resumen de métricas F1 para una sola corrida de escenario.
    seed: int
    s_f1_fractal: float
    s_f1_nonfractal: float
    delta_s_f1: float
    e_in_fractal: float
    e_in_nonfractal: float
    e_ood_fractal: float
    e_ood_nonfractal: float
    mdl_fractal: float
    mdl_nonfractal: float
    k_fractal: int
    k_nonfractal: int
    cost_fractal: float
    cost_nonfractal: float


@dataclass(frozen=True)
class F1ScenarioBatchSummary:
    # Resumen agregado de un escenario F1-IND sobre varias semillas.
    spec: F1InductiveScenarioSpec
    seeds: Sequence[int]
    runs: Sequence[F1ScenarioRunSummary]
    mean_s_f1_fractal: float
    mean_s_f1_nonfractal: float
    mean_delta_s_f1: float
    std_delta_s_f1: float
    mean_e_in_fractal: float
    mean_e_in_nonfractal: float
    mean_e_ood_fractal: float
    mean_e_ood_nonfractal: float


def run_f1_inductive_scenario_multi_seed(
    spec: F1InductiveScenarioSpec,
    world_provider: WorldSequenceProvider,
    level_masks: Sequence[np.ndarray],
    seeds: Sequence[int],
) -> F1ScenarioBatchSummary:
    # Ejecuta un escenario F1-IND sobre varias semillas y agrega las métricas clave.
    if not seeds:
        raise ValueError("seeds no puede estar vacío.")

    run_summaries: list[F1ScenarioRunSummary] = []

    s_f1_f_vals = []
    s_f1_n_vals = []
    delta_vals = []
    e_in_f_vals = []
    e_in_n_vals = []
    e_ood_f_vals = []
    e_ood_n_vals = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        scenario_result = run_f1_inductive_scenario(
            spec=spec,
            world_provider=world_provider,
            level_masks=level_masks,
            rng=rng,
        )

        m_f = scenario_result.result_fractal.phase1_result.metrics
        m_n = scenario_result.result_nonfractal.phase1_result.metrics
        r_f = scenario_result.result_fractal.raw_metrics
        r_n = scenario_result.result_nonfractal.raw_metrics

        summary = F1ScenarioRunSummary(
            seed=seed,
            s_f1_fractal=m_f.s_f1,
            s_f1_nonfractal=m_n.s_f1,
            delta_s_f1=scenario_result.delta_s_f1,
            e_in_fractal=r_f.e_in,
            e_in_nonfractal=r_n.e_in,
            e_ood_fractal=r_f.e_ood,
            e_ood_nonfractal=r_n.e_ood,
            mdl_fractal=r_f.mdl,
            mdl_nonfractal=r_n.mdl,
            k_fractal=r_f.k_interactions,
            k_nonfractal=r_n.k_interactions,
            cost_fractal=r_f.cost,
            cost_nonfractal=r_n.cost,
        )
        run_summaries.append(summary)

        s_f1_f_vals.append(m_f.s_f1)
        s_f1_n_vals.append(m_n.s_f1)
        delta_vals.append(scenario_result.delta_s_f1)
        e_in_f_vals.append(r_f.e_in)
        e_in_n_vals.append(r_n.e_in)
        e_ood_f_vals.append(r_f.e_ood)
        e_ood_n_vals.append(r_n.e_ood)

    mean_s_f1_fractal = float(np.mean(s_f1_f_vals))
    mean_s_f1_nonfractal = float(np.mean(s_f1_n_vals))
    mean_delta_s_f1 = float(np.mean(delta_vals))
    std_delta_s_f1 = float(np.std(delta_vals))

    mean_e_in_fractal = float(np.mean(e_in_f_vals))
    mean_e_in_nonfractal = float(np.mean(e_in_n_vals))
    mean_e_ood_fractal = float(np.mean(e_ood_f_vals))
    mean_e_ood_nonfractal = float(np.mean(e_ood_n_vals))

    return F1ScenarioBatchSummary(
        spec=spec,
        seeds=list(seeds),
        runs=run_summaries,
        mean_s_f1_fractal=mean_s_f1_fractal,
        mean_s_f1_nonfractal=mean_s_f1_nonfractal,
        mean_delta_s_f1=mean_delta_s_f1,
        std_delta_s_f1=std_delta_s_f1,
        mean_e_in_fractal=mean_e_in_fractal,
        mean_e_in_nonfractal=mean_e_in_nonfractal,
        mean_e_ood_fractal=mean_e_ood_fractal,
        mean_e_ood_nonfractal=mean_e_ood_nonfractal,
    )
