from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

EPS = 1e-8


def _clip01(x: float) -> float:
    """Recorta un valor al intervalo [0, 1] sin usar funciones externas."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@dataclass(frozen=True)
class Phase1Config:
    """
    Configuración mínima para una corrida de Fase 1 (unimodal).

    En esta versión PMV v0 no acoplamos aquí hiperparámetros del modelo;
    solo metadatos básicos y el peso de coste.
    """
    mode_name: str
    lambda_cost: float = 0.1  # penalización del coste físico en S_F1


@dataclass(frozen=True)
class Phase1RawMetrics:
    """
    Métricas "crudas" de un modo de razonamiento antes de normalizar.

    Todos los valores son escalares agregados sobre una corrida
    (por ejemplo, errores medios, coste acumulado).
    """

    # Errores en distribución y fuera de distribución
    e_in: float
    e_in_base: float
    e_ood: float
    e_ood_base: float

    # Complejidad y eficiencia epistemológica
    mdl: float  # longitud de descripción mínima (o proxy)
    k_interactions: int  # número de interacciones epistemológicas

    # Coste físico agregado (tiempo, VRAM, energía, combinado)
    cost: float


@dataclass(frozen=True)
class Phase1Metrics:
    """
    Métricas normalizadas de F1 para un modo dado.

    Todos los componentes salvo s_f1 están en [0, 1] por construcción.
    """

    # Componentes elementales normalizados
    s_law: float       # comprensión de la ley interna
    s_ood: float       # generalización fuera de distribución
    s_mdl: float       # simplicidad relativa
    s_eff: float       # eficiencia epistemológica relativa
    s_cost: float      # coste físico relativo (0 barato, 1 caro)

    # Agregados intermedios
    s_core: float      # núcleo de comprensión (ley + OOD)
    s_aux: float       # cualidades auxiliares (MDL + eficiencia)

    # Métrica global de F1
    s_f1: float        # S_F1 = s_core * s_aux - lambda_cost * s_cost


@dataclass(frozen=True)
class Phase1Result:
    """
    Resultado de Fase 1 para un modo concreto.

    Contiene:
    - La configuración asociada.
    - El nombre del modo (por ejemplo, "IND").
    - Las métricas crudas usadas para normalizar.
    - Las métricas normalizadas (S_law, S_ood, ..., S_F1).
    """
    config: Phase1Config
    mode_name: str
    raw: Phase1RawMetrics
    metrics: Phase1Metrics


# ---------------------------------------------------------------------------
# Funciones de normalización elemental
# ---------------------------------------------------------------------------


def compute_s_law(e_in: float, e_in_base: float, eps: float = EPS) -> float:
    """
    Normaliza la comprensión de la ley interna.

    S_law = clip(1 - e_in / (e_in_base + eps), 0, 1)

    - Si e_in == e_in_base  -> ~0 (no mejora sobre el baseline).
    - Si e_in -> 0          -> ~1 (comprensión casi perfecta).
    - Si e_in > e_in_base   -> se recorta a 0 (peor que el baseline).
    """
    # Si no mejora o es peor que el baseline, comprensión nula.
    if e_in >= e_in_base:
        return 0.0

    denom = e_in_base + eps
    if denom <= 0.0:
        # Corrida mal formada; se devuelve 0 por seguridad.
        return 0.0

    value = 1.0 - (e_in / denom)
    return _clip01(value)


def compute_s_ood(e_ood: float, e_ood_base: float, eps: float = EPS) -> float:
    """
    Normaliza la generalización fuera de distribución.

    Regla de diseño:
    - Si e_ood >= e_ood_base: no mejora al baseline -> S_ood = 0.0.
    - Si 0 <= e_ood < e_ood_base:
        S_ood = clip(1 - e_ood / (e_ood_base + eps), 0, 1)
    """
    if e_ood >= e_ood_base:
        return 0.0

    denom = e_ood_base + eps
    if denom <= 0.0:
        return 0.0

    value = 1.0 - (e_ood / denom)
    return _clip01(value)


def compute_s_mdl(
    mdl: float,
    mdl_min: float,
    mdl_max: float,
    eps: float = EPS,
) -> float:
    """
    Normaliza la simplicidad relativa dentro de una familia de modos.

    S_mdl = 1 - (mdl - mdl_min) / (mdl_max - mdl_min + eps)

    - El modelo más compacto (mdl = mdl_min) obtiene S_mdl ≈ 1.
    - El más complejo (mdl = mdl_max) obtiene S_mdl ≈ 0.
    """
    span = mdl_max - mdl_min
    value = 1.0 - (mdl - mdl_min) / (span + eps)
    return _clip01(value)


def compute_s_eff(
    k: int,
    k_min: int,
    k_max: int,
    eps: float = EPS,
) -> float:
    """
    Normaliza la eficiencia epistemológica relativa (uso de interacciones).

    S_eff = 1 - (k - k_min) / (k_max - k_min + eps)

    - Menos interacciones (cerca de k_min) -> S_eff ≈ 1.
    - Más interacciones (cerca de k_max) -> S_eff ≈ 0.
    """
    span = float(k_max - k_min)
    value = 1.0 - (float(k) - float(k_min)) / (span + eps)
    return _clip01(value)


def compute_s_cost(
    cost: float,
    cost_min: float,
    cost_max: float,
    eps: float = EPS,
) -> float:
    """
    Normaliza el coste físico relativo dentro de una familia de modos.

    S_cost = (cost - cost_min) / (cost_max - cost_min + eps)

    - El modo más barato (cost = cost_min) -> S_cost ≈ 0.
    - El modo más caro (cost = cost_max)  -> S_cost ≈ 1.
    """
    span = cost_max - cost_min
    value = (cost - cost_min) / (span + eps)
    return _clip01(value)


# ---------------------------------------------------------------------------
# Normalización conjunta sobre una colección de modos
# ---------------------------------------------------------------------------


def compute_phase1_metrics_for_all_modes(
    raw_by_mode: Mapping[str, Phase1RawMetrics],
    lambda_cost: float = 0.1,
) -> Dict[str, Phase1Metrics]:
    """
    Normaliza todas las métricas de F1 para una colección de modos.

    Se asume que raw_by_mode contiene al menos un modo. En el caso
    extremo de un solo modo (PMV v0), los valores min/max coinciden
    y las funciones se encargan de evitar divisiones por cero.

    Parámetros
    ----------
    raw_by_mode:
        Diccionario {nombre_modo -> Phase1RawMetrics} con las métricas
        crudas de cada modo.
    lambda_cost:
        Peso de penalización del coste físico en S_F1.

    Devuelve
    --------
    Dict[str, Phase1Metrics]
        Diccionario {nombre_modo -> métricas normalizadas}.
    """
    if not raw_by_mode:
        raise ValueError("Se requiere al menos un modo para normalizar F1.")

    mdls = [rm.mdl for rm in raw_by_mode.values()]
    ks = [rm.k_interactions for rm in raw_by_mode.values()]
    costs = [rm.cost for rm in raw_by_mode.values()]

    mdl_min = min(mdls)
    mdl_max = max(mdls)
    k_min = min(ks)
    k_max = max(ks)
    cost_min = min(costs)
    cost_max = max(costs)

    metrics_by_mode: Dict[str, Phase1Metrics] = {}

    for mode_name, raw in raw_by_mode.items():
        s_law = compute_s_law(raw.e_in, raw.e_in_base)
        s_ood = compute_s_ood(raw.e_ood, raw.e_ood_base)
        s_mdl = compute_s_mdl(raw.mdl, mdl_min, mdl_max)
        s_eff = compute_s_eff(raw.k_interactions, k_min, k_max)
        s_cost = compute_s_cost(raw.cost, cost_min, cost_max)

        # Núcleo de comprensión
        s_core = 0.5 * (s_law + s_ood)

        # Cualidades auxiliares (simplicidad + eficiencia)
        s_aux = 0.5 * (s_mdl + s_eff)

        # Factor auxiliar: nunca anula la comprensión, solo la modula
        # - si s_aux = 0  -> factor_aux = 0.5
        # - si s_aux = 1  -> factor_aux = 1.0
        factor_aux = 0.5 + 0.5 * s_aux

        # Factor de coste: penaliza suavemente, sin hacer negativo S_F1
        # - si s_cost = 0 -> factor_cost = 1
        # - si s_cost = 1 -> factor_cost = 1 - lambda_cost (por ejemplo 0.9)
        factor_cost = 1.0 - lambda_cost * s_cost
        if factor_cost < 0.0:
            factor_cost = 0.0

        # Métrica global: cascada comprensión -> auxiliares -> coste
        raw_s_f1 = s_core * factor_aux * factor_cost

        # Aseguramos que S_F1 queda en [0, 1]
        if raw_s_f1 < 0.0:
            s_f1 = 0.0
        elif raw_s_f1 > 1.0:
            s_f1 = 1.0
        else:
            s_f1 = raw_s_f1

        metrics_by_mode[mode_name] = Phase1Metrics(
            s_law=s_law,
            s_ood=s_ood,
            s_mdl=s_mdl,
            s_eff=s_eff,
            s_cost=s_cost,
            s_core=s_core,
            s_aux=s_aux,
            s_f1=s_f1,
        )

    return metrics_by_mode


def build_phase1_results(
    config: Phase1Config,
    raw_by_mode: Mapping[str, Phase1RawMetrics],
) -> Dict[str, Phase1Result]:
    """
    Construye Phase1Result para cada modo a partir de las métricas crudas.

    Esta función es un pegamento fino: aplica la normalización conjunta
    y envuelve los resultados en dataclasses Phase1Result.
    """
    metrics_by_mode = compute_phase1_metrics_for_all_modes(
        raw_by_mode=raw_by_mode,
        lambda_cost=config.lambda_cost,
    )

    results: Dict[str, Phase1Result] = {}
    for mode_name, raw in raw_by_mode.items():
        metrics = metrics_by_mode[mode_name]
        result = Phase1Result(
            config=config,
            mode_name=mode_name,
            raw=raw,
            metrics=metrics,
        )
        results[mode_name] = result

    return results

