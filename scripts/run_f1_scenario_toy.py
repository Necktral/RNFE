#!/usr/bin/env python
"""
Toy F1: comparación fractal vs no fractal en BoxWorld 150x150.

- Mundo fractal: latentes G (global), Q_k (macro 3x3), R_m (micro),
  enganchados a las máscaras P0..P4 usadas en F1.
- Mundo no fractal: AR(1) plano.

Ejecuta algo tipo:
    PYTHONPATH=src python scripts/run_f1_scenario_toy.py
"""

import argparse
import numpy as np

from rnfe.pmv.experiments.scenarios_f1 import (
    F1InductiveScenarioSpec,
    run_f1_inductive_scenario_multi_seed,
    make_boxworld_provider,
)

# ----------------------------------------------------------------------
# Máscaras P0..P4 para el análisis de leyes F1 sobre la grilla
# ----------------------------------------------------------------------
def build_level_masks(grid_shape: tuple[int, int]) -> list[np.ndarray]:
    """
    Construye las máscaras geométricas P0..P4 en una grilla cuadrada:

    - P0: todo (global).
    - P1: macros de la diagonal (3x3).
    - P2: macros fuera de la diagonal.
    - P3: micro-bloques TL (top-left) en cada macro.
    - P4: micro-bloques BR (bottom-right) en cada macro.

    Para mantener coherencia con make_boxworld_provider:
    - (12,12): 3x3 macros de 4x4, cada macro subdividida en 2x2 micro-bloques de 2x2.
    - (150,150): 3x3 macros de 50x50, cada macro subdividida en 5x5 micro-bloques de 10x10.
    """
    H, W = grid_shape
    import numpy as np

    if (H, W) not in [(12, 12), (150, 150)]:
        raise ValueError(
            f"build_level_masks toy sólo definido para (12,12) y (150,150); "
            f"recibido grid_shape={grid_shape}."
        )

    n_macro_rows = n_macro_cols = 3
    macro_h = H // n_macro_rows
    macro_w = W // n_macro_cols

    if (H, W) == (12, 12):
        n_micro_rows = n_micro_cols = 2
    else:
        n_micro_rows = n_micro_cols = 5

    micro_h = macro_h // n_micro_rows
    micro_w = macro_w // n_micro_cols

    mask_global = np.ones((H, W), dtype=bool)
    mask_diag = np.zeros((H, W), dtype=bool)
    mask_offdiag = np.zeros((H, W), dtype=bool)
    mask_micro_tl = np.zeros((H, W), dtype=bool)
    mask_micro_br = np.zeros((H, W), dtype=bool)

    for iM in range(n_macro_rows):
        for jM in range(n_macro_cols):
            rM0 = iM * macro_h
            cM0 = jM * macro_w

            if iM == jM:
                mask_diag[rM0 : rM0 + macro_h, cM0 : cM0 + macro_w] = True
            else:
                mask_offdiag[rM0 : rM0 + macro_h, cM0 : cM0 + macro_w] = True

            # micro TL
            r0_tl = rM0
            c0_tl = cM0
            mask_micro_tl[r0_tl : r0_tl + micro_h, c0_tl : c0_tl + micro_w] = True

            # micro BR
            r0_br = rM0 + (n_micro_rows - 1) * micro_h
            c0_br = cM0 + (n_micro_cols - 1) * micro_w
            mask_micro_br[r0_br : r0_br + micro_h, c0_br : c0_br + micro_w] = True

    return [mask_global, mask_diag, mask_offdiag, mask_micro_tl, mask_micro_br]

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=20, help="Tamaño de ventana temporal W.")
    p.add_argument("--n-train", type=int, default=400, help="Número de secuencias train por mundo.")
    p.add_argument("--n-test", type=int, default=400, help="Número de secuencias test IN por mundo.")
    p.add_argument(
        "--n-ood",
        type=int,
        default=400,
        help="Número de secuencias OOD por mundo.",
    )
    p.add_argument("--seeds", type=int, default=5, help="Número de semillas para promediar.")
    p.add_argument("--lambda-cost", type=float, default=0.0, help="Peso de regularización de coste.")
    p.add_argument("--gamma-k", type=float, default=0.0, help="Peso de penalización en K.")
    p.add_argument("--gamma-mdl", type=float, default=0.0, help="Peso de penalización MDL.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(12345)

    # Grilla grande para este experimento
    grid_shape = (150, 150)
    embedding_dim = 64

    world_provider = make_boxworld_provider(
        grid_shape=grid_shape,
        embedding_dim=embedding_dim,
    )

    level_masks = build_level_masks(grid_shape)

    # Coeficientes para la ley de coste F1 (puedes afinarlos después)
    law_coeffs = np.asarray([1.0, 0.5, 0.5, 0.25, 0.25], dtype=float)

    spec = F1InductiveScenarioSpec(
        grid_shape=grid_shape,
        embedding_dim=embedding_dim,
        window_size=args.window,
        n_train=args.n_train,
        n_test=args.n_test,
        n_ood=args.n_ood,
        level_masks=level_masks,
        law_coeffs=law_coeffs,
        lambda_cost=args.lambda_cost,
        gamma_k=args.gamma_k,
        gamma_mdl=args.gamma_mdl,
    )

    summary = run_f1_inductive_scenario_multi_seed(
        spec=spec,
        world_provider=world_provider,
        n_seeds=args.seeds,
        base_seed=42,
    )

    # Resumen rápido en consola
    print("=== F1 BoxWorld 150x150 ===")
    print(f"grid_shape        : {grid_shape}")
    print(f"embedding_dim     : {embedding_dim}")
    print(f"window_size       : {args.window}")
    print(f"n_train / test / ood : {args.n_train} / {args.n_test} / {args.n_ood}")
    print()
    print("=== métrica S_F1 (fractal vs no fractal) ===")
    print(f"S_F1(fractal)     : {summary.result_fractal.s_f1:.6f}")
    print(f"S_F1(no fractal)  : {summary.result_nonfractal.s_f1:.6f}")
    print(f"Δ S_F1            : {summary.delta_s_f1:.6f}")
    print()
    print("=== error IN / OOD ===")
    print(f"E_in(fractal)     : {summary.result_fractal.e_in:.6f}")
    print(f"E_in(no fractal)  : {summary.result_nonfractal.e_in:.6f}")
    print(f"Δ E_in            : {summary.delta_e_in:.6f}")
    print()
    print(f"E_ood(fractal)    : {summary.result_fractal.e_ood:.6f}")
    print(f"E_ood(no fractal) : {summary.result_nonfractal.e_ood:.6f}")
    print(f"Δ E_ood           : {summary.delta_e_ood:.6f}")


if __name__ == "__main__":
    main()
