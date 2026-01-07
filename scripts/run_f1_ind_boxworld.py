import numpy as np

from rnfe.pmv.experiments.scenarios_f1 import (
    F1InductiveScenarioSpec,
    run_f1_inductive_scenario,
    make_boxworld_provider,
)

def main():
    grid_shape = (4, 4)      # H, W reales de tu BoxWorld
    embedding_dim = 8        # dimensión real de tu embedding MFM

    provider = make_boxworld_provider(
        grid_shape=grid_shape,
        embedding_dim=embedding_dim,
    )

    # Máscaras fractales (ejemplo mínima; aquí irían tus máscaras reales)
    H, W = grid_shape
    mask0 = np.zeros((H, W), dtype=bool); mask0[0, 0] = True
    mask1 = np.zeros((H, W), dtype=bool); mask1[0, 1] = True
    level_masks = [mask0, mask1]

    spec = F1InductiveScenarioSpec(
        scenario_id="F1_IND_boxworld_real_v0",
        window_size=3,
        n_train=24,
        n_test=8,
        n_ood=8,
        law_coeffs=[1.0, 0.0],  # solo primer nivel, puedes ajustar
        lambda_cost=0.1,
        gamma_k=1.0,
        gamma_mdl=1.0,
    )

    rng = np.random.default_rng(123)

    result = run_f1_inductive_scenario(
        spec=spec,
        world_provider=provider,
        level_masks=level_masks,
        rng=rng,
    )

    m_f = result.result_fractal.phase1_result.metrics
    m_n = result.result_nonfractal.phase1_result.metrics

    print("S_F1 fractal    :", m_f.s_f1)
    print("S_F1 no fractal :", m_n.s_f1)
    print("ΔS_F1           :", result.delta_s_f1)

if __name__ == "__main__":
    main()
