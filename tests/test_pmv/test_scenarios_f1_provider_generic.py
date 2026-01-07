import numpy as np

from rnfe.pmv.experiments.scenarios_f1 import (
    GenericWorldSequenceProvider,
)


def test_generic_world_sequence_provider_basic_shapes():
    rng = np.random.default_rng(123)
    H, W = 3, 3
    d = 4

    def init_fn(condition: str, rng_local: np.random.Generator) -> np.ndarray:
        # estado inicial simple: todo ceros salvo una celda codificando la condición
        state = np.zeros((H, W), dtype=float)
        if condition == "fractal_main":
            state[0, 0] = 1.0
        elif condition == "nonfr_main":
            state[0, 1] = 1.0
        else:
            state[1, 1] = 0.5
        return state

    def step_fn(state: np.ndarray, rng_local: np.random.Generator):
        # Mundo de juguete: incrementa todo en 1 y genera un embedding sencillo
        next_state = state + 1.0
        t_val = float(np.mean(next_state))
        emb = np.array(
            [
                t_val,
                t_val**2,
                rng_local.normal(scale=0.1),
                1.0,
            ],
            dtype=float,
        )
        return next_state, emb

    provider = GenericWorldSequenceProvider(
        init_fn=init_fn,
        step_fn=step_fn,
        grid_shape=(H, W),
        embedding_dim=d,
    )

    n_steps = 10
    X_seq, E_seq = provider("fractal_main", n_steps, rng)

    assert X_seq.shape == (n_steps, H, W)
    assert E_seq.shape == (n_steps, d)

    # Comprobar que la evolución es monótona (cada paso suma 1)
    for t in range(1, n_steps):
        diff = X_seq[t] - X_seq[t - 1]
        assert np.allclose(diff, 1.0)
