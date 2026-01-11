from __future__ import annotations

import argparse
from rnfe.pmv.reasoning.deductive_d2_1_stratneg import (
    generate_ded_d2_1_tasks,
    generate_ded_d2_1_trap_tasks,
    evaluate_d2_1,
    export_jsonl,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_kb", type=int, default=200)
    ap.add_argument("--queries_per_kb", type=int, default=6)
    ap.add_argument("--trap_n", type=int, default=500)
    ap.add_argument("--out", type=str, default="ded_d2_1.jsonl")
    args = ap.parse_args()

    tasks = []

    for difficulty in ["easy", "mid", "hard"]:
        for split in ["id", "ood"]:
            part = generate_ded_d2_1_tasks(
                seed=args.seed + (hash((difficulty, split, "d2_1")) % 10_000),
                n_kb=args.n_kb,
                queries_per_kb=args.queries_per_kb,
                split=split,
                difficulty=difficulty,
            )
            tasks.extend(part)

    # “Trap” no estratificable
    tasks.extend(generate_ded_d2_1_trap_tasks(seed=args.seed + 1337, n=args.trap_n))

    metrics = evaluate_d2_1(tasks)
    print(
        "[DED–D2.1] "
        f"n={metrics.n} "
        f"acc_valid={metrics.acc_valid:.6f} "
        f"exception_rate_valid={metrics.exception_rate_valid:.6f} "
        f"invalid_detection_rate={metrics.invalid_detection_rate:.6f} "
        f"wall_total={metrics.wall_seconds_total:.3f}s "
        f"wall_problog={metrics.wall_seconds_problog:.3f}s"
    )

    export_jsonl(tasks, args.out)
    print(f"Exported: {args.out} | tasks={len(tasks)}")

if __name__ == "__main__":
    main()
