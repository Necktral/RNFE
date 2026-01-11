from __future__ import annotations

import argparse
from rnfe.pmv.reasoning.deductive_d1 import (
    generate_ded_d1_tasks,
    evaluate_tasks,
    export_jsonl,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_kb", type=int, default=200)
    ap.add_argument("--queries_per_kb", type=int, default=6)
    ap.add_argument("--out", type=str, default="ded_d1.jsonl")
    args = ap.parse_args()

    all_tasks = []
    for difficulty in ["easy", "mid", "hard"]:
        for split in ["id", "ood"]:
            tasks = generate_ded_d1_tasks(
                seed=args.seed + hash((difficulty, split)) % 10_000,
                n_kb=args.n_kb,
                queries_per_kb=args.queries_per_kb,
                split=split,
                difficulty=difficulty,
            )
            metrics = evaluate_tasks(tasks)
            print(f"[{difficulty}/{split}] {metrics}")
            all_tasks.extend(tasks)

    export_jsonl(all_tasks, args.out)
    print(f"Exported: {args.out} | tasks={len(all_tasks)}")

if __name__ == "__main__":
    main()
