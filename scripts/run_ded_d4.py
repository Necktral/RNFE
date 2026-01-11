from __future__ import annotations

import argparse
from pathlib import Path

from rnfe.pmv.reasoning.ded.ded_d4_z3_fixedpoint_horn import (
    build_manifest,
    export_jsonl,
    export_manifest,
    generate_ded_d4_tasks,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--iid-ratio", type=float, default=0.75)
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--no-z3-validate", action="store_true")
    ap.add_argument("--out", type=str, default="artifacts/ded_d4.jsonl")
    args = ap.parse_args()

    tasks = generate_ded_d4_tasks(
        num_tasks=args.n,
        seed=args.seed,
        iid_ratio=args.iid_ratio,
        validate_with_z3=(not args.no_z3_validate),
        timeout_ms=args.timeout_ms,
    )

    out_path = Path(args.out)
    export_jsonl(tasks, out_path)

    manifest = build_manifest(tasks)
    export_manifest(manifest, out_path.with_suffix(".manifest.json"))

    print(f"[DED-D4] wrote: {out_path}  tasks={manifest['n_tasks']} iid={manifest['iid']} ood={manifest['ood']}")
    print(f"[DED-D4] manifest: {out_path.with_suffix('.manifest.json')}")
    print(f"[DED-D4] z3_unknown: {manifest['z3_unknown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
