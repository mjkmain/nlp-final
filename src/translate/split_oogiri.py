"""Split the final clean oogiri set into SFT / rollout-eval / GRPO / test buckets.

The clean set is 1 boke : 1 odai (no image repeats), so a row-level seeded shuffle
gives an odai-disjoint split with no leakage. Buckets (see the project plan):
  - test         : held-out final eval
  - rollout_eval : small set the 6 raters score -> persona in-context examples
  - sft          : shared cold-start SFT
  - grpo         : remaining prompt pool for per-persona GRPO

Usage:
    .venv/bin/python src/translate/split_oogiri.py
    .venv/bin/python src/translate/split_oogiri.py --test 2000 --rollout-eval 200 --sft 25000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "src" / "data" / "oogiri_visual_ko_clean"
OUT_DIR = REPO_ROOT / "src" / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--rollout-eval", type=int, default=200)
    p.add_argument("--grpo", type=int, default=10000, help="GRPO is compute-bound; SFT gets the rest.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-proc", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = datasets.load_from_disk(str(args.inp)).shuffle(seed=args.seed)
    n = len(ds)
    reserved = args.test + args.rollout_eval + args.grpo
    if reserved > n:
        raise SystemExit(f"clean set too small: have {n}, need {reserved} for test+rollout+grpo")

    # carve fixed buckets first; SFT (cold-start) takes the remainder.
    t, r, g = args.test, args.rollout_eval, args.grpo
    cuts = {
        "test": (0, t),
        "rollout_eval": (t, t + r),
        "grpo": (t + r, t + r + g),
        "sft": (t + r + g, n),
    }
    print(f"[split] clean={n} (seed={args.seed})")
    for name, (a, b) in cuts.items():
        sub = ds.select(range(a, b))
        out = OUT_DIR / f"oogiri_{name}"
        sub.save_to_disk(str(out), num_proc=args.num_proc)
        print(f"  {name:12}: {len(sub):6} -> {out}")


if __name__ == "__main__":
    main()
