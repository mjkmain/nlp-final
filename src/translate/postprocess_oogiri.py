"""Post-process the JP->KO oogiri translations into an SFT-ready clean set.

The translator flags obvious non-text boke (symbols, names) with translatable=False,
but a few residual hallucinations slip through (garbled text or a bare proper noun
that the model expanded into a sentence). This script:

  1) DROPs rows that are definitely unusable (translatable=False, empty ko, error notes),
  2) sets aside a REVIEW set matching heuristic hallucination signals (kept, not deleted),
  3) writes the remaining CLEAN rows for SFT.

It also prints a distribution report with samples so thresholds can be tuned to the
actual data. Nothing is deleted in place — outputs go to new dataset dirs.

Usage:
    .venv/bin/python src/translate/postprocess_oogiri.py            # report + write clean/review
    .venv/bin/python src/translate/postprocess_oogiri.py --report-only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko"
DEFAULT_CLEAN = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko_clean"
DEFAULT_REVIEW = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko_review"

ERROR_NOTES = {"parse_failed", "empty_boke"}
# Japanese script ranges: hiragana, katakana, kanji, halfwidth katakana.
_JP_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿ｦ-ﾝ]")


def jp_char_count(s: str) -> int:
    return len(_JP_RE.findall(s or ""))


def is_error_note(note: str) -> bool:
    note = note or ""
    return note in ERROR_NOTES or note.startswith(("gen_error", "build_error"))


def classify(row: dict, min_jp: int, short_len: int) -> str:
    """Return 'drop' | 'review' | 'clean' for a translated row."""
    ko = (row.get("boke_ko") or "").strip()
    boke = (row.get("boke") or "").strip()
    note = row.get("trans_note") or ""

    if not row.get("translatable", False) or not ko or is_error_note(note):
        return "drop"

    # Residual-hallucination signals (heuristic -> review, not delete).
    # NOTE: a bare kanji name like "芦田愛菜" is NOT deterministically separable
    # from a normal short Japanese phrase, so some name->sentence cases will slip
    # through to 'clean'. The model's translatable flag is the only reliable catch.
    jp = jp_char_count(boke)
    # (a) boke carries almost no Japanese script (name / garbled / non-JP token)
    #     yet the model still produced a KO line.
    if jp < min_jp and len(boke) <= short_len:
        return "review"
    # (b) a short boke (proper-noun / fragment risk) rendered as a multi-word or
    #     much-longer KO sentence -> likely an expansion/hallucination.
    if len(boke) <= short_len and (" " in ko or len(ko) >= 2 * len(boke)):
        return "review"
    return "clean"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--clean-out", type=Path, default=DEFAULT_CLEAN)
    p.add_argument("--review-out", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--min-jp-chars", type=int, default=1, help="boke below this many JP chars is suspect.")
    p.add_argument("--short-len", type=int, default=4, help="boke at/under this length is proper-noun risk.")
    p.add_argument("--samples", type=int, default=8, help="example rows to print per bucket.")
    p.add_argument("--report-only", action="store_true", help="print report; do not write datasets.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = datasets.load_from_disk(str(args.inp))
    n = len(ds)
    labels = [classify(ds[i], args.min_jp_chars, args.short_len) for i in range(n)]

    buckets = {"clean": [], "review": [], "drop": []}
    for i, lab in enumerate(labels):
        buckets[lab].append(i)

    print(f"=== postprocess report: {args.inp} ({n} rows) ===")
    for lab in ("clean", "review", "drop"):
        idxs = buckets[lab]
        print(f"\n[{lab}] {len(idxs)} rows ({100*len(idxs)/n:.1f}%)")
        for i in idxs[:: max(1, len(idxs) // args.samples)][: args.samples]:
            r = ds[i]
            note = r.get("trans_note") or ""
            print(f"   JP={r.get('boke','')!r}  KO={r.get('boke_ko','')!r}"
                  + (f"  note={note}" if note else ""))

    if args.report_only:
        print("\n[report-only] no datasets written.")
        return

    clean = ds.select(buckets["clean"])
    review = ds.select(buckets["review"])
    clean.save_to_disk(str(args.clean_out))
    review.save_to_disk(str(args.review_out))
    print(f"\n[save] clean -> {args.clean_out}  ({len(clean)} rows)")
    print(f"[save] review -> {args.review_out}  ({len(review)} rows)")
    print("[note] 'drop' rows are simply excluded from both outputs.")


if __name__ == "__main__":
    main()
