"""LLM-as-judge quality filter for the JP->KO oogiri translations (Qwen3.5-122B).

Feeds ONLY the two texts (Japanese boke + Korean translation) — no image — and asks
the model to score 1-5 whether the Korean conveys the *meaning and intent* of the
Japanese. This is a fidelity-with-localization judge, NOT a literal-translation judge:
good transcreation should score high; content unrelated to / invented from the source
should score low. That directly catches the residual hallucinations (a bare name or
garbled token rendered as a fluent KO sentence) that deterministic rules cannot.

Pipeline position:  translate_oogiri.py  ->  judge_oogiri.py  ->  filter at threshold.
Rows the translator already flagged (translatable=False) are auto-scored 0 (dropped),
so we only spend judge compute on the 47k candidate rows.

Text-only, so it runs much faster than the translation pass. Resumable via JSONL
checkpoint keyed by row index; writes a scored dataset plus a clean subset.

Usage:
    .venv/bin/python src/translate/judge_oogiri.py --tp 8 --threshold 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import datasets
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko"
DEFAULT_SCORED = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko_scored"
DEFAULT_CLEAN = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko_clean"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"

# kana (hiragana/katakana, incl. halfwidth) + kanji -> leftover Japanese in KO output
_JP_RE = re.compile(r"[぀-ゟ゠-ヿ一-鿿ｦ-ﾟ]")

SYSTEM_PROMPT = """\
당신은 일본어→한국어 번역 품질 평가자입니다.
'일본어 원문(오오기리 보케)'과 그에 대한 '한국어 번역'이 주어집니다.
한국어가 일본어의 '의미와 의도'를 제대로 옮겼는지 1~5점으로 평가하세요.

[중요] 이건 직역 충실도 평가가 아니다. 유머 현지화(의역/transcreation)는 허용된다.
표현이 달라도 의미·의도·웃음 포인트가 통하면 높은 점수를 준다.

[채점 기준]
- 5: 의미와 뉘앙스를 정확히 전달. 자연스러운 한국어.
- 4: 의미는 정확. 약간의 뉘앙스·말맛 손실.
- 3: 대체로 통하나 일부 의미 누락이나 어색함.
- 2: 의미가 상당히 어긋남. 원문과 부분적으로만 관련.
- 1: 원문과 무관하거나 원문에 없는 내용을 지어냄(번역이 아님).

[특히 주의]
- 한국어가 원문과 무관한 문장을 '창작'했으면 1점.
- 원문이 고유명사·기호뿐인데 한국어가 멀쩡한 문장이면 1~2점.
- 번역이 원문의 욕설·비속어를 순화/약화했으면 의도 손실로 보고 감점한다.

[culture_block 판정 — 점수와 독립]
- true: 보케의 웃음 포인트가 '한국인이 모를 일본 한정 고유명사'(일본 연예인·일본 한정/지역
  브랜드·일본 한정 작품/광고 등)를 알아야만 성립한다.
- false: 고유명사 의존이 없거나, 글로벌하게 유명해 한국에도 통하는 것
  (마리오·도라에몽·포켓몬·드래곤볼 등)에만 의존한다.
- 주의: 번역이 충실해 점수가 높아도, 한국 전이가 안 되면 culture_block=true일 수 있다.
  점수와 별개로 판단한다.

반드시 아래 JSON 하나만 출력한다. 다른 말/마크다운 금지.
{"score": 1~5 정수, "culture_block": true 또는 false, "reason": "<한 줄 근거>"}
"""

USER_TEMPLATE = "일본어 원문: {boke}\n한국어 번역: {ko}\n\n점수를 매기세요."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--scored-out", type=Path, default=DEFAULT_SCORED)
    p.add_argument("--clean-out", type=Path, default=DEFAULT_CLEAN)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--threshold", type=int, default=4, help="keep rows with score >= this in clean set.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=4000)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic judging.")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--gdn-prefill-backend", default="triton")
    p.add_argument("--allow-jit-fp8", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_done(ckpt: Path) -> set[int]:
    done: set[int] = set()
    if ckpt.exists():
        with ckpt.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["idx"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"[resume] {len(done)} rows already judged in {ckpt.name}")
    return done


def parse_score(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            sc = int(obj.get("score"))
            sc = min(5, max(1, sc))
            return {
                "score": sc,
                "culture_block": bool(obj.get("culture_block", False)),
                "reason": str(obj.get("reason", "")).strip(),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return {"score": 0, "culture_block": False, "reason": "parse_failed"}


def build_prompt(processor, boke: str, ko: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(boke=boke, ko=ko)},
    ]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt}


def main() -> None:
    args = parse_args()
    if not args.allow_jit_fp8:
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    ckpt = args.checkpoint or args.scored_out.with_suffix(".jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_from_disk(str(args.inp))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"[input] {n} rows | columns={ds.column_names}")

    # Only judge rows the translator accepted; flagged ones are auto-dropped (score 0).
    candidates = [i for i in range(n) if ds[i].get("translatable", False)]
    print(f"[plan] judging {len(candidates)} translatable rows; "
          f"{n - len(candidates)} flagged rows auto-scored 0")

    done = load_done(ckpt)
    todo = [i for i in candidates if i not in done]

    if todo:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            limit_mm_per_prompt={"image": 0},
            gdn_prefill_backend=args.gdn_prefill_backend,
        )
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        print(f"[run] judging {len(todo)} rows in chunks of {args.chunk_size}")
        with ckpt.open("a") as fout:
            for start in tqdm(range(0, len(todo), args.chunk_size), desc="chunks"):
                idxs = todo[start : start + args.chunk_size]
                batch = ds.select(idxs)
                prompts = [
                    build_prompt(processor, batch[k].get("boke", ""), batch[k].get("boke_ko", ""))
                    for k in range(len(idxs))
                ]
                try:
                    outputs = llm.generate(prompts, sampling)
                    for idx, out in zip(idxs, outputs):
                        rec = {"idx": idx, **parse_score(out.outputs[0].text)}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[warn] chunk failed ({type(e).__name__}); retrying per-item")
                    for idx, pr in zip(idxs, prompts):
                        try:
                            out = llm.generate([pr], sampling)[0]
                            rec = {"idx": idx, **parse_score(out.outputs[0].text)}
                        except Exception as ie:
                            rec = {"idx": idx, "score": 0, "culture_block": False,
                                   "reason": f"gen_error:{type(ie).__name__}"}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
    else:
        print("[done] all candidates already judged; assembling outputs.")

    finalize(ds, ckpt, args)


def finalize(ds: datasets.Dataset, ckpt: Path, args: argparse.Namespace) -> None:
    scores: dict[int, dict] = {}
    with ckpt.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores[r["idx"]] = r

    n = len(ds)
    score_col = [int(scores.get(i, {}).get("score", 0)) for i in range(n)]
    block_col = [bool(scores.get(i, {}).get("culture_block", False)) for i in range(n)]
    reason_col = [scores.get(i, {}).get("reason", "flagged_by_translator") for i in range(n)]
    # deterministic safety rule: any kana/kanji left in the Korean output.
    jp_col = [bool(_JP_RE.search(ds[i].get("boke_ko") or "")) for i in range(n)]

    out = ds.add_column("quality_score", score_col)
    out = out.add_column("culture_block", block_col)
    out = out.add_column("ko_has_jp", jp_col)
    out = out.add_column("judge_reason", reason_col)
    out.save_to_disk(str(args.scored_out))

    # final clean = high fidelity AND transfers to Korean AND no leftover Japanese
    clean = out.filter(
        lambda r: r["quality_score"] >= args.threshold
        and not r["culture_block"]
        and not r["ko_has_jp"]
    )
    clean.save_to_disk(str(args.clean_out))

    dist = {s: score_col.count(s) for s in range(6)}
    print("\n=== score distribution ===")
    for s in range(6):
        tag = " (flagged/parse-fail)" if s == 0 else ""
        print(f"  score {s}: {dist[s]:6}  ({100*dist[s]/n:.1f}%){tag}")
    n_block = sum(block_col)
    n_jp = sum(jp_col)
    print(f"\nculture_block=true : {n_block} ({100*n_block/n:.1f}%)")
    print(f"ko_has_jp=true     : {n_jp} ({100*n_jp/n:.1f}%)")
    print(f"\n[save] scored -> {args.scored_out}  ({n} rows, +quality_score +culture_block +ko_has_jp +judge_reason)")
    print(f"[save] clean (score>={args.threshold} & not culture_block & not ko_has_jp) "
          f"-> {args.clean_out}  ({len(clean)} rows, {100*len(clean)/n:.1f}%)")


if __name__ == "__main__":
    main()
