"""Classify each odai image so text-dominant images can be filtered out.

Many bokete odai are not visual prompts at all: a Japanese question printed on a
card (text_only), or a manga panel / screenshot with Japanese text baked into it
(text_in_image). For a Korean image-based oogiri model these are noise — the joke
needs the viewer to *read Japanese*, which doesn't transfer.

This labels each image (image-only, the model does NOT see the boke) into:
  - text_only      : essentially just printed text, no real visual subject
  - text_in_image  : has a visual subject but Japanese text is baked in
  - visual         : photo/illustration, no or negligible text

Adds an `image_type` (+ `image_type_reason`) column. The drop policy is applied
later at final assembly, so we just label here. Resumable JSONL checkpoint by idx.

Usage:
    .venv/bin/python src/translate/classify_image_oogiri.py --tp 8
    .venv/bin/python src/translate/classify_image_oogiri.py --limit 8 \
        --out /tmp/imgcls --checkpoint /tmp/imgcls.jsonl   # quick validation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import datasets
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate_oogiri import normalize_image  # noqa: E402  (reuse aspect-ratio guard)

DEFAULT_IN = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko_scored"
DEFAULT_OUT = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko_imgcls"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"

LABELS = ("text_only", "text_in_image", "visual")

SYSTEM_PROMPT = """\
당신은 일본 오오기리(大喜利) 오다이(お題) 이미지를 분류하는 분류기입니다.
주어진 이미지 하나를 아래 세 가지 중 정확히 하나로 분류하세요.

- "text_only": 실제 시각적 피사체가 없고 글자/문장이 이미지의 거의 전부다.
  예) 일본어 질문이 적힌 카드, 자막만 있는 화면, 글씨만 있는 짤.
- "text_in_image": 사진·그림 등 시각적 피사체는 있으나, 일본어 글자가 이미지 안에 박혀 있다.
  예) 만화 말풍선/대사, 간판·자막·캡션이 들어간 짤, 스크린샷.
- "visual": 사진·일러스트 등 시각 콘텐츠가 중심이고, 글자가 없거나 무시할 수준이다.

핵심 기준: 한국어 사용자가 일본어 글자를 못 읽는다고 할 때,
- 글자를 못 읽으면 이해 불가능하면 -> text_only 또는 text_in_image
- 그림만으로 충분히 이해되면 -> visual

반드시 아래 JSON 하나만 출력한다. 다른 말/마크다운 금지.
{"image_type": "text_only" 또는 "text_in_image" 또는 "visual", "reason": "<한 줄 근거>"}
"""

USER_TEXT = "이 오다이 이미지를 분류하세요."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--temperature", type=float, default=0.0)
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
        print(f"[resume] {len(done)} images already classified in {ckpt.name}")
    return done


def parse_label(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            t = str(obj.get("image_type", "")).strip()
            if t in LABELS:
                return {"image_type": t, "reason": str(obj.get("reason", "")).strip()}
        except json.JSONDecodeError:
            pass
    # default unknown -> treat as visual-unsure but flag in reason for review
    return {"image_type": "unknown", "reason": "parse_failed"}


def build_request(processor, image):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_TEXT}]},
    ]
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "multi_modal_data": {"image": normalize_image(image)}}


def main() -> None:
    args = parse_args()
    if not args.allow_jit_fp8:
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    ckpt = args.checkpoint or args.out.with_suffix(".jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    ds = datasets.load_from_disk(str(args.inp))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"[input] {n} images from {args.inp}")

    done = load_done(ckpt)
    todo = [i for i in range(n) if i not in done]

    if todo:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        )
        llm = LLM(
            model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len, trust_remote_code=args.trust_remote_code,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
            gdn_prefill_backend=args.gdn_prefill_backend,
        )
        sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        print(f"[run] classifying {len(todo)} images in chunks of {args.chunk_size}")
        with ckpt.open("a") as fout:
            for start in tqdm(range(0, len(todo), args.chunk_size), desc="chunks"):
                idxs = todo[start : start + args.chunk_size]
                batch = ds.select(idxs)
                reqs, valid = [], []
                for k, idx in enumerate(idxs):
                    try:
                        reqs.append(build_request(processor, batch[k]["image"]))
                        valid.append(idx)
                    except Exception as e:
                        fout.write(json.dumps({"idx": idx, "image_type": "unknown",
                                               "reason": f"build_error:{type(e).__name__}"}, ensure_ascii=False) + "\n")
                if not reqs:
                    fout.flush(); continue
                try:
                    outs = llm.generate(reqs, sampling)
                    for idx, o in zip(valid, outs):
                        fout.write(json.dumps({"idx": idx, **parse_label(o.outputs[0].text)}, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[warn] chunk failed ({type(e).__name__}); per-item retry")
                    for idx, req in zip(valid, reqs):
                        try:
                            o = llm.generate([req], sampling)[0]
                            rec = {"idx": idx, **parse_label(o.outputs[0].text)}
                        except Exception as ie:
                            rec = {"idx": idx, "image_type": "unknown", "reason": f"gen_error:{type(ie).__name__}"}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
    else:
        print("[done] all images already classified; assembling.")

    finalize(ds, ckpt, args.out)


def finalize(ds: datasets.Dataset, ckpt: Path, out: Path) -> None:
    res: dict[int, dict] = {}
    with ckpt.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            res[r["idx"]] = r

    n = len(ds)
    types = [res.get(i, {}).get("image_type", "unknown") for i in range(n)]
    reasons = [res.get(i, {}).get("reason", "missing") for i in range(n)]
    out_ds = ds.add_column("image_type", types).add_column("image_type_reason", reasons)
    out_ds.save_to_disk(str(out))

    from collections import Counter
    dist = Counter(types)
    print("\n=== image_type distribution ===")
    for t in (*LABELS, "unknown"):
        print(f"  {t:14}: {dist.get(t,0):6}  ({100*dist.get(t,0)/n:.1f}%)")
    print(f"\n[save] {out}  ({n} rows, +image_type, +image_type_reason)")


if __name__ == "__main__":
    main()
