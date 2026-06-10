"""Translate Japanese oogiri (大喜利) boke into Korean with a local VLM via vLLM.

The boke is a *response to an image* (the odai), so by default we feed the model
the odai image together with the Japanese boke and ask for a *transcreation* into
natural Korean humor (not a literal gloss). Puns that cannot survive translation
are flagged (`translatable=false`) so they can be filtered downstream.

Cached model note
-----------------
`Qwen/Qwen3.5-122B-A10B` (bf16, ~250GB) is NOT fully present in the shared cache;
only the FP8 build `Qwen/Qwen3.5-122B-A10B-FP8` (~119GB, all shards + processor) is.
We default to the FP8 build, which also fits more comfortably on 8x H100.

Usage
-----
    HF_HOME=/raid/MLP/.cache/huggingface \\
    .venv/bin/python src/translate/translate_oogiri.py --tp 8

    # text-only ablation (no image)
    .venv/bin/python src/translate/translate_oogiri.py --no-image

The run is resumable: results are appended to a JSONL checkpoint keyed by row
index, and re-running skips rows already done. When all rows are translated, an
HF dataset with the extra columns (`boke_ko`, `translatable`, `trans_note`) is
written to disk.
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
DEFAULT_INPUT = REPO_ROOT / "src" / "data" / "oogiri_cold_start_jp"
DEFAULT_OUTPUT = REPO_ROOT / "src" / "data" / "oogiri_cold_start_ko"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"

SYSTEM_PROMPT = """\
당신은 일본 오오기리(大喜利)의 보케(ボケ)를 한국어로 옮기는 전문 유머 번역가입니다.
'오다이 이미지'와 '일본어 보케'가 주어집니다.

[작업 순서]
1) 먼저 이미지를 보고 어떤 상황인지, 보케가 무엇을 빗댄 것인지 파악한다.
2) 그 상황에서 한국 사람이 실제로 툭 내뱉을 법한, 자연스러운 구어체 한 줄로 옮긴다.

[핵심 원칙]
- 직역 금지. 사전적 의미가 아니라 '웃음 포인트'와 '말맛'을 살린다.
- 기본은 반말·구어체. 짧고 리듬감 있게. 설명·부연 금지.
- 일본어 감탄사/추임새(はい, あ, え, ほら, おい 등)는 글자대로 옮기지 말고
  상황의 어조로 바꾼다. (ほら→"거봐", はい→"옳지/걸렸어", おい→"야")
- 일본어 1인칭(オイラ, ワシ, 拙者, あたし 등)·방언·인터넷 슬랭·캐릭터 말투를
  먼저 파악해 화자와 어투를 정한다.
- 말장난·동음이의·드립은 한국어 정서의 비슷한 드립으로 현지화(transcreation)한다.
- 이미지와의 연결(웃음의 근거)을 깨지 말고, 없는 내용은 지어내지 않는다.
- 한국어로 도저히 살릴 수 없는 발음 말장난이면 translatable=false,
  최선의 직역을 ko에, 이유를 note에 한 줄.

[표기 규칙 — 일본어 잔존 금지]
- ko에는 일본 문자(히라가나·가타카나·한자)를 절대 남기지 않는다. 100% 한국어로 적는다.
- 한자어는 한국어 단어로 옮긴다. (課金→과금, 優秀→우수한, A君→A군, 梯子→사다리)
- 뜻이 있는 일반 단어는 번역한다. 음차(한글로 발음 표기)는 고유명사에만 쓴다.
- 고유명사(인명·브랜드·작품명)는 한국 것으로 바꾸지 말고 일본어 발음을 한글로 음차한다. (綾鷹→아야타카)
  단 글로벌하게 알려진 것은 한국에서 쓰는 명칭으로 적는다. (マリオ→마리오, ドラえもん→도라에몽, ポケモン→포켓몬)

[욕설·수위]
- 욕설·비속어·저속/성적 표현은 검열하거나 순화하지 말고, 원문 강도 그대로 한국어 비속어로 옮긴다.
  (연구용 충실 번역이므로 톤다운하지 않는다.)

[반드시 지킬 것 — 환각 금지]
- **보케 텍스트에 실제로 있는 말만 옮긴다.** 이미지는 '뜻을 파악하는 용도'일 뿐,
  보케에 없는 문장을 이미지를 보고 지어내면 안 된다.
- 보케가 다음 중 하나뿐이면 **문장을 창작하지 말고 translatable=false** 로 표시한다:
  (a) 사람·브랜드·작품·지명 등 고유명사 하나,
  (b) 이모지·기호·마크(예: 〠, ★),
  (c) 깨진 글자·의미불명·읽을 수 없는 텍스트.
  이때 ko에는 원문 그대로(또는 인명이면 한글 음차)를 넣고, note에 이유를 적는다.

[출력] 아래 JSON 하나만. 다른 말/마크다운/코드블록 금지.
{"scene": "<이미지 상황 한 줄>", "ko": "<자연스러운 한국어 보케>", "translatable": true 또는 false, "note": "<불가 시 한 줄, 아니면 빈 문자열>"}
- scene: 이미지에서 실제로 보이는 것만 한 줄로. 확실치 않으면 무리해서 단정하지 말 것.
- ko: 위 원칙대로 옮긴 한국어 보케. translatable=false면 원문/음차를 그대로.
"""

USER_TEMPLATE = "이 이미지에 대한 일본어 보케입니다.\n원문 보케(일본어): {boke}\n\n위 규칙에 따라 JSON으로 출력하세요."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local path (default: FP8 build).")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="saved JP dataset (load_from_disk).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output dataset dir.")
    p.add_argument("--checkpoint", type=Path, default=None, help="JSONL checkpoint (default: <output>.jsonl).")
    p.add_argument("--limit", type=int, default=None, help="translate only first N rows (debug).")
    p.add_argument("--chunk-size", type=int, default=2000, help="rows per checkpoint flush.")
    p.add_argument("--no-image", action="store_true", help="text-only translation (ablation).")
    p.add_argument("--tp", type=int, default=8, help="tensor parallel size.")
    p.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization.")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28, help="Qwen-VL max image pixels (speed/quality knob).")
    p.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--gdn-prefill-backend",
        default="triton",
        help="Qwen3.5 GDN linear-attn prefill kernel. Default 'triton' avoids the "
        "FlashInfer JIT path, which needs a CUDA toolkit (nvcc) this box lacks.",
    )
    p.add_argument(
        "--allow-jit-fp8",
        action="store_true",
        help="Keep DeepGEMM FP8 kernels on. Requires nvcc; off by default here.",
    )
    return p.parse_args()


def load_input(path: Path) -> datasets.Dataset:
    """Load the saved JP dataset; fall back to the loader util if it's missing."""
    if path.exists():
        return datasets.load_from_disk(str(path))
    sys.path.insert(0, str(REPO_ROOT))
    from utils.load_oogiri_jp import load_oogiri  # noqa: E402

    print(f"[input] {path} not found -> building via utils.load_oogiri_jp.load_oogiri()")
    return load_oogiri()


def load_done_indices(ckpt: Path) -> set[int]:
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
        print(f"[resume] {len(done)} rows already in {ckpt.name}")
    return done


def extract_json(text: str) -> dict:
    """Best-effort parse of the model's JSON object."""
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(text[s : e + 1])
            return {
                "scene": str(obj.get("scene", "")).strip(),
                "ko": str(obj.get("ko", "")).strip(),
                "translatable": bool(obj.get("translatable", True)),
                "note": str(obj.get("note", "")).strip(),
            }
        except json.JSONDecodeError:
            pass
    # Fallback: keep raw output, mark as needing review.
    return {"scene": "", "ko": text.strip(), "translatable": False, "note": "parse_failed"}


def flag(note: str) -> dict:
    """A flagged (un-translated) result record."""
    return {"scene": "", "ko": "", "translatable": False, "note": note}


def write_rec(fout, idx: int, boke: str, parsed: dict) -> None:
    fout.write(json.dumps({"idx": idx, "boke": boke, **parsed}, ensure_ascii=False) + "\n")


def normalize_image(image, max_ratio: int = 180, min_side: int = 28):
    """RGB + clamp extreme aspect ratios by white-padding the short side.

    Qwen2-VL's smart_resize rejects abs aspect ratio >= 200; the bokete corpus
    has a few banner-like images (e.g. 400:1) that would crash the whole batch.
    Padding (not resizing) keeps the original content undistorted.
    """
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    long_side, short_side = max(w, h), max(min(w, h), 1)
    if long_side / short_side <= max_ratio and short_side >= min_side:
        return image

    target_short = max(short_side, min_side)
    if long_side / target_short > max_ratio:
        import math

        target_short = math.ceil(long_side / max_ratio)
    if w >= h:  # height is the short side
        new_w, new_h = w, max(target_short, min_side)
    else:  # width is the short side
        new_w, new_h = max(target_short, min_side), h
    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
    return canvas


def build_request(processor, boke: str, image, use_image: bool):
    """Return a vLLM request dict: {"prompt": str, "multi_modal_data": {...}}."""
    user_text = USER_TEMPLATE.format(boke=boke)
    if use_image and image is not None:
        content = [{"type": "image"}, {"type": "text", "text": user_text}]
    else:
        content = [{"type": "text", "text": user_text}]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    # Disable Qwen "thinking" for bulk throughput; tolerate templates without the flag.
    try:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    req: dict = {"prompt": prompt}
    if use_image and image is not None:
        req["multi_modal_data"] = {"image": normalize_image(image)}
    return req


def main() -> None:
    args = parse_args()
    use_image = not args.no_image

    # This box has no CUDA toolkit (nvcc), so JIT-compiled FP8 kernels (DeepGEMM)
    # can't build and the engine dies. Fall back to non-JIT kernels by default.
    # Must be set before vLLM is imported. (GDN linear-attn uses --gdn-prefill-backend.)
    if not args.allow_jit_fp8:
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        # FlashInfer also JIT-compiles its sampler kernel (needs nvcc); use the
        # native PyTorch sampler instead.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # bf16 (unquantized) MoE routes to FlashInfer cutlass (JIT/nvcc) unless
        # VLLM_USE_FLASHINFER_MOE_FP16 is *explicitly* set false -> Triton MoE.
        os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP16", "0")

    ckpt = args.checkpoint or args.output.with_suffix(".jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    ds = load_input(args.input)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[input] {len(ds)} rows | columns={ds.column_names} | image={'on' if use_image else 'off'}")

    done = load_done_indices(ckpt)
    todo = [i for i in range(len(ds)) if i not in done]
    if not todo:
        print("[done] every row already translated; assembling dataset.")
        finalize(ds, ckpt, args.output)
        return

    # Heavy imports after cheap exits.
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": 1} if use_image else {"image": 0},
        mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
        gdn_prefill_backend=args.gdn_prefill_backend,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    print(f"[run] translating {len(todo)} rows in chunks of {args.chunk_size}")
    with ckpt.open("a") as fout:
        for start in tqdm(range(0, len(todo), args.chunk_size), desc="chunks"):
            idxs = todo[start : start + args.chunk_size]
            batch = ds.select(idxs)
            requests, valid_idxs = [], []
            for local, idx in enumerate(idxs):
                row = batch[local]
                boke = (row.get("boke") or "").strip()
                if not boke:
                    write_rec(fout, idx, "", flag("empty_boke"))
                    continue
                try:
                    req = build_request(processor, boke, row.get("image"), use_image)
                except Exception as e:  # corrupt/odd image -> flag, don't crash
                    write_rec(fout, idx, boke, flag(f"build_error:{type(e).__name__}"))
                    continue
                requests.append(req)
                valid_idxs.append((idx, boke))

            if not requests:
                fout.flush()
                continue

            try:
                outputs = llm.generate(requests, sampling)
                for (idx, boke), out in zip(valid_idxs, outputs):
                    write_rec(fout, idx, boke, extract_json(out.outputs[0].text))
            except Exception as e:
                # One bad item shouldn't kill a 2k-row chunk: isolate per item.
                print(f"[warn] chunk generate failed ({type(e).__name__}); retrying per-item")
                for (idx, boke), req in zip(valid_idxs, requests):
                    try:
                        out = llm.generate([req], sampling)[0]
                        parsed = extract_json(out.outputs[0].text)
                    except Exception as ie:
                        parsed = flag(f"gen_error:{type(ie).__name__}")
                    write_rec(fout, idx, boke, parsed)
            fout.flush()

    finalize(ds, ckpt, args.output)


def finalize(ds: datasets.Dataset, ckpt: Path, output: Path) -> None:
    """Merge checkpoint results back onto the dataset and save_to_disk."""
    results: dict[int, dict] = {}
    with ckpt.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            results[r["idx"]] = r  # last write wins on re-runs

    n = len(ds)
    ko = [results.get(i, {}).get("ko", "") for i in range(n)]
    scene = [results.get(i, {}).get("scene", "") for i in range(n)]
    translatable = [bool(results.get(i, {}).get("translatable", False)) for i in range(n)]
    note = [results.get(i, {}).get("note", "missing") for i in range(n)]

    out = ds.add_column("boke_ko", ko)
    out = out.add_column("scene", scene)
    out = out.add_column("translatable", translatable)
    out = out.add_column("trans_note", note)
    out.save_to_disk(str(output))

    n_ok = sum(translatable)
    n_missing = sum(1 for v in note if v == "missing")
    print(
        f"[save] {output}  | rows={n}  translatable={n_ok}  "
        f"untranslatable/flagged={n - n_ok - n_missing}  missing={n_missing}"
    )


if __name__ == "__main__":
    main()
