# Korean Oogiri — Personalized Humor LLM

Making a VLM funny at **Korean oogiri (大喜利)** — given an image (odai), generate a
witty one-line *boke*. Final goal: **6 persona-personalized models** via SFT → rollout
→ human eval → in-context reward → per-persona GRPO.

> GCT799 (NLP) final project.

## Pipeline

```
bokete (JP) ──► sample top-star (200k)
            ──► image filter (keep "visual" only; drop text-only / JP-text-in-image)
            ──► translate JP→KO  (Qwen3.5-122B-A10B, image-grounded transcreation)
            ──► judge (1–5 fidelity + culture_block + leftover-JP rule) → clean
            ──► split: sft / grpo / rollout_eval / test
            ──► SFT cold-start (gemma-4-E2B-it) → rollout → reward → GRPO ×6
```

Final clean set: **76,683** (image, `boke` JP, `boke_ko`, `scene`, quality flags).

## Layout

| path | what |
|---|---|
| `utils/load_oogiri_jp.py` | sample top-star JP boke from bokete |
| `src/translate/classify_image_oogiri.py` | label image: text_only / text_in_image / visual |
| `src/translate/translate_oogiri.py` | JP→KO transcreation (vLLM, image-grounded) |
| `src/translate/judge_oogiri.py` | LLM-judge quality score + culture_block |
| `src/translate/split_oogiri.py` | split into sft/grpo/rollout_eval/test |
| `src/train/sft.py` | cold-start SFT (gemma-4-E2B-it) |

Generated data (`src/data/`), checkpoints, and `.venv/` are git-ignored.

## Setup

```bash
uv sync                       # recreate env from uv.lock
export HF_HOME=/path/to/hf_cache
```

## Run

```bash
# data pipeline (each resumable via JSONL checkpoint)
.venv/bin/python utils/load_oogiri_jp.py
.venv/bin/python src/translate/classify_image_oogiri.py --tp 8
.venv/bin/python src/translate/translate_oogiri.py --tp 8 --model Qwen/Qwen3.5-122B-A10B --input src/data/oogiri_visual
.venv/bin/python src/translate/judge_oogiri.py --tp 8 --threshold 4
.venv/bin/python src/translate/split_oogiri.py

# SFT (use the venv's accelerate, not the system one)
.venv/bin/accelerate launch --num_processes 8 src/train/sft.py --dataset src/data/oogiri_sft
```

## Notes

- vLLM on this box needs no-JIT kernel flags (no CUDA toolkit / nvcc) — baked into the
  scripts (DeepGEMM off, FlashInfer sampler/MoE-fp16 off, GDN prefill = triton).
- Source data is from bokete; respect its terms before redistributing.
