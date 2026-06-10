"""Cold-start SFT for the Korean oogiri model (image -> funny boke).

Student: google/gemma-4-E2B-it (instruction-tuned VLM) -> we use a fixed instruction
prompt so the input format matches GRPO/inference (train == test). Loss is computed on
the assistant (boke) tokens only; the prompt + image tokens are masked.

Dataset: the SFT split (image + boke_ko columns) produced after judge + split.

Install (one-time):  uv pip install trl peft accelerate
Run (8x H100, full FT):
    HF_HOME=/raid/MLP/.cache/huggingface accelerate launch \\
        --num_processes 8 src/train/sft.py --dataset src/data/oogiri_sft

Smoke test first:  ... src/train/sft.py --dataset src/data/oogiri_sft --limit 64 --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_DATASET = REPO_ROOT / "src" / "data" / "oogiri_sft"

# Fixed task prompt — keep IDENTICAL across SFT / GRPO / inference.
PROMPT = "이 사진을 보고 빵 터지는 한 줄 보케(오오기리)를 해줘. 짧고 위트있게, 설명 없이 한 줄만."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "checkpoints" / "oogiri_sft")
    p.add_argument("--text-col", default="boke_ko")
    p.add_argument("--image-col", default="image")
    p.add_argument("--limit", type=int, default=None, help="debug: train on first N rows.")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune.")
    p.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    p.add_argument("--save-steps", type=int, default=500)
    return p.parse_args()


def build_collator(processor, max_len: int):
    """Completion-only collator: loss on the assistant (boke) tokens only.

    The processor batches the (variable-size) image features correctly, so we run it
    once on the whole batch. To mask the prompt we run the prefix-only batch too and
    use its real-token lengths (right padding -> prompt sits at the front).
    """
    tok = processor.tokenizer
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    image_token_id = getattr(processor, "image_token_id", None)

    def user_messages():
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]

    def full_messages(boke):
        return user_messages() + [
            {"role": "assistant", "content": [{"type": "text", "text": boke}]}
        ]

    def collate(examples):
        images = [ex["__image__"] for ex in examples]
        bokes = [(ex["__text__"] or "").strip() for ex in examples]
        fulls = [processor.apply_chat_template(full_messages(b), tokenize=False) for b in bokes]
        prefixes = [
            processor.apply_chat_template(user_messages(), tokenize=False, add_generation_prompt=True)
            for _ in bokes
        ]
        # gemma-4 processor wants nested images: one image-list per text sample.
        imgs = [[im] for im in images]
        batch = processor(text=fulls, images=imgs, return_tensors="pt", padding=True,
                          add_special_tokens=False)
        pref = processor(text=prefixes, images=imgs, return_tensors="pt", padding=True,
                         add_special_tokens=False)
        plens = pref["attention_mask"].sum(dim=1)  # real prompt length per example

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100        # padding
        for i, pl in enumerate(plens):
            labels[i, : int(pl)] = -100                     # prompt + image tokens
        if image_token_id is not None:
            labels[batch["input_ids"] == image_token_id] = -100
        batch["labels"] = labels
        return dict(batch)

    return collate


def main() -> None:
    args = parse_args()
    from transformers import Trainer, TrainingArguments

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    )

    ds = datasets.load_from_disk(str(args.dataset))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    # rename to stable internal keys; keep image decoded as PIL
    ds = ds.rename_columns({args.text_col: "__text__", args.image_col: "__image__"})
    keep = {"__text__", "__image__"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    print(f"[sft] {len(ds)} examples | model={args.model} | mode={'LoRA' if args.lora else 'full'}")

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # gemma-4 E2B is omni (image+video+audio); image-only training leaves the
        # video/audio towers without grads -> DDP needs this to not error.
        ddp_find_unused_parameters=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=build_collator(processor, args.max_len),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(str(args.output))
    print(f"[done] saved -> {args.output}")


if __name__ == "__main__":
    main()
