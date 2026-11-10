"""Local, opt-in LoRA supervised fine-tuning on validated instruction JSONL.

Install: pip install -e '.[train]'
Run: python -m slm_lab.train_lora --dataset data.jsonl --model sshleifer/tiny-gpt2 --output ./artifacts/lora
This is a reference training pipeline, not a claim of GPU production readiness.
"""
import argparse
import json
from pathlib import Path
from .core import load_jsonl, split


def train(*, dataset: Path, model_name: str, output: Path, epochs: float = 1,
          max_length: int = 256, seed: int = 42, batch_size: int = 1) -> dict:
    if not 1 <= max_length <= 4096 or not 1 <= batch_size <= 32 or not 0 < epochs <= 20:
        raise ValueError("Invalid bounded training configuration")
    records = load_jsonl(dataset)
    training, validation = split(records, seed=seed)
    # Heavy dependencies are optional and never loaded by core unit tests.
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForLanguageModeling)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    ))
    def prepare(rows):
        prompts = [f"Instruction:\n{r['instruction']}\nResponse:\n{r['response']}{tokenizer.eos_token}" for r in rows]
        return Dataset.from_dict({"text": prompts}).map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=["text"])
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        save_strategy="epoch", eval_strategy="epoch", logging_steps=1,
        report_to="none", seed=seed, fp16=False, bf16=False,
    )
    trainer = Trainer(model=model, args=args,
        train_dataset=prepare(training), eval_dataset=prepare(validation),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    adapter_path = output / "adapter"
    model.save_pretrained(str(adapter_path), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_path))
    result = {"base_model": model_name, "train_records": len(training),
              "validation_records": len(validation), "seed": seed,
              "adapter_path": str(adapter_path), "metrics": trainer.evaluate()}
    (output / "manifest.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train a local LoRA adapter on instruction JSONL")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(train(dataset=args.dataset, model_name=args.model,
                           output=args.output, epochs=args.epochs,
                           max_length=args.max_length, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
