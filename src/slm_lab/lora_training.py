"""Optional LoRA/QLoRA fine-tuning entrypoint. GPU and licensed model required."""
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    dataset_path: str
    output_dir: str = "artifacts/lora"
    rank: int = 8
    alpha: int = 16
    epochs: int = 1
    max_length: int = 512
    use_4bit: bool = False

    def validate(self):
        if not self.model_id or not self.dataset_path:
            raise ValueError("Model and dataset required")
        if min(self.rank, self.alpha, self.epochs, self.max_length) < 1:
            raise ValueError("Training hyperparameters must be positive")

def train(config: TrainConfig):
    config.validate()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install torch, datasets, peft and transformers to train") from exc
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    if config.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        except ImportError as exc:
            raise RuntimeError("4-bit quantization requires bitsandbytes") from exc
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config.rank, lora_alpha=config.alpha, target_modules="all-linear", task_type="CAUSAL_LM"))
    dataset = load_dataset("json", data_files=config.dataset_path, split="train")
    def tokenize(batch):
        text = [f"Instruction: {i}\nResponse: {r}" for i, r in zip(batch["instruction"], batch["response"])]
        out = tokenizer(text, truncation=True, max_length=config.max_length, padding="max_length")
        out["labels"] = [[token if token != tokenizer.pad_token_id else -100 for token in seq] for seq in out["input_ids"]]
        return out
    encoded = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    args = TrainingArguments(output_dir=config.output_dir, num_train_epochs=config.epochs, per_device_train_batch_size=1, gradient_accumulation_steps=4, logging_steps=10, save_strategy="epoch", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=encoded)
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return config.output_dir
