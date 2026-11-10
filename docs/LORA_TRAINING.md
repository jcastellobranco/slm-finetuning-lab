# Local LoRA fine-tuning

This is an **opt-in** training implementation; no hosted model API or AWS deployment is required. It downloads the explicitly named model from the model registry, so review its license and provenance before running.

```bash
python -m pip install -e '.[train]'
python -m slm_lab.train_lora --dataset examples/instructions.jsonl --model sshleifer/tiny-gpt2 --output artifacts/demo
```

The dataset is JSONL with nonempty `instruction` and `response` fields. At least two distinct instructions are required; duplicates are removed before deterministic train/evaluation splitting. Training writes a PEFT LoRA adapter and `manifest.json` under the output directory. Do not commit model weights, private datasets or artifacts. The tiny model is for smoke execution, not meaningful instruction quality.

Limits: this is standard LoRA, **not QLoRA**; no quantization, distributed training, GPU cost management, benchmark suite or serving integration. For serious quality use a compatible instruction-tuned causal LM, a curated dataset and held-out evaluation. Some model architectures require an explicit LoRA target-module configuration.
