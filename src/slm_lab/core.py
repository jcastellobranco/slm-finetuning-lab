from hashlib import sha256
import json
from pathlib import Path
import random

REQUIRED = {"instruction", "response"}

def validate(records: list[dict]) -> list[dict]:
    seen = set()
    cleaned = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not REQUIRED.issubset(record):
            raise ValueError(f"Invalid record at {index}")
        instruction, response = record["instruction"], record["response"]
        if not isinstance(instruction, str) or not isinstance(response, str) or not instruction.strip() or not response.strip():
            raise ValueError(f"Empty or non-string field at {index}")
        key = sha256(instruction.strip().lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"instruction": instruction.strip(), "response": response.strip()})
    return cleaned

def split(records: list[dict], test_ratio: float = 0.2, seed: int = 42):
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records required")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, min(len(shuffled)-1, round(len(shuffled)*test_ratio)))
    return shuffled[test_size:], shuffled[:test_size]

def exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions or len(predictions) != len(references):
        raise ValueError("Nonempty equal-length sequences required")
    return sum(a.strip().lower() == b.strip().lower() for a, b in zip(predictions, references)) / len(references)

def load_jsonl(path: Path):
    return validate([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
