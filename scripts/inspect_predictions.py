import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generate import Text2SQLInferenceEngine
from src.sql.safety import validate_sql_safety

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "base" / "epoch_13.pt"
TOKENIZER = PROJECT_ROOT / "checkpoints" / "tokenizer"
TEST = PROJECT_ROOT / "data" / "splits" / "test.jsonl"

engine = Text2SQLInferenceEngine(CHECKPOINT, TOKENIZER)

examples = []
with TEST.open(encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

exact = []
valid_wrong = []
invalid = []

for i, rec in enumerate(examples):
    predicted = engine.generate_sql(
        rec["question"],
        rec["schema"],
        max_new_tokens=128,
    )

    gold = rec["sql"]
    safety = validate_sql_safety(predicted)

    norm_gold = " ".join(gold.split()).lower()
    norm_pred = " ".join(predicted.split()).lower()

    item = {
        "question": rec["question"],
        "gold": gold,
        "predicted": predicted,
    }

    if norm_gold == norm_pred:
        exact.append(item)
    elif safety.is_safe:
        valid_wrong.append(item)
    else:
        invalid.append(item)

    if len(exact) >= 3 and len(valid_wrong) >= 3 and len(invalid) >= 3:
        break

def show(title, items):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for n, x in enumerate(items[:3], 1):
        print(f"\n--- Example {n} ---")
        print("QUESTION:", x["question"])
        print("GOLD:    ", x["gold"])
        print("PREDICTED:", x["predicted"])

show("EXACT MATCHES", exact)
show("VALID BUT WRONG", valid_wrong)
show("INVALID / UNSAFE", invalid)

print("\nCounts found:")
print("Exact matches:", len(exact))
print("Valid but wrong:", len(valid_wrong))
print("Invalid/unsafe:", len(invalid))
