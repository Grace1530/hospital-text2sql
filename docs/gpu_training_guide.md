# GPU Training Guide (NVIDIA machine)

This project was developed on a CPU-only Windows laptop. All code is
device-agnostic (`src/utils/device.py` auto-selects CUDA if available,
CPU otherwise) -- nothing needs to change to run on a GPU machine except
installing a CUDA-enabled PyTorch build.

## 1. Copy the project

Copy the entire project directory to the NVIDIA machine (or `git clone` /
`git pull` if using a shared remote). Do NOT copy `.venv/` -- rebuild it
fresh on the target machine (different OS/CUDA toolchain).

Do copy (or regenerate if missing/excluded from git):
- `data/raw/Hospital_Management_System.sql` (source of truth, small)
- `data/raw/external/` (or re-run `scripts/download_general_dataset.py`)
- `database/hospital.duckdb` (or regenerate with `scripts/build_database.py`)
- `data/splits/*.jsonl` (or regenerate with `scripts/build_training_corpus.py`)
- `checkpoints/tokenizer/` (our trained BPE vocab -- reruns are
  reproducible but there is no reason to retrain the tokenizer on a GPU
  machine; it is a CPU-bound text-processing step)

## 2. Create the environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build BEFORE the rest of requirements.txt
(pick the command matching the machine's installed CUDA toolkit version
from https://pytorch.org/get-started/locally/, e.g. for CUDA 12.1):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 3. Verify CUDA is actually detected

```powershell
python -c "from src.utils.device import device_report; import json; print(json.dumps(device_report(), indent=2))"
```

Expected: `"cuda_available": true` and a real `"cuda_device_name"`. If this
still reports `cpu`, STOP and fix the PyTorch/CUDA install before training
-- do not proceed and do not claim GPU training happened.

## 4. (Re)build data artifacts if not copied over

```powershell
python scripts/build_database.py
python scripts/extract_schema.py
python scripts/generate_hospital_dataset.py
python scripts/download_general_dataset.py
python scripts/process_general_dataset.py
python scripts/build_training_corpus.py
```

(Skip `train_tokenizer.py` if `checkpoints/tokenizer/` was copied over --
retraining it would change the vocabulary and invalidate consistency with
any existing checkpoints. Only retrain the tokenizer if you are
deliberately starting a fresh model/vocabulary.)

## 5. Run the real training job

```powershell
python -m src.training.train --config configs/base.yaml
```

This is the SAME code path already verified end-to-end on the CPU
(pipeline-verification runs, see HANDOFF_REPORT.md) -- only the device and
the number of epochs/data scale change. Adjust `configs/base.yaml` first
if you want a different model size, batch size, or epoch count (bigger GPU
memory allows a larger `batch_size` and/or `d_model`/layer counts than
were practical on the CPU dev machine).

To resume an interrupted run:

```powershell
python -m src.training.train --config configs/base.yaml --resume checkpoints/base/last.pt
```

## 6. Expected outputs

- `checkpoints/base/epoch_N.pt` -- one checkpoint per epoch
- `checkpoints/base/last.pt` -- most recent checkpoint (used for resume and
  for inference/the web app by default)
- `checkpoints/base/train_log.jsonl` -- structured per-step/per-epoch logs
  (loss, learning rate, epoch time, device)

## 7. Evaluate the trained model

```powershell
python scripts/run_evaluation.py --checkpoint checkpoints/base/last.pt --split data/splits/test.jsonl
```

Produces `docs/evaluation_report.md` / `.json` with exact-match, execution
accuracy, valid-SQL rate, table/column/join/aggregation/filter accuracy,
and syntax-error rate, broken down by difficulty (see
`src/evaluation/evaluate.py`).

## 8. Point the web app at the trained checkpoint

```powershell
$env:T2SQL_CHECKPOINT = "checkpoints/base/last.pt"
uvicorn web.app:app --reload
```

## What NOT to do

- Do not retrain the tokenizer after training a model against the current
  one -- the vocabulary/token ids must stay fixed for a given checkpoint.
- Do not skip step 3 -- silently falling back to CPU on a "GPU machine"
  and training for hours would waste the whole point of the GPU handoff.
