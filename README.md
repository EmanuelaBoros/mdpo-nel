# NEL Retrieval, Selection, and MDPO Fine-Tuning

This repository contains the main code for a two-phase Named Entity Linking
(NEL) pipeline for historical newspaper text, plus the fine-tuning code for the
phase-2 selector:

1. **Retrieval**: generate normalized entity-name candidates with an LLM, then
   retrieve matching Wikidata QIDs (optionally promoting QIDs from a train-set
   alias dictionary).
2. **Selection**: choose the best Wikidata QID from the retrieved candidates.
3. **Fine-tuning** (supports phase 2): build preference datasets and train the
   selector with multi-negative DPO / MDPO-style objectives.

The code was refactored from exploratory notebooks into reusable Python modules
and command-line entry points.

For end-to-end environment, data, checkpoint, MXFP4 export, and scorer details,
see `REPRODUCIBILITY.md`.

## Layout

```text
nel_mdpo/
  data.py                 HIPE TSV loading, entity extraction, JSONL/TSV writing
  prompts.py              Shared retrieval and selection prompts
  wikidata.py             Async Wikidata search and entity lookup helpers
  alias.py                Alias dictionary building and alias-QID promotion
  retrieval.py            Phase 1 candidate retrieval pipeline
  postprocess.py          Re-apply alias promotion to existing retrieval JSONL
  selection.py            Phase 2 candidate selection pipeline
  dpo_data.py             Preference dataset builder and MDPO format converter
  trainers.py             Multi-negative DPO and MDPO trainers/collators
  train_mdpo.py           Fine-tuning CLI
scripts/
  retrieve_candidates.py  Run retrieval
  postprocess_retrieval.py Normalize/deduplicate retrievals and promote aliases
  select_entities.py      Run selection
  build_dpo_data.py       Build preference data from HIPE TSV
legacy_newpaper/          Original experiment scripts and alias dictionary
data/retrieval_gpt_oss_120b_256/  Committed segment-256 retrieval files
results/                  Prediction TSVs, HIPE scorer outputs, MANIFEST.json
requirements.txt          Pins for training/export/evaluation environment
requirements-serve.txt    Pins for the vLLM serving environment
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

The pinned package versions observed in the local `IE-2025` training/export
environment are recorded in `requirements.txt`. That environment was used for
the HIPE scorer checks and Unsloth MXFP4 export workflow.

```bash
pip install -r requirements.txt
```

**External dependency, not installable from this repo:** anything that reads or
writes HIPE TSV files (`--hipe-root` alias building, `select_entities.py`,
`build_dpo_data.py`) imports `hipe_commons.helpers.tsv`, which comes from the
HIPE tooling and is not packaged here. Put it on `PYTHONPATH` first. The
HIPE-2022 data (`../HIPE-2022-data`) and scorer (`../HIPE-scorer`) are also
expected as sibling checkouts.

### Model servers

The retrieval, selection, and DPO-data CLIs call an OpenAI-compatible server
(`--base-url`, default `http://0.0.0.0:8007/v1`; `--model` must match the
server's served model name). Serving uses a separate environment whose pins are
in `requirements-serve.txt` (includes `vllm==0.19.0`).

Retrieval and DPO-data construction use the **GPT-OSS-120B** teacher
(`openai/gpt-oss-120b`). Selection with the released fine-tuned model uses the
**GPT-OSS-20B MDPO** model, so run it as a second server (different port) or
swap servers between phases:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4 \
  --served-model-name GPT-OSS-20B-MDPO-NEL-3003-2000-mxfp4 \
  --host 0.0.0.0 \
  --port 8007
```

## Phase 1: Retrieval

Input JSONL records need `input_candidate`, `context`, `gold_qid`,
`abs_token_start` and `abs_token_end`. Optional fields: `summary` (used instead
of `context` when present) and `period` (year hint for the LLM).

```json
{"input_candidate": "Paris", "context": "...", "gold_qid": "Q90", "abs_token_start": 10, "abs_token_end": 11}
```

Run:

```bash
python scripts/retrieve_candidates.py \
  --input-file <input>.jsonl \
  --output-dir runs/retrieval \
  --dataset-name hipe2020 \
  --language fr \
  --model openai/gpt-oss-120b \
  --base-url http://0.0.0.0:8007/v1 \
  --max-candidates 8 \
  --alias-dict legacy_newpaper/alias_dictionary_multilingual.json \
  --hipe-root ../HIPE-2022-data/data/v2.1
```

Notes:

- The output is
  `<output-dir>/<model-slug>-<dataset-name>-<language>-retrieved.jsonl`
  (`openai/` is stripped from the model name and `/` becomes `-`, so
  `gpt-oss-120b-hipe2020-fr-retrieved.jsonl` here).
- Each output record keeps the input fields and adds `retrievals` (candidate
  QIDs, alias hits first), `alias_fired` and `is_correct` (gold is in
  `retrievals`; for `NIL`, `retrievals` is empty).
- Wikidata is searched in `--language` plus English unless `--search-languages`
  (comma-separated) is given. `--dataset-name` and `--language` also name the
  output file.
- Progress is saved in a `.state.json` next to the output and records are
  appended, so a rerun resumes. Delete both files to start over.
- `--alias-dict`: if the file exists it is loaded; if it does not, it is built
  from the HIPE train TSVs under `--hipe-root` and written to that path.
  Omitting `--alias-dict` disables alias promotion. The committed dictionary is
  `legacy_newpaper/alias_dictionary_multilingual.json`.

Alias promotion can also be applied separately to an existing retrieval file:

```bash
python scripts/postprocess_retrieval.py \
  --input-file runs/retrieval/gpt-oss-120b-hipe2020-fr-retrieved.jsonl \
  --output-file runs/retrieval/gpt-oss-120b-hipe2020-fr-256_alias.jsonl \
  --alias-dict legacy_newpaper/alias_dictionary_multilingual.json \
  --hipe-root ../HIPE-2022-data/data/v2.1
```

This rewrites `retrievals` as a deduplicated QID list and prints
`{records, alias_hits, correct}`. Its `--max-candidates` default is 10 (retrieval
uses 8).

## Phase 2: Selection

Run selection over a retrieval file and write token-level predictions. The
retrieval JSONL files under `data/retrieval_gpt_oss_120b_256/` can be used
directly as `--input-file`.

```bash
python scripts/select_entities.py \
  --input-file runs/retrieval/gpt-oss-120b-hipe2020-fr-retrieved.jsonl \
  --output-dir runs/selection \
  --tsv-path ../HIPE-2022-data/data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv \
  --language fr \
  --model GPT-OSS-20B-MDPO-NEL-3003-2000-mxfp4 \
  --base-url http://0.0.0.0:8007/v1 \
  --run-tag hipe2020-fr
```

`--model` is required and must be the served model name of the selector.
Outputs in `--output-dir`, named with `--run-tag` (default `selection`):

- `<run-tag>.jsonl`: per-mention prediction, ground truth and candidates
- `<run-tag>-flat.json`: token-level predictions and golds
- `<run-tag>-predictions.tsv`: the HIPE TSV with the `NEL-LIT` column replaced
  (`NIL` is written as `_`); score this with the HIPE scorer

Mentions with no retrieved candidates are predicted `NIL` without calling the
model.
Mentions are processed concurrently; `--agent-concurrency` bounds simultaneous
model calls and Wikidata requests.

## Build Preference Data

```bash
python scripts/build_dpo_data.py \
  --tsv-path ../HIPE-2022-data/data/v2.1/hipe2020/fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv \
  --output-file data/dpo_hipe2020_fr_train.jsonl \
  --language fr \
  --model openai/gpt-oss-120b \
  --base-url http://0.0.0.0:8007/v1 \
  --chunk-size 256 \
  --max-negatives 3
```

Each output row has `scenario` (`standard` or `nil`), `prompt`, `chosen` and a
list of `rejected` answers. Negatives are the retriever's non-gold candidates.
Candidates are written as `QID:description`, with the label and description
fetched from Wikidata in `--language` (falling back to English, then to the QID).

## Train MDPO / Multi-Negative DPO

`--objective` selects the loss and, with it, the expected data format:

| Objective | Data fields | Notes |
| --- | --- | --- |
| `multidpo` | `prompt`, `chosen`, `rejected` (list) | Chosen vs. many negatives. Used for the released checkpoint. |
| `mdpo` (CLI default) | `prompt`, `candidate_texts`, `labels` | Rank loss plus NIL abstention loss. |

`build_dpo_data.py` produces `multidpo` rows:

```bash
python -m nel_mdpo.train_mdpo \
  --train-files "data/*train*.jsonl" \
  --eval-files "data/*dev*.jsonl" \
  --model-name unsloth/gpt-oss-20b \
  --output-dir outputs/multidpo \
  --objective multidpo
```

For `--objective mdpo`, first convert those rows with
`nel_mdpo.dpo_data.convert_jsonl_to_mdpo(input_file, output_file)` (there is no
CLI wrapper for it), then train on the converted files.

Defaults (see `nel_mdpo/train_mdpo.py`): 4-bit Unsloth model with LoRA r=16,
3 epochs, batch size 8 x 8 gradient accumulation, learning rate 5e-6, beta 0.1,
eval and checkpoint every 200 steps (keeping 2), best checkpoint by `eval_loss`.
The CLI saves a LoRA/PEFT checkpoint only.

## Export MXFP4 Weights

The training CLI saves the trained LoRA/PEFT checkpoint with
`trainer.save_model(...)`. It does not write the final MXFP4 merged model
folder.

The MXFP4 model hosted on Hugging Face was produced as a separate export step
in the original notebooks (`convert_model_to_mpfx4.ipynb`,
`Finetuning-DPO-MultiNeg-Copy5.ipynb`; not included in this repository). The
equivalent code is:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="dpo_multineg_3003_randomly_shuffle/checkpoint-2000",
    dtype=None,
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

model.save_pretrained_merged(
    "dpo_multineg_3003_randomly_shuffle_2000_mxfp4",
    tokenizer,
    save_method="mxfp4",
)
tokenizer.save_pretrained("dpo_multineg_3003_randomly_shuffle_2000_mxfp4")
```

For GPT-OSS 20B, the exported folder also keeps the base model `config.json`
from the pretrained `unsloth/gpt-oss-20b` snapshot so that the MXFP4 shards can
be served with Transformers/vLLM.

## Hugging Face Assets

Model weights:

```text
https://huggingface.co/Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4
```

Although the repository name contains `1600`, the root model files have been
replaced with checkpoint `2000`. The model was fine-tuned from the GPT-OSS 20B
base/pretrained model. Retrieval and alias-candidate generation artifacts in
this repository use GPT-OSS 120B as the teacher/retriever model.

Training data for the MDPO/DPO run is hosted in the same repository:

```text
https://huggingface.co/Nampfiev1995/GPT-OSS-20B-MDPO-NEL-3003-1600-mxfp4/tree/main/training_data/dpo_dataset_2003_chunk256
```

It contains the chunk-256 DPO JSONL train/dev files for:

```text
hipe2020: de, fr
newseye: de, fi, fr, sv
```

These correspond to the `NewPaper/DPO_dataset_2003` chunk-256 DPO files,
excluding the extra `ajmc`, `letemps`, and `topres19th` subsets.

## Committed Retrieval Files

`data/retrieval_gpt_oss_120b_256/` contains both retrieval versions for each
subset/language:

- original retrieval: `gpt-oss-120b-<dataset>-<lang>-256.jsonl`
- retrieval after adding aliases: `gpt-oss-120b-<dataset>-<lang>-256_alias.jsonl`

```text
hipe2020: de, en, fr
newseye: de, fi, fr, sv
```

The `legacy_newpaper/` folder keeps the original scripts used to produce them
(segment-256 retrieval variants and alias workflow):

```text
legacy_newpaper/retrieval_simple_with_summary_context.py
legacy_newpaper/retrieval_with_alias.py
legacy_newpaper/add_alias_report.py
legacy_newpaper/alias_dictionary_multilingual.json
```

## Results

The validated `3003-2000` paper-score artifacts are committed in:

```text
results/gpt_oss_20b_mdpo_3003_2000_alias_chunk256_paper_scores/
```

This folder contains the HIPE scorer `_nel.tsv` outputs for all seven
dataset/language columns, the four available prediction TSVs for the exact
`DPO-SampleGPTOSS120B-MultiNegative-3003-GPTOSS20B-2000-ALIAS-256` family, and
a `MANIFEST.json` with the scores.

```text
HIPE2020_FR  72.4  DPO-Sample...3003-GPTOSS20B-2000-ALIAS-256
HIPE2020_DE  65.6  DPO-Sample...3003-GPTOSS20B-2000-ALIAS-256
HIPE2020_EN  73.8  DPO-MultiNegative...3003-GPTOSS20B-2000-ALIAS-GPT-OSS-120B
NEWSEYE_FR   68.7  DPO-Sample...3003-GPTOSS20B-2000-ALIAS-256
NEWSEYE_DE   59.2  DPO-Sample...3003-GPTOSS20B-2000-ALIAS-256
NEWSEYE_FI   62.7  DPO-MultiNegative...3003-GPTOSS20B-2000-ALIAS-GPT-OSS-120B
NEWSEYE_SV   65.0  DPO-MultiNegative...3003-GPTOSS20B-2000-ALIAS-GPT-OSS-120B
```

Scores are `NEL-LIT-micro-strict` F1 from
`HIPE-scorer/clef_evaluation.py --task nel --original_nel`. The scorer files
show `NEWSEYE_FI = 62.7` and `NEWSEYE_SV = 65.0`; if a table lists
`FI = 65.0` and `SV = 62.7`, those two columns are swapped relative to the
files.

Other retained result sets:

- `results/gpt_oss_20b_mdpo_3003_1600_chunk256/`: earlier `1600` prediction
  TSVs.
- `results/gpt_oss_20b_mdpo_3003_2000_alias_base_gpt_oss_120b_1600_chunk256/`:
  the broader alias/base `1600` result set referenced in
  `HIPE-2022-baseline/evaluate_notebook.ipynb` cell 87. No `NEWSEYE_SV` TSV was
  found for that exact `GPT-OSS-120B-1600` filename pattern.

## Git-Ignored Content

`.gitignore` excludes `*.jsonl`, `data/`, `runs/`, `outputs/`, checkpoints and
model weights, so generated data and weights stay out of GitHub. The committed
retrieval files under `data/retrieval_gpt_oss_120b_256/` match those patterns
but are tracked anyway, so new files there need `git add -f`.
