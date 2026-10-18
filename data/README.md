# Data

Built by `modelrouter build-data` from public datasets (via the Hugging Face datasets-server API).
Training and benchmark queries never overlap: different splits or row ranges, and an exact-text
dedupe pass removes any training query that also appears in the benchmark.

| Source | What | Licence | Benchmark | Train |
|---|---|---|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | grade-school math word problems (gold = number after `####`) | MIT | 100 (`test`) | 250 (`train`) |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | 4-choice questions from 16 subjects (8 reasoning-heavy, 8 recall-style) | MIT | 100 | ~248 |
| [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (`rc.nocontext`) | short factual questions with answer aliases | Apache-2.0 | 100 | 250 |

MMLU subjects. Hard tier: abstract_algebra, college_mathematics, college_physics, formal_logic,
high_school_mathematics, college_chemistry, electrical_engineering, machine_learning. Easy tier:
high_school_geography, high_school_government_and_politics, sociology, us_foreign_policy,
marketing, high_school_psychology, world_religions, management.

## How the labels were created

A training label is `1` = *hard*, meaning "the cheap model is likely to get this wrong".
Two labelling procedures exist, and each record stores which one produced its label:

1. **Prior labels** (free, no API calls). The label comes from the *source tier*: GSM8K = hard,
   TriviaQA = easy, MMLU = hard or easy by subject (lists above). This is a weak label: it mostly
   teaches the classifier to recognise the task type, and it will mark an easy GSM8K problem as
   hard. It exists so the pipeline can run before any API key is configured.
2. **Empirical labels** (`modelrouter label`, costs real money on the cheap model only).
   The cheap model answers every training query once at temperature 0, the answer is scored
   against the gold answer, and `label = 1` if it was wrong. This is the label the router actually
   needs, because the router's job is to predict where the cheap model fails.

`modelrouter train --labels auto` uses empirical labels when every record has one, otherwise
prior labels. The shipped classifier's `models/classifier.json` records its label source,
training-set size and cross-validated metrics in its `meta` field.

Files: `train.jsonl`, `bench.jsonl` (one JSON record per line; schema in
`src/modelrouter/data/records.py`).
