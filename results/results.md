## Results

Cheap model `openai/gpt-oss-20b`, strong model `openai/gpt-oss-120b`, 52 queries ({'gsm8k': 52}). Generated 2026-10-18T15:30:24+00:00.

| Strategy | Accuracy (95% CI) | Total cost | Cost / 1k queries | Avg latency | Answered by cheap | Escalated | Cost saved vs strong | Accuracy vs strong |
|---|---|---|---|---|---|---|---|---|
| Always cheap | 100.0% (93.1%-100.0%) | $0.00391 | $0.0752 | 0.60s | 100% | 0% | +52.2% | +0.0 pp |
| Always strong | 100.0% (93.1%-100.0%) | $0.00818 | $0.1572 | 0.81s | 0% | 0% | +0.0% | +0.0 pp |
| Router (embedding, threshold 0.5) | 100.0% (93.1%-100.0%) | $0.00436 | $0.0838 | 0.63s | 87% | 0% | +46.7% | +0.0 pp |

### Cost saved at a given accuracy (best threshold on the benchmark itself, optimistic)

| Router | keep >= 100% of strong accuracy | keep >= 99% of strong accuracy | keep >= 98% of strong accuracy | keep >= 95% of strong accuracy | keep >= 90% of strong accuracy |
|---|---|---|---|---|---|
| Router: embeddings + LR | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) |
| Router: heuristic baseline | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) |
| Router: random (sanity check) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) | 52.2% (acc 100.0%, t=0.00) |

### Same, with the threshold chosen on one half and measured on the other half

Target: keep >= 95% of strong-model accuracy. 300 random splits.

| Router | Cost saved on held-out half (mean, 5th-95th pct) | Accuracy retained (mean) |
|---|---|---|
| Router: embeddings + LR | 52.2% (47.6% to 56.1%) | 100.0% |
| Router: heuristic baseline | 52.2% (47.6% to 56.1%) | 100.0% |
| Router: random (sanity check) | 52.2% (47.6% to 56.1%) | 100.0% |

### Accuracy by source

| Strategy | gsm8k |
|---|---|
| Always cheap | 100.0% |
| Always strong | 100.0% |

### Classifier quality on the benchmark

| Classifier | AUC for predicting cheap-model failure | Avg classify time |
|---|---|---|
| embedding | n/a | 8.59 ms |
| heuristic | n/a | 0.04 ms |
| random | n/a | 0.00 ms |

### Full threshold sweep (embedding)

| Threshold | Accuracy | Total cost | Answered by cheap | Escalated | Cost saved |
|---|---|---|---|---|---|
| 0.00 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.05 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.10 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.15 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.20 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.25 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.30 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.35 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.40 | 100.0% | $0.00391 | 100% | 0% | +52.2% |
| 0.45 | 100.0% | $0.00394 | 98% | 0% | +51.8% |
| 0.50 | 100.0% | $0.00436 | 87% | 0% | +46.7% |
| 0.55 | 100.0% | $0.00439 | 85% | 0% | +46.3% |
| 0.60 | 100.0% | $0.00489 | 75% | 0% | +40.2% |
| 0.65 | 100.0% | $0.00556 | 62% | 0% | +31.9% |
| 0.70 | 100.0% | $0.00665 | 42% | 0% | +18.6% |
| 0.75 | 100.0% | $0.00723 | 29% | 0% | +11.5% |
| 0.80 | 100.0% | $0.00777 | 10% | 0% | +5.0% |
| 0.85 | 100.0% | $0.00810 | 2% | 0% | +0.9% |
| 0.90 | 100.0% | $0.00818 | 0% | 0% | +0.0% |
| 0.95 | 100.0% | $0.00818 | 0% | 0% | +0.0% |
| 1.00 | 100.0% | $0.00818 | 0% | 0% | +0.0% |
