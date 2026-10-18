"""Turn an Analysis into files: results.json, results.md, per_query.jsonl and Pareto charts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..costing import format_usd
from .analysis import RETENTIONS, Analysis, StrategyMetrics

# Reference categorical palette (slots 1-3), validated light/dark; neutral for the baseline.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
        "embedding": "#2a78d6",
        "heuristic": "#eb6834",
        "random": "#8a8984",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#33322f",
        "embedding": "#3987e5",
        "heuristic": "#d95926",
        "random": "#9a9993",
    },
}
LABELS = {
    "embedding": "Router: embeddings + LR",
    "heuristic": "Router: heuristic baseline",
    "random": "Router: random (sanity check)",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_pareto(
    analysis: Analysis, path: Path, theme: str = "light", *, mock: bool = False
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(c["grid"])
    ax.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=c["ink2"], labelsize=9)

    for cname, pts in analysis.sweeps.items():
        if cname not in c:
            continue
        pts = sorted(pts, key=lambda m: (m.total_cost_usd, m.accuracy))
        xs = [m.total_cost_usd for m in pts]
        ys = [m.accuracy * 100 for m in pts]
        ax.plot(
            xs,
            ys,
            color=c[cname],
            linewidth=2,
            marker="o",
            markersize=4.5,
            markeredgecolor=c["surface"],
            markeredgewidth=1.5,
            label=LABELS[cname],
            zorder=3,
        )
        for m in pts:
            if m.threshold is not None and abs(m.threshold - analysis.default_threshold) < 1e-9:
                ax.scatter(
                    [m.total_cost_usd],
                    [m.accuracy * 100],
                    s=130,
                    facecolors="none",
                    edgecolors=c[cname],
                    linewidths=2,
                    zorder=4,
                )

    base = analysis.baselines
    for key, label, marker in (
        ("always_cheap", "Always cheap", "s"),
        ("always_strong", "Always strong", "s"),
        ("oracle", "Oracle (upper bound)", "D"),
    ):
        m = base[key]
        ax.scatter(
            [m.total_cost_usd],
            [m.accuracy * 100],
            marker=marker,
            s=60,
            color=c["ink"],
            edgecolors=c["surface"],
            linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            label,
            (m.total_cost_usd, m.accuracy * 100),
            textcoords="offset points",
            xytext=(8, -12 if key != "always_strong" else 8),
            fontsize=9,
            color=c["ink"],
        )

    ax.set_xlabel("Total cost for the benchmark (USD)", color=c["ink2"], fontsize=10)
    ax.set_ylabel("Accuracy (%)", color=c["ink2"], fontsize=10)
    title = "Cost vs accuracy: closer to the top-left is better"
    ax.set_title(title, loc="left", color=c["ink"], fontsize=12, fontweight="bold", pad=12)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(c["ink2"])
    ax.text(
        0.0,
        -0.17,
        "Ring = default threshold. Each router line sweeps the confidence threshold from 0 to 1.",
        transform=ax.transAxes,
        fontsize=8,
        color=c["ink2"],
    )
    if mock:
        ax.text(
            0.5,
            0.5,
            "MOCK DATA - NOT REAL RESULTS",
            transform=ax.transAxes,
            fontsize=22,
            color="#cc0000",
            alpha=0.35,
            ha="center",
            va="center",
            rotation=20,
        )
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _row(m: StrategyMetrics) -> str:
    return (
        f"| {m.name} | {_pct(m.accuracy)} ({_pct(m.ci_low)}-{_pct(m.ci_high)}) | "
        f"{format_usd(m.total_cost_usd)} | {format_usd(m.cost_per_1k_usd)} | "
        f"{m.avg_latency_s:.2f}s | {m.pct_answered_by_cheap:.0f}% | {m.escalation_rate_pct:.0f}% | "
        f"{m.cost_saved_pct:+.1f}% | {m.accuracy_delta_pp:+.1f} pp |"
    )


TABLE_HEAD = (
    "| Strategy | Accuracy (95% CI) | Total cost | Cost / 1k queries | Avg latency "
    "| Answered by cheap | Escalated | Cost saved vs strong | Accuracy vs strong |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def main_table(analysis: Analysis, classifier: str) -> str:
    sweep = analysis.sweeps[classifier]
    default = next(m for m in sweep if abs((m.threshold or -1) - analysis.default_threshold) < 1e-9)
    default.name = f"Router ({classifier}, threshold {analysis.default_threshold})"
    rows = [analysis.baselines["always_cheap"], analysis.baselines["always_strong"], default]
    return TABLE_HEAD + "\n" + "\n".join(_row(m) for m in rows)


def saved_table(analysis: Analysis) -> str:
    head = (
        "| Router | "
        + " | ".join(f"keep >= {r * 100:.0f}% of strong accuracy" for r in RETENTIONS)
        + " |"
    )
    lines = [head, "|---|" + "---|" * len(RETENTIONS)]
    for cname, table in analysis.saved_at.items():
        cells = []
        for r in RETENTIONS:
            m = table[f"{r:.2f}"]
            cells.append(
                "n/a"
                if m is None
                else f"{m['cost_saved_pct']:.1f}% (acc {m['accuracy'] * 100:.1f}%, t={m['threshold']:.2f})"
            )
        lines.append(f"| {LABELS.get(cname, cname)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def robustness_table(analysis: Analysis) -> str:
    lines = [
        "| Router | Cost saved on held-out half (mean, 5th-95th pct) | Accuracy retained (mean) |",
        "|---|---|---|",
    ]
    for cname, r in analysis.robustness.items():
        lines.append(
            f"| {LABELS.get(cname, cname)} | {r['mean_cost_saved_pct']:.1f}% "
            f"({r['p05_cost_saved_pct']:.1f}% to {r['p95_cost_saved_pct']:.1f}%) | "
            f"{r['mean_accuracy_retained'] * 100:.1f}% |"
        )
    return "\n".join(lines)


def source_table(analysis: Analysis) -> str:
    sources = sorted(analysis.baselines["always_cheap"].per_source_accuracy)
    lines = ["| Strategy | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for m in (analysis.baselines["always_cheap"], analysis.baselines["always_strong"]):
        lines.append(
            f"| {m.name} | " + " | ".join(_pct(m.per_source_accuracy[s]) for s in sources) + " |"
        )
    return "\n".join(lines)


def classifier_table(analysis: Analysis) -> str:
    lines = [
        "| Classifier | AUC for predicting cheap-model failure | Avg classify time |",
        "|---|---|---|",
    ]
    for cname, e in analysis.classifier_eval.items():
        auc = e["auc_predicting_cheap_failure"]
        lines.append(
            f"| {cname} | {'n/a' if auc is None else f'{auc:.3f}'} | {e['avg_classify_ms']:.2f} ms |"
        )
    return "\n".join(lines)


def sweep_table(analysis: Analysis, classifier: str) -> str:
    lines = [
        "| Threshold | Accuracy | Total cost | Answered by cheap | Escalated | Cost saved |",
        "|---|---|---|---|---|---|",
    ]
    for m in analysis.sweeps[classifier]:
        lines.append(
            f"| {m.threshold:.2f} | {_pct(m.accuracy)} | {format_usd(m.total_cost_usd)} | "
            f"{m.pct_answered_by_cheap:.0f}% | {m.escalation_rate_pct:.0f}% | {m.cost_saved_pct:+.1f}% |"
        )
    return "\n".join(lines)


def write_all(
    analysis: Analysis,
    out_dir: str | Path,
    meta: dict[str, Any],
    *,
    default_classifier: str = "embedding",
    mock: bool = False,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        **meta,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mock": mock,
    }
    paths = {
        "json": out / "results.json",
        "md": out / "results.md",
        "pareto": out / "pareto.png",
        "pareto_dark": out / "pareto-dark.png",
        "per_query": out / "per_query.jsonl",
    }
    paths["json"].write_text(
        json.dumps({"meta": meta, **analysis.to_dict()}, indent=2), encoding="utf-8"
    )
    with open(paths["per_query"], "w", encoding="utf-8") as fh:
        for row in analysis.per_query:
            fh.write(json.dumps(row) + "\n")
    write_pareto(analysis, paths["pareto"], "light", mock=mock)
    write_pareto(analysis, paths["pareto_dark"], "dark", mock=mock)

    banner = "> **MOCK RUN - synthetic model outputs, not real results.**\n\n" if mock else ""
    md = (
        f"{banner}## Results\n\n"
        f"Cheap model `{meta.get('cheap_model')}`, strong model `{meta.get('strong_model')}`, "
        f"{meta.get('n_queries')} queries ({meta.get('by_source')}). "
        f"Generated {meta['generated_at']}.\n\n"
        f"{main_table(analysis, default_classifier)}\n\n"
        f"### Cost saved at a given accuracy (best threshold on the benchmark itself, optimistic)\n\n"
        f"{saved_table(analysis)}\n\n"
        f"### Same, with the threshold chosen on one half and measured on the other half\n\n"
        f"Target: keep >= 95% of strong-model accuracy. 300 random splits.\n\n"
        f"{robustness_table(analysis)}\n\n"
        f"### Accuracy by source\n\n{source_table(analysis)}\n\n"
        f"### Classifier quality on the benchmark\n\n{classifier_table(analysis)}\n\n"
        f"### Full threshold sweep ({default_classifier})\n\n{sweep_table(analysis, default_classifier)}\n"
    )
    paths["md"].write_text(md, encoding="utf-8")
    return paths
