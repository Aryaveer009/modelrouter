"""Streamlit dashboard: live routing decisions, savings and benchmark results.

Run with:  streamlit run src/modelrouter/dashboard.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import streamlit as st

from modelrouter import __author__, __author_url__, __version__
from modelrouter.config import Settings
from modelrouter.factory import build_router
from modelrouter.tracking import RequestLog

st.set_page_config(page_title="ModelRouter", page_icon="🔀", layout="wide")

settings = Settings.from_env()
RESULTS_DIR = Path(os.environ.get("MODELROUTER_RESULTS", "results"))


@st.cache_resource
def get_log(path: str) -> RequestLog:
    return RequestLog(path)


@st.cache_resource
def get_router():
    return build_router(settings)


def money(x: float) -> str:
    return f"${x:.5f}" if abs(x) < 0.01 else f"${x:,.4f}"


st.title("🔀 ModelRouter")
st.caption("Route easy queries to a cheap model and hard ones to a strong model.")

log = get_log(str(settings.db_path))
live_tab, bench_tab, try_tab = st.tabs(["Live routing", "Benchmark", "Try it"])


@st.fragment(run_every="3s")
def live_panel() -> None:
    summary = log.summary()
    if summary["mock_requests"]:
        st.warning(
            f"{summary['mock_requests']} of {summary['requests']} logged requests came from "
            "MOCK models: synthetic data, not real savings."
        )
    c = st.columns(5)
    c[0].metric("Requests", f"{summary['requests']:,}")
    c[1].metric("Actual cost", money(summary["cost_usd"]))
    c[2].metric("If all strong", money(summary["baseline_strong_cost_usd"]))
    c[3].metric("Saved", f"{summary['saved_pct']:.1f}%", money(summary["saved_usd"]))
    c[4].metric("Escalation rate", f"{summary['escalation_rate_pct']:.1f}%")
    rows = log.recent(200)
    if not rows:
        st.info(
            "No requests yet. Send traffic to the proxy, use the 'Try it' tab, or run "
            "`modelrouter demo` for synthetic mock traffic."
        )
        return
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("id")
    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative cost vs always-strong")
        chart = pd.DataFrame(
            {
                "ModelRouter": df["cost_usd"].cumsum(),
                "Always strong": df["baseline_strong_cost_usd"].cumsum(),
            }
        )
        st.line_chart(chart)
    with right:
        st.subheader("Where answers came from")
        st.bar_chart(df["final_tier"].value_counts())
    st.subheader("Recent routing decisions")
    view = df.sort_values("id", ascending=False).head(30).copy()
    view["time"] = view["ts"].map(lambda t: time.strftime("%H:%M:%S", time.localtime(t)))
    view["saved_usd"] = view["baseline_strong_cost_usd"] - view["cost_usd"]
    st.dataframe(
        view[
            [
                "time",
                "query",
                "p_hard",
                "initial_tier",
                "final_tier",
                "escalation_reason",
                "cost_usd",
                "saved_usd",
                "latency_ms",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


with live_tab:
    live_panel()

with bench_tab:
    results_path = RESULTS_DIR / "results.json"
    if not results_path.exists():
        st.info("No benchmark results yet. Run `modelrouter bench` (see the README).")
    else:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        meta = data["meta"]
        if meta.get("mock"):
            st.error("These are MOCK results (synthetic).")
        st.write(
            f"**{meta['cheap_model']}** (cheap) vs **{meta['strong_model']}** (strong), "
            f"{meta['n_queries']} queries, generated {meta['generated_at']}."
        )
        base = data["baselines"]
        default_t = data["default_threshold"]
        clf = "embedding" if "embedding" in data["sweeps"] else next(iter(data["sweeps"]))
        router_pt = next(m for m in data["sweeps"][clf] if abs(m["threshold"] - default_t) < 1e-9)
        import pandas as pd

        table = pd.DataFrame(
            [
                {
                    "Strategy": "Always cheap",
                    **{
                        k: base["always_cheap"][k]
                        for k in ("accuracy", "total_cost_usd", "avg_latency_s", "cost_saved_pct")
                    },
                },
                {
                    "Strategy": "Always strong",
                    **{
                        k: base["always_strong"][k]
                        for k in ("accuracy", "total_cost_usd", "avg_latency_s", "cost_saved_pct")
                    },
                },
                {
                    "Strategy": f"Router ({clf}, t={default_t})",
                    **{
                        k: router_pt[k]
                        for k in ("accuracy", "total_cost_usd", "avg_latency_s", "cost_saved_pct")
                    },
                },
            ]
        )
        table["accuracy"] = (table["accuracy"] * 100).round(1)
        st.dataframe(
            table.rename(
                columns={
                    "accuracy": "Accuracy %",
                    "total_cost_usd": "Total cost $",
                    "avg_latency_s": "Avg latency s",
                    "cost_saved_pct": "Cost saved vs strong %",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        png = RESULTS_DIR / "pareto.png"
        if png.exists():
            st.image(str(png), caption="Cost vs accuracy (Pareto view)")
        st.subheader("Threshold sweep")
        sweep = pd.DataFrame(data["sweeps"][clf])[
            ["threshold", "accuracy", "total_cost_usd", "pct_answered_by_cheap", "cost_saved_pct"]
        ]
        st.line_chart(sweep.set_index("total_cost_usd")["accuracy"])
        st.dataframe(sweep, hide_index=True, width="stretch")

with try_tab:
    st.write(
        "Send a query through the router in-process. Mock mode: "
        f"**{'on' if settings.mock else 'off (real API calls, real cost)'}**."
    )
    prompt = st.text_area("Query", "What is the capital of France?")
    if st.button("Route it") and prompt.strip():
        try:
            res = asyncio.run(get_router().handle([{"role": "user", "content": prompt}]))
            log.record(res, mock=settings.mock)
            st.success(
                f"Answered by the **{res.final_tier}** model ({res.final.model}); "
                f"P(hard) = {res.p_hard:.2f}; cost {money(res.cost_usd)}; "
                f"{'escalated: ' + str(res.escalation_reason) if res.escalated else 'no escalation'}"
            )
            st.write(res.final.text)
        except Exception as exc:  # surface config/provider errors in the UI
            st.error(str(exc))

st.divider()
st.markdown(
    f"<div style='text-align:center;opacity:.75'>Built by "
    f"<a href='{__author_url__}' target='_blank'>{__author__}</a> · ModelRouter v{__version__}</div>",
    unsafe_allow_html=True,
)
