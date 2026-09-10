"""
plots.py — grouped comparison plots for multi-element datasets.

When the input contains more than mono-element files, both the Kinetics and the
Isotherms tabs show the same two extra views, each with the relevant
competitiveness indicators next to it:

  A — grouped by element:  one figure per metal, holding its Mono / Bi / Tri
      curves, next to the Level 2 indicators (mixture vs mono) for that metal.
  B — grouped by solution type:  one figure per Mono / Bi / Tri, holding the
      different elements, next to the Level 1 indicators (within-mixture
      fractions, K_d and selectivity) for each matrix.

Both tabs pass the same two structures: *series* holds the plottable curves in
the unit basis currently selected, *records* holds the mass-basis parameters the
indicators are derived from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from indicators import (
    availability,
    level1,
    level2,
    selectivity_matrix,
    solution_sort_key,
)

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_SOLUTION_DASH = {"Mono": "solid", "Bi": "dash", "Tri": "dot"}


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _fmt(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round numeric columns and drop all-empty ones for compact display."""
    if df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(decimals)
    return out.dropna(axis=1, how="all")


def render_grouped_comparison(
    series: list[dict],
    records: list[dict],
    *,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    key_prefix: str,
) -> None:
    """
    Render the by-element and by-solution comparison sections.

    *series* entries need ``Metal``, ``System``, ``Solution``, ``label`` and the
    two array fields named by *x_field* / *y_field*.
    *records* entries need ``metal``, ``system``, ``solution``, ``qe_mg`` and
    ``Ce_mg``; ``k1`` and ``k2_mg`` are optional.
    """
    if not series:
        return

    st.divider()
    st.subheader("🧭 Multi-element comparison")

    avail = availability(records)
    st.caption(
        "Solution types present: **" + ", ".join(avail["solutions"]) + "**. "
        "Indicator ratios (R, S, α, f) are dimensionless and identical on both "
        "unit bases; the capacities behind them are shown on the mass basis."
    )

    _by_element(series, records, x_field, y_field, x_title, y_title,
                key_prefix, avail)
    _by_solution(series, records, x_field, y_field, x_title, y_title,
                 key_prefix)


# ── A — grouped by element ─────────────────────────────────────────────────────

def _by_element(series, records, x_field, y_field, x_title, y_title,
                key_prefix, avail) -> None:
    st.markdown("### Grouped by element")
    st.caption(
        "Each metal across the solutions it was measured in — how much the "
        "co-solutes changed its uptake."
    )

    metals = sorted({s["Metal"] for s in series})
    if not metals:
        return

    mono_records = [r for r in records if r.get("solution") == "Mono"]

    for metal in metals:
        subset = sorted(
            [s for s in series if s["Metal"] == metal],
            key=lambda s: solution_sort_key(s["Solution"]) + (s["System"],),
        )
        if not subset:
            continue

        fig = go.Figure()
        for i, s in enumerate(subset):
            x = np.asarray(s[x_field], dtype=float)
            y = np.asarray(s[y_field], dtype=float)
            if not len(x):
                continue
            order = np.argsort(x)
            fig.add_trace(go.Scatter(
                x=x[order], y=y[order],
                mode="lines+markers",
                name=f"{s['Solution']} — {s['System']}",
                line=dict(color=_color(i),
                          dash=_SOLUTION_DASH.get(s["Solution"], "solid")),
                marker=dict(color=_color(i), size=8),
            ))

        fig.update_layout(
            title=f"{metal} — Mono vs Bi vs Tri",
            xaxis_title=x_title, yaxis_title=y_title,
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig, width='stretch',
                        key=f"{key_prefix}_elem_{metal}")

        # Level 2 — mixture vs mono, for this metal
        mix = [r for r in records
               if r["metal"] == metal and r.get("solution") != "Mono"]
        if not mix:
            continue
        if not avail["level2"]["ok"] or not any(
            r["metal"] == metal for r in mono_records
        ):
            st.caption(
                f"Level 2 indicators for {metal} need a mono-element "
                f"reference for this metal — none was loaded."
            )
            continue

        tbl = level2(mix, [r for r in mono_records if r["metal"] == metal])
        cols = [c for c in ["System", "Solution", "qe mono (mg/g)",
                            "qe mix (mg/g)", "R_q", "S_q (%)",
                            "R_k1", "S_k1", "R_k2", "S_k2", "Effect"]
                if c in tbl.columns]
        st.markdown(f"**Level 2 — competitive effect on {metal}**")
        st.dataframe(_fmt(tbl[cols]), width='stretch', hide_index=True)


# ── B — grouped by solution type ───────────────────────────────────────────────

def _by_solution(series, records, x_field, y_field, x_title, y_title,
                 key_prefix) -> None:
    st.markdown("### Grouped by solution type")
    st.caption(
        "Each solution with all the elements it contains — who dominates "
        "inside the mixture."
    )

    solutions = sorted({s["Solution"] for s in series}, key=solution_sort_key)

    for sol in solutions:
        subset = sorted(
            [s for s in series if s["Solution"] == sol],
            key=lambda s: (s["System"], s["Metal"]),
        )
        if not subset:
            continue

        fig = go.Figure()
        for i, s in enumerate(subset):
            x = np.asarray(s[x_field], dtype=float)
            y = np.asarray(s[y_field], dtype=float)
            if not len(x):
                continue
            order = np.argsort(x)
            name = (s["Metal"] if sol == "Mono"
                    else f"{s['Metal']} [{s['System']}]")
            fig.add_trace(go.Scatter(
                x=x[order], y=y[order],
                mode="lines+markers", name=name,
                line=dict(color=_color(i)),
                marker=dict(color=_color(i), size=8),
            ))

        fig.update_layout(
            title=f"{sol}-element solutions — elements compared",
            xaxis_title=x_title, yaxis_title=y_title,
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig, width='stretch',
                        key=f"{key_prefix}_sol_{sol}")

        if sol == "Mono":
            continue

        # Level 1 — within each matrix of this solution type
        systems = sorted({r["system"] for r in records
                          if r.get("solution") == sol})
        for system in systems:
            recs = [r for r in records if r.get("system") == system]
            if len(recs) < 2:
                continue

            st.markdown(f"**Level 1 — apparent competition in `{system}`**")
            l1 = level1(recs)
            cols = [c for c in ["Metal", "qe (mg/g)", "qe (mmol/g)",
                                "qe (meq/g)", "f_mass", "f_mol", "f_charge",
                                "Kd (L/g)", "rank_qe_mass", "rank_qe_mol",
                                "rank_qe_charge"]
                    if c in l1.columns]
            st.dataframe(_fmt(l1[cols]), width='stretch', hide_index=True)

            alpha = selectivity_matrix(recs)
            if not alpha.empty:
                st.caption(
                    "Selectivity factor α(i/j) = K_d,i / K_d,j — "
                    "row *i* over column *j*; α > 1 means the adsorbent "
                    "prefers *i*."
                )
                st.dataframe(_fmt(alpha), width='stretch')
