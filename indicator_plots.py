"""
indicator_plots.py — the competitiveness figures of the Indicators tab.

These reproduce the figure set of the project report *Adsorption
Competitiveness — Pb, Cu, Cr(III)*:

    1  kinetic fits, mono (solid) vs mixture (dashed)
    2  isotherm fits, mono (solid) vs mixture (dashed)
    3  Level 2, kinetic side:  R_q, R_v, R_h
    4  Level 2, isotherm side: R_q, R_K
    5  Level 3, who suppresses whom (colour = co-ion)
    6  cross reading, velocity (R_v) vs capacity (R_q^iso)
    7  adsorbed fraction on the three bases
    8  isotherm of the mixture on the three bases
    9  kinetics of the mixture on the three bases
   10  total adsorbed charge per system (CEC signature)

Every figure is built from the mass-basis results stored by the analysis tabs
and reprojects to the molar and charge bases through :mod:`units`, so the
figures never depend on which basis is selected for display.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from units import (
    CHARGE_BASIS,
    MASS_BASIS,
    MOLAR_BASIS,
    basis_factor,
    basis_name,
    cap_unit,
    conc_unit,
    resolve_basis,
)

# Deterministic colour per metal, so a metal keeps its colour in every figure.
_METAL_PALETTE = [
    "#2c3e50", "#e67e22", "#27ae60", "#c0392b",
    "#8e44ad", "#16a085", "#d4ac0d", "#7f8c8d",
]

# Indicator colours, matching the report
_C_CAPACITY = "#3498db"     # R_q
_C_VELOCITY = "#e67e22"     # R_v
_C_RATE     = "#27ae60"     # R_h
_C_AFFINITY = "#c0392b"     # R_K

_REF_LINE = dict(color="#7f8c8d", dash="dash", width=1.5)

_THREE_BASES = [MASS_BASIS, MOLAR_BASIS, CHARGE_BASIS]


def metal_colors(metals) -> dict[str, str]:
    """Stable metal → colour map (alphabetical, so it never shifts)."""
    return {m: _METAL_PALETTE[i % len(_METAL_PALETTE)]
            for i, m in enumerate(sorted(metals))}


def _basis_note(basis: str, blocked: list[str]) -> str:
    if not blocked:
        return ""
    return (f"  — {', '.join(blocked)} cannot be converted, "
            f"so the whole chart stays on the mass basis")


def _labels(values, fmt: str = "{:.2f}") -> list[str]:
    out = []
    for v in values:
        try:
            out.append(fmt.format(float(v)) if v is not None and
                       np.isfinite(float(v)) else "")
        except (TypeError, ValueError):
            out.append("")
    return out


def _ratio_layout(fig: go.Figure, title: str, y_title: str = "R  (mono = 1)",
                  y_max: float | None = None) -> go.Figure:
    """Common layout for the R-ratio bar charts, with the R = 1 reference line."""
    fig.add_hline(y=1.0, line=_REF_LINE)
    fig.update_layout(
        title=title, yaxis_title=y_title, template="plotly_white",
        height=420, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uniformtext=dict(mode="hide", minsize=8),
    )
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 1 & 2 — model fits, mono vs mixture
# ══════════════════════════════════════════════════════════════════════════════

def _pso(t, qe, k2):
    denom = 1.0 + k2 * qe * t
    return (k2 * qe ** 2 * t) / np.where(denom == 0, 1e-12, denom)


def _pfo(t, qe, k1):
    return qe * (1.0 - np.exp(-k1 * t))


def _langmuir(Ce, qmax, KL):
    return (qmax * KL * Ce) / (1.0 + KL * Ce)


def fig_kinetic_fits(series: list[dict], mixture: str,
                     basis: str = MASS_BASIS) -> go.Figure | None:
    """
    Fig. 1 — q(t) for every metal: its mono-element series (solid) against the
    same metal inside *mixture* (dashed), with the fitted kinetic curve.
    """
    mono = {s["Metal"]: s for s in series if s.get("Solution") == "Mono"}
    mix = {s["Metal"]: s for s in series if s.get("System") == mixture}
    metals = sorted(set(mono) | set(mix))
    if not metals:
        return None

    basis, blocked = resolve_basis(metals, basis)
    colors = metal_colors(metals)
    fig = go.Figure()
    model_used = set()

    for metal in metals:
        f = basis_factor(metal, basis)
        if f is None:
            f = 1.0
        for s, dash, tag in ((mono.get(metal), "solid", "mono"),
                             (mix.get(metal), "dash", mixture)):
            if s is None:
                continue
            t = np.asarray(s["t"], dtype=float)
            q = np.asarray(s["q_mg"], dtype=float) * f
            if not len(t):
                continue
            fig.add_trace(go.Scatter(
                x=t, y=q, mode="markers", showlegend=False,
                marker=dict(color=colors[metal], size=7,
                            symbol="circle" if dash == "solid" else "triangle-up"),
                hovertemplate=f"{metal} {tag}<br>t=%{{x}} min<br>q=%{{y:.3f}}<extra></extra>",
            ))
            fit = s.get("2nd") or s.get("1st")
            if fit is None:
                continue
            model_used.add("PSO" if "k2_mg" in fit else "PFO")
            t_fit = np.linspace(0.0, t.max() * 1.02, 200)
            q_fit = (_pso(t_fit, fit["qe_mg"], fit["k2_mg"]) if "k2_mg" in fit
                     else _pfo(t_fit, fit["qe_mg"], fit["k1"])) * f
            fig.add_trace(go.Scatter(
                x=t_fit, y=q_fit, mode="lines",
                name=f"{metal} {tag}",
                line=dict(color=colors[metal], dash=dash, width=2.5),
            ))

    model = "/".join(sorted(model_used)) or "fit"
    fig.update_layout(
        title=(f"Kinetics ({model}) — mono (solid) vs {mixture} (dashed)"
               f"{_basis_note(basis, blocked)}"),
        xaxis_title="t  (min)", yaxis_title=f"q  ({cap_unit(basis)})",
        template="plotly_white", height=460,
    )
    return fig


def fig_isotherm_fits(series: list[dict], mixture: str,
                      basis: str = MASS_BASIS) -> go.Figure | None:
    """
    Fig. 2 — qₑ(Cₑ) for every metal: mono-element (solid) against the same
    metal inside *mixture* (dashed), with the fitted Langmuir curve.
    """
    mono = {s["Metal"]: s for s in series if s.get("Solution") == "Mono"}
    mix = {s["Metal"]: s for s in series if s.get("System") == mixture}
    metals = sorted(set(mono) | set(mix))
    if not metals:
        return None

    basis, blocked = resolve_basis(metals, basis)
    colors = metal_colors(metals)
    fig = go.Figure()

    for metal in metals:
        f = basis_factor(metal, basis)
        if f is None:
            f = 1.0
        for s, dash, tag in ((mono.get(metal), "solid", "mono"),
                             (mix.get(metal), "dash", mixture)):
            if s is None:
                continue
            ce = np.asarray(s["Ce_mg"], dtype=float)
            qe = np.asarray(s["qe_mg"], dtype=float)
            if not len(ce):
                continue
            fig.add_trace(go.Scatter(
                x=ce * f, y=qe * f, mode="markers", showlegend=False,
                marker=dict(color=colors[metal], size=7,
                            symbol="circle" if dash == "solid" else "triangle-up"),
                hovertemplate=f"{metal} {tag}<br>Ce=%{{x:.3f}}<br>qe=%{{y:.3f}}<extra></extra>",
            ))
            lang = s.get("langmuir")
            if not lang:
                continue
            ce_fit = np.linspace(max(ce.min() * 0.2, 1e-3), ce.max() * 1.05, 250)
            fig.add_trace(go.Scatter(
                x=ce_fit * f,
                y=_langmuir(ce_fit, lang["qmax_mg"], lang["KL_mg"]) * f,
                mode="lines", name=f"{metal} {tag}",
                line=dict(color=colors[metal], dash=dash, width=2.5),
            ))

    fig.update_layout(
        title=(f"Langmuir isotherm — mono (solid) vs {mixture} (dashed)"
               f"{_basis_note(basis, blocked)}"),
        xaxis_title=f"Cₑ  ({conc_unit(basis)})",
        yaxis_title=f"qₑ  ({cap_unit(basis)})",
        template="plotly_white", height=460,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 3, 4 & 6 — Level 2 and the cross reading
# ══════════════════════════════════════════════════════════════════════════════

def _l2_bars(l2, mixture: str, spec: list[tuple[str, str, str]],
             title: str) -> go.Figure | None:
    """Grouped R-ratio bars for one mixture; *spec* is [(column, label, colour)]."""
    if l2 is None or l2.empty or "System" not in l2.columns:
        return None
    sub = l2[l2["System"] == mixture]
    if sub.empty:
        return None

    metals = list(sub["Metal"])
    fig = go.Figure()
    any_data = False
    for col, label, color in spec:
        if col not in sub.columns:
            continue
        vals = sub[col].tolist()
        if all(v is None or (isinstance(v, float) and not np.isfinite(v))
               for v in vals):
            continue
        any_data = True
        fig.add_trace(go.Bar(
            x=metals, y=vals, name=label, marker_color=color,
            text=_labels(vals), textposition="outside", cliponaxis=False,
        ))
    if not any_data:
        return None

    top = max([v for tr in fig.data for v in tr.y
               if v is not None and np.isfinite(v)] + [1.0])
    return _ratio_layout(fig, title, y_max=top * 1.25)


def fig_level2_kinetics(l2, mixture: str) -> go.Figure | None:
    """Fig. 3 — capacity, velocity and initial-rate ratios in *mixture*."""
    return _l2_bars(
        l2, mixture,
        [("R_q", "R_q (capacity)", _C_CAPACITY),
         ("R_v", "R_v (velocity)", _C_VELOCITY),
         ("R_h", "R_h (initial rate)", _C_RATE)],
        f"Kinetic Level 2 — {mixture} vs mono",
    )


def fig_level2_isotherm(l2, mixture: str) -> go.Figure | None:
    """Fig. 4 — maximum-capacity and affinity ratios in *mixture*."""
    return _l2_bars(
        l2, mixture,
        [("R_q", "R_q (q_max, capacity)", _C_CAPACITY),
         ("R_K", "R_K (K_L, affinity)", _C_AFFINITY)],
        f"Isotherm Level 2 — {mixture} vs mono",
    )


def fig_cross_axis(cross, mixture: str | None = None) -> go.Figure | None:
    """
    Fig. 6 — velocity (kinetics) against capacity (isotherm) per metal:
    a velocity bar close to 1 beside a much lower capacity bar means the
    competition is thermodynamic rather than kinetic.
    """
    if cross is None or cross.empty:
        return None
    sub = cross if mixture is None else cross[cross["System"] == mixture]
    if sub.empty:
        return None

    metals = list(sub["Metal"])
    fig = go.Figure()
    for col, label, color in (("R_v (kinetics)", "R_v (velocity)", _C_VELOCITY),
                              ("R_q iso (q_max)", "R_q iso (capacity)", _C_CAPACITY)):
        if col not in sub.columns:
            continue
        vals = sub[col].tolist()
        fig.add_trace(go.Bar(
            x=metals, y=vals, name=label, marker_color=color,
            text=_labels(vals), textposition="outside", cliponaxis=False,
        ))
    if not fig.data:
        return None

    title = "Cross reading — velocity × capacity"
    if mixture:
        title += f"  ({mixture})"
    top = max([v for tr in fig.data for v in tr.y
               if v is not None and np.isfinite(v)] + [1.0])
    return _ratio_layout(fig, title, y_max=top * 1.25)


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 5 — Level 3, who suppresses whom
# ══════════════════════════════════════════════════════════════════════════════

def fig_level3_pairwise(pw) -> go.Figure | None:
    """
    Fig. 5 — for each target metal *i*, one bar per co-ion *j* showing the
    suppression S_i(j) it causes.  The colour is the co-ion, so the tallest
    bar of a group names the strongest competitor of that target.
    """
    if pw is None or pw.empty:
        return None

    targets = sorted(pw["Metal i"].unique())
    co_ions = sorted(pw["Co-ion j"].unique())
    colors = metal_colors(set(targets) | set(co_ions))

    fig = go.Figure()
    for j in co_ions:
        sub = pw[pw["Co-ion j"] == j].set_index("Metal i")
        vals = [float(sub.loc[i, "S_i(j)"]) if i in sub.index else None
                for i in targets]
        fig.add_trace(go.Bar(
            x=[f"target {i}" for i in targets], y=vals, name=j,
            marker_color=colors[j],
            text=[j if v is not None else "" for v in vals],
            textposition="outside", cliponaxis=False,
        ))

    fig.update_layout(
        title="Level 3 — who suppresses whom (colour = co-ion)",
        yaxis_title="S<sub>i(j)</sub>  (fraction of capacity lost)",
        template="plotly_white", height=420, barmode="group",
        legend=dict(title="co-ion", orientation="h", yanchor="bottom", y=1.02, x=0),
        uniformtext=dict(mode="hide", minsize=8),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 7 — adsorbed fraction on the three bases
# ══════════════════════════════════════════════════════════════════════════════

def fig_fractions(l1, system: str) -> go.Figure | None:
    """
    Fig. 7 — f_mass, f_mol and f_charge side by side.  When the metals differ
    in molar mass or valence the three bars disagree, and the basis decides
    which metal "dominates" the adsorbed phase.
    """
    if l1 is None or l1.empty:
        return None

    metals = list(l1["Metal"])
    fig = go.Figure()
    for col, label, color in (("f_mass", "f mass", "#95a5a6"),
                              ("f_mol", "f molar", _C_CAPACITY),
                              ("f_charge", "f charge", _C_RATE)):
        if col not in l1.columns:
            continue
        vals = l1[col].tolist()
        if all(v is None or (isinstance(v, float) and not np.isfinite(v))
               for v in vals):
            continue
        fig.add_trace(go.Bar(
            x=metals, y=vals, name=label, marker_color=color,
            text=_labels(vals), textposition="outside", cliponaxis=False,
        ))
    if not fig.data:
        return None

    fig.update_layout(
        title=f"Adsorbed fraction in {system} — the three bases",
        yaxis_title="f<sub>i</sub>", template="plotly_white",
        height=420, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uniformtext=dict(mode="hide", minsize=8),
    )
    fig.update_yaxes(range=[0, 1.15])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 8 & 9 — the mixture seen on the three bases
# ══════════════════════════════════════════════════════════════════════════════

def _three_base_panels(series: list[dict], system: str, kind: str) -> go.Figure | None:
    """Shared builder for Figs. 8 (isotherm) and 9 (kinetics)."""
    sub = [s for s in series if s.get("System") == system]
    if not sub:
        return None
    metals = sorted({s["Metal"] for s in sub})
    colors = metal_colors(metals)

    titles = [f"{basis_name(b)} ({cap_unit(b)})" for b in _THREE_BASES]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.07)

    skipped: set[str] = set()
    for col_i, b in enumerate(_THREE_BASES, start=1):
        for s in sub:
            metal = s["Metal"]
            f = basis_factor(metal, b)
            if f is None:
                skipped.add(metal)
                continue
            if kind == "kinetics":
                t = np.asarray(s["t"], dtype=float)
                fit = s.get("2nd") or s.get("1st")
                if not len(t) or fit is None:
                    continue
                x = np.linspace(0.0, t.max() * 1.02, 200)
                y = (_pso(x, fit["qe_mg"], fit["k2_mg"]) if "k2_mg" in fit
                     else _pfo(x, fit["qe_mg"], fit["k1"])) * f
            else:
                ce = np.asarray(s["Ce_mg"], dtype=float)
                lang = s.get("langmuir")
                if not len(ce) or not lang:
                    continue
                # Cₑ stays in mg/L in all three panels: only the adsorbed
                # amount is reprojected, so the panels isolate the effect of
                # the basis on q rather than merely rescaling both axes.
                x = np.linspace(max(ce.min() * 0.2, 1e-3), ce.max() * 1.05, 250)
                y = _langmuir(x, lang["qmax_mg"], lang["KL_mg"]) * f
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines", name=metal,
                legendgroup=metal, showlegend=(col_i == 1),
                line=dict(color=colors[metal], width=2.5),
            ), row=1, col=col_i)

    if not fig.data:
        return None

    x_title = "t  (min)" if kind == "kinetics" else f"Cₑ  ({conc_unit(MASS_BASIS)})"
    for c in range(1, 4):
        fig.update_xaxes(title_text=x_title, row=1, col=c)

    what = "Kinetics" if kind == "kinetics" else "Isotherm"
    note = ""
    if skipped:
        note = f"  — {', '.join(sorted(skipped))} omitted where unconvertible"
    fig.update_layout(
        title=f"{what} of {system} on the three bases{note}",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.14, x=0),
    )
    return fig


def fig_three_bases_kinetics(series: list[dict], system: str) -> go.Figure | None:
    """Fig. 9 — the mixture's kinetic curves in mg/g, mmol/g and meq/g."""
    return _three_base_panels(series, system, "kinetics")


def fig_three_bases_isotherm(series: list[dict], system: str) -> go.Figure | None:
    """Fig. 8 — the mixture's isotherms in mg/g, mmol/g and meq/g."""
    return _three_base_panels(series, system, "isotherm")


# ══════════════════════════════════════════════════════════════════════════════
# Fig. 10 — total adsorbed charge per system
# ══════════════════════════════════════════════════════════════════════════════

def fig_total_charge(records: list[dict], study: str = "") -> go.Figure | None:
    """
    Fig. 10 — Σ q (meq/g) per system.  If the mixtures converge on a common
    value while the mono systems sit below it, the adsorbent is behaving like
    a fixed pool of exchangeable charge (its CEC) shared among the ions.
    """
    totals: dict[str, float] = {}
    n_metals: dict[str, int] = {}
    for r in records:
        sys_ = r.get("system")
        q = r.get("qe_mg")
        if sys_ is None or q is None:
            continue
        f = basis_factor(r["metal"], CHARGE_BASIS)
        if f is None:                       # no valence → no charge sum
            totals.pop(sys_, None)
            n_metals[sys_] = -1
            continue
        if n_metals.get(sys_) == -1:
            continue
        totals[sys_] = totals.get(sys_, 0.0) + q * f
        n_metals[sys_] = n_metals.get(sys_, 0) + 1

    if not totals:
        return None

    systems = sorted(totals, key=lambda s: (len(s.split("+")), s))
    values = [totals[s] for s in systems]
    is_mix = [len(s.split("+")) > 1 for s in systems]
    colors = ["#3b6ea5" if m else "#95a5a6" for m in is_mix]

    fig = go.Figure(go.Bar(
        x=systems, y=values, marker_color=colors,
        text=_labels(values), textposition="outside", cliponaxis=False,
        showlegend=False,
    ))

    mix_vals = [v for v, m in zip(values, is_mix) if m]
    if len(mix_vals) >= 2:
        fig.add_hrect(y0=min(mix_vals), y1=max(mix_vals),
                      fillcolor="#27ae60", opacity=0.12, line_width=0,
                      annotation_text="mixtures", annotation_position="top left")

    suffix = f" ({study.lower()})" if study else ""
    fig.update_layout(
        title=f"Total adsorbed charge per system{suffix} — CEC signature",
        yaxis_title="Σ q  (meq/g)", xaxis_title="System",
        template="plotly_white", height=420,
        uniformtext=dict(mode="hide", minsize=8),
    )
    fig.update_yaxes(range=[0, max(values) * 1.22])
    return fig
