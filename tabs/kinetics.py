"""
tabs/kinetics.py — Kinetics tab (HMLab).

Filename convention: AK{n}{Metal1}{C1}[{Metal2}{C2}…]{AdsCode}{AdsConc}t{time}R{rep}.csv
  e.g.  AK1Cu30HIB6p6t005R1.csv  →  1 metal Cu@30 ppm, HIB@6.6 g/L, t=5 min, R=1
        AK2Pb100Cr20MXB2t030R1.csv → 2 metals Pb@100 + Cr@20 ppm, MXB@2 g/L, t=30 min

Each CSV file (XRF output) contains the measured equilibrium concentration Cₜ
and its standard deviation for each metal.  No calibration is required because
the instrument output is already in concentration units (mg/L = ppm).

Grouping:
  Series key: (Metal, N, AdsCode, AdsConc)
  Per series: data points are (t, q(t)) averaged over replicates.

Units: everything is computed once on the mass basis (mg/L, mg/g).  The Base
selector (Mass / Mol / Charge) is a presentation lens — q and Cₜ are scaled by
1/M (mol) or |z|/M (charge), k₁ is basis-invariant and k₂ scales inversely,
which is exactly equivalent to re-fitting in the chosen basis.
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    list_csv_files,
    read_hmlab_csv,
    read_hmlab_csv_bytes,
    parse_hmlab_filename,
    model_first_order,
    model_second_order,
    fit_kinetic,
    linear_first_order,
    linear_second_order,
)
from units import (
    MASS_BASIS,
    basis_factor,
    scale_kinetic_params,
    cap_unit,
    conc_unit,
    k2_unit,
    unit_selector,
    warn_missing_for_basis,
)
from indicators import solution_type
from mode2 import list_mode2_files, read_mode2_file
from plots import render_grouped_comparison

# ── Colour palette ─────────────────────────────────────────────────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

def series_color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]

# ── Parity helper ──────────────────────────────────────────────────────────────

def _use_parity(kinetics_dir: str) -> bool:
    parts = re.split(r"[/\\]", os.path.normcase(kinetics_dir))
    return "M07" in parts


def _build_records(series_results: list[dict]) -> list[dict]:
    """
    Condense the fitted series into the mass-basis records the competitiveness
    indicators consume.

    For kinetics the reference point is the most stable one — the longest
    contact time — so qₑ and Cₑ are the experimental values measured there,
    while k₁ and k₂ come from the PFO / PSO fits.
    """
    records = []
    for s in series_results:
        records.append({
            "metal":    s["Metal"],
            "system":   s["System"],
            "solution": s["Solution"],
            "label":    s["label"],
            "ads":      s["AdsCode"],
            "adsconc":  s["AdsConc"],
            "qe_mg":    s.get("q_last_mg"),
            "Ce_mg":    s.get("Ct_last_mg"),
            "t_ref":    s.get("t_last"),
            "qe_pfo_mg": s.get("1st", {}).get("qe_mg"),
            "qe_pso_mg": s.get("2nd", {}).get("qe_mg"),
            "k1":       s.get("1st", {}).get("k1"),
            "k2_mg":    s.get("2nd", {}).get("k2_mg"),
            # Fit quality travels with the parameters: a rate-constant ratio
            # built on a fit that did not describe the data is meaningless,
            # and the Indicators tab flags it rather than hiding it.
            "r2_pfo":   s.get("1st", {}).get("r2"),
            "r2_pso":   s.get("2nd", {}).get("r2"),
        })
    return records


def _scale(metal: str, basis: str) -> float | None:
    """
    Factor converting a mass-based quantity (mg/L, mg/g) of *metal* to
    *basis* (1 for mass, 1/M for mol, |z|/M for charge).  None when the
    required molar mass / ionic charge is unknown.
    """
    return basis_factor(metal, basis)

# ── Main render ────────────────────────────────────────────────────────────────

def render(kinetics_dir: str) -> None:

    st.info(
        "ℹ️ Concentrations (Cₜ) are read directly from the XRF CSV files, "
        "which are already calibrated in mg/L (= ppm)."
    )

    basis = unit_selector("kinetics")

    use_parity = _use_parity(kinetics_dir)
    if use_parity:
        st.info("ℹ️ **M07 mode**: replicates are split by parity (odd/even).")

    st.divider()

    # ── File selector ──────────────────────────────────────────────────────────
    st.subheader("📂 Input Files — Kinetics")
    st.caption(
        "Two input modes can be combined: **Mode 1** — one XRF CSV per "
        "experimental point (`AK…R#.csv`); **Mode 2** — one Excel workbook "
        "per complete series (`SK<x><Metals>M<xx>[pH<y>]<Ads><conc>.xlsx`)."
    )

    source_kin = st.radio("File source:", ["📁 Folder", "⬆️ Upload"],
                          horizontal=True, key="kin_source")

    kin_entries = []      # Mode 1: {"name": str, "csv_data": list[dict] | None}
    kin_m2_entries = []   # Mode 2: {"name": str, "data": dict} (read_mode2_file)
    m2_errors = []

    if source_kin == "📁 Folder":
        all_kin = list_csv_files(kinetics_dir, r"AK\d")
        all_m2 = list_mode2_files(kinetics_dir, "kinetics")
        if not all_kin and not all_m2:
            st.warning(
                f"No AK* CSV (Mode 1) nor SK* XLSX (Mode 2) files found "
                f"in **{kinetics_dir}**."
            )
            return
        kin_sel = []
        if all_kin:
            saved_kin = st.session_state.get("kin_selected_files", all_kin)
            kin_sel = st.multiselect(
                "Mode 1 — Kinetics point files (AK*.csv):",
                options=all_kin,
                default=[f for f in saved_kin if f in all_kin] or all_kin,
                key="kin_file_selector",
            )
            st.session_state["kin_selected_files"] = kin_sel
        m2_sel = []
        if all_m2:
            saved_m2 = st.session_state.get("kin_m2_selected_files", all_m2)
            m2_sel = st.multiselect(
                "Mode 2 — Kinetics series files (SK*.xlsx):",
                options=all_m2,
                default=[f for f in saved_m2 if f in all_m2] or all_m2,
                key="kin_m2_file_selector",
            )
            st.session_state["kin_m2_selected_files"] = m2_sel
        if not kin_sel and not m2_sel:
            st.info("Select at least one kinetics file.")
            return
        for fname in kin_sel:
            csv_data = read_hmlab_csv(os.path.join(kinetics_dir, fname))
            kin_entries.append({"name": fname, "csv_data": csv_data})
        for fname in m2_sel:
            try:
                data = read_mode2_file(os.path.join(kinetics_dir, fname), fname)
                kin_m2_entries.append({"name": fname, "data": data})
            except ValueError as exc:
                m2_errors.append(f"{fname}: {exc}")
    else:
        uploaded_kin = st.file_uploader(
            "Upload kinetics files — AK*.csv (Mode 1) and/or SK*.xlsx (Mode 2)",
            type=["csv", "xlsx"], accept_multiple_files=True, key="kin_uploader",
        )
        if not uploaded_kin:
            st.info("Upload at least one kinetics file.")
            return
        st.session_state["kin_selected_files"] = [f.name for f in uploaded_kin]
        for f in uploaded_kin:
            if f.name.lower().endswith(".xlsx"):
                try:
                    data = read_mode2_file(f, f.name)
                    kin_m2_entries.append({"name": f.name, "data": data})
                except ValueError as exc:
                    m2_errors.append(f"{f.name}: {exc}")
            else:
                csv_data = read_hmlab_csv_bytes(f)
                kin_entries.append({"name": f.name, "csv_data": csv_data})

    for msg in m2_errors:
        st.warning(f"Skipped Mode 2 file: {msg}")

    # ── Parse filenames & compute q(t) for every file × every metal ───────────
    raw_rows = []
    skipped  = []

    for e in kin_entries:
        meta = parse_hmlab_filename(e["name"])
        if meta is None or meta["type"] != "kinetics":
            skipped.append(f"{e['name']} (cannot parse filename — "
                           "expected AK<n><Metal><Conc><AdsCode><AdsConc>t<time>R<rep>.csv)")
            continue

        csv_data = e["csv_data"]
        if csv_data is None:
            skipped.append(f"{e['name']} (cannot read CSV data)")
            continue

        system_label = "+".join(m["code"] for m in meta["metals"])
        sol_label    = solution_type(meta["n_metals"])

        # Iterate over metals encoded in the filename
        for metal_idx, metal_info in enumerate(meta["metals"]):
            metal_code = metal_info["code"]
            N_metal    = metal_info["N"]

            if metal_idx >= len(csv_data):
                skipped.append(f"{e['name']} — metal {metal_code} not found in CSV data")
                continue

            Ct     = csv_data[metal_idx]["avg"]
            Ct_std = csv_data[metal_idx]["std"]
            q_t    = (N_metal - Ct) / meta["AdsConc"]

            raw_rows.append({
                "Filename":        e["name"],
                "pH":              meta.get("pH"),
                "System":          system_label,
                "Solution":        sol_label,
                "Metal":           metal_code,
                "N (mg/L)":        N_metal,
                "AdsCode":         meta["AdsCode"],
                "AdsConc (g/L)":   meta["AdsConc"],
                "t (min)":         meta["t"],
                "R":               meta["R"],
                "Cₜ (mg/L)":      round(Ct, 4),
                "Cₜ std (mg/L)":  round(Ct_std, 4),
                "%R":             round((N_metal - Ct) / N_metal * 100, 4) if N_metal > 1e-12 else None,
                "q(t) (mg/g)":    round(q_t, 6),
            })

    # ── Mode 2 series files: one workbook already holds every (t, rep) ────────
    for e in kin_m2_entries:
        meta = e["data"]["meta"]
        for w in e["data"]["warnings"]:
            st.warning(f"{e['name']}: {w}")
        system_label = "+".join(m["code"] for m in meta["metals"])
        sol_label    = solution_type(meta["n_metals"])
        for p in e["data"]["points"]:
            q_t = (p["C0"] - p["C"]) / meta["AdsConc"]
            raw_rows.append({
                "Filename":        e["name"],
                "pH":              meta.get("pH"),
                "System":          system_label,
                "Solution":        sol_label,
                "Metal":           p["metal"],
                "N (mg/L)":        p["C0"],
                "AdsCode":         meta["AdsCode"],
                "AdsConc (g/L)":   meta["AdsConc"],
                "t (min)":         p["x"],
                "R":               p["rep"],
                "Cₜ (mg/L)":      round(p["C"], 4),
                "Cₜ std (mg/L)":  round(p["C_std"], 4),
                "%R":             round((p["C0"] - p["C"]) / p["C0"] * 100, 4) if p["C0"] > 1e-12 else None,
                "q(t) (mg/g)":    round(q_t, 6),
            })

    for s in skipped:
        st.warning(f"Skipped: {s}")

    if not raw_rows:
        st.error("No valid kinetics data could be processed.")
        return

    raw_df = pd.DataFrame(raw_rows).sort_values(
        ["System", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)", "t (min)", "R"]
    ).reset_index(drop=True)

    # ── Parity column ──────────────────────────────────────────────────────────
    raw_df["Parity"] = raw_df["R"].apply(lambda r: "odd" if r % 2 != 0 else "even")

    # ── Basis availability ─────────────────────────────────────────────────────
    if basis != MASS_BASIS:
        missing = warn_missing_for_basis(raw_df["Metal"].unique(), basis,
                                         context="for the affected series")
        if missing:
            st.caption(
                "Series of the affected metals are shown on the mass basis "
                "(mg/L · mg/g) instead."
            )

    with st.expander("📋 Raw per-file data", expanded=False):
        st.dataframe(raw_df.drop(columns=["Parity"]), width='stretch')

    # ── Solution types present ─────────────────────────────────────────────────
    solutions_present = list(raw_df["Solution"].unique())
    multi_solution = len(solutions_present) > 1 or any(
        s != "Mono" for s in solutions_present
    )
    if multi_solution:
        st.info(
            f"🧪 Solution types detected: **{', '.join(sorted(solutions_present))}** — "
            "grouped comparison plots are shown below the per-series fits."
        )

    # ── Multi-pH detection ─────────────────────────────────────────────────────
    ph_values = set(raw_df["pH"].dropna().unique())
    multi_ph  = len(ph_values) > 1

    if multi_ph:
        st.info(f"🧪 Detected **{len(ph_values)} pH values**: {sorted(ph_values)}.")

    # ── Average & build series keys ────────────────────────────────────────────
    # System/Solution are part of the key so that the same metal measured in a
    # mono, binary and ternary solution stays in three separate series.
    if use_parity:
        grp_cols = (["pH", "System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)", "Parity", "t (min)"]
                    if multi_ph else
                    ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)", "Parity", "t (min)"])
    else:
        grp_cols = (["pH", "System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)", "t (min)"]
                    if multi_ph else
                    ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)", "t (min)"])

    avg_df = (
        raw_df.groupby(grp_cols, as_index=False)
        .agg(
            **{
                "q (mg/g)":        ("q(t) (mg/g)",   "mean"),
                "q std":           ("q(t) (mg/g)",   "std"),
                "Reps":            ("q(t) (mg/g)",   "count"),
                "Cₜ (mg/L)":      ("Cₜ (mg/L)",     "mean"),
                "Cₜ std (mg/L)":  ("Cₜ std (mg/L)", "mean"),
                "%R":             ("%R",             "mean"),
                "%R std":         ("%R",             "std"),
            }
        )
        .reset_index()
    )

    expander_label = (
        "📋 Averaged data (mean per parity group)"
        if use_parity else
        "📋 Averaged data (mean per series)"
    )
    with st.expander(expander_label, expanded=False):
        col_order = tuple(
            (["pH"] if multi_ph else []) +
            ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)"] +
            (["Parity"] if use_parity else []) +
            ["t (min)", "Reps", "Cₜ (mg/L)", "Cₜ std (mg/L)",
             "%R", "%R std", "q (mg/g)", "q std"]
        )
        st.dataframe(avg_df.round(5), width='stretch',
                     column_order=col_order)

    # ── Series keys ────────────────────────────────────────────────────────────
    key_cols = (
        (["pH"] if multi_ph else []) +
        ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)"] +
        (["Parity"] if use_parity else [])
    )

    series_keys = (
        avg_df[key_cols]
        .drop_duplicates()
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    st.divider()
    st.subheader("Kinetic Model Fitting  —  per series")

    tpc1, tpc2 = st.columns(2)
    with tpc1:
        st.markdown("**Pseudo-1st-order:**")
        st.latex(r"q(t)=q_e(1-e^{-k_1 t})")
    with tpc2:
        st.markdown("**Pseudo-2nd-order:**")
        st.latex(r"q(t)=\frac{k_2 q_e^2 t}{1+k_2 q_e t}")

    fig_qt   = go.Figure()
    fig_lin1 = go.Figure()
    fig_lin2 = go.Figure()

    all_series_results = []

    for idx, row in series_keys.iterrows():
        metal  = row["Metal"]
        system = row["System"]
        sol    = row["Solution"]
        N_val  = row["N (mg/L)"]
        ads    = row["AdsCode"]
        adsc   = row["AdsConc (g/L)"]
        parity = row["Parity"] if use_parity else None
        ph_val = row["pH"] if multi_ph else None

        mask = (
            (avg_df["System"]          == system) &
            (avg_df["Metal"]           == metal) &
            (avg_df["N (mg/L)"]        == N_val) &
            (avg_df["AdsCode"]         == ads)   &
            (avg_df["AdsConc (g/L)"]   == adsc)
        )
        if use_parity:
            mask &= (avg_df["Parity"] == parity)
        if multi_ph:
            mask &= (avg_df["pH"] == ph_val)

        sub   = avg_df[mask].sort_values("t (min)")
        t_arr = sub["t (min)"].values.astype(float)
        q_mg  = sub["q (mg/g)"].values.astype(float)
        q_std_mg = sub["q std"].fillna(0).values.astype(float)
        Ct_mg = sub["Cₜ (mg/L)"].values.astype(float)

        # Unit lens: fall back to the mass basis when the molar mass is unknown
        sc = _scale(metal, basis)
        series_basis = basis if sc is not None else "mg"
        if sc is None:
            sc = 1.0
        q_arr = q_mg * sc
        q_std = q_std_mg * sc

        q_unit = cap_unit(series_basis)
        c_unit = conc_unit(series_basis)

        color = series_color(idx)
        sys_tag   = f" [{system}]" if sol != "Mono" else ""
        ph_prefix = f"pH={ph_val} | " if multi_ph else ""
        par_sfx   = f" | {parity} reps" if use_parity else ""
        label = (f"{ph_prefix}{metal}{sys_tag} ({sol}, N={N_val}) | "
                 f"{ads} {adsc:.4g} g/L{par_sfx}")

        # ── Fits ──────────────────────────────────────────────────────────────
        # Always fitted on the mass basis, then converted, so that the numbers
        # never depend on which unit is selected.  Fitting the rescaled data
        # instead would be algebraically equivalent but not numerically so:
        # an ill-conditioned series can land on a different local optimum.
        qe_guess = max(q_mg.max() * 1.1, 1e-6) if len(q_mg) else 1.0
        popt1_mg, r2_1, mae1_mg = fit_kinetic(
            model_first_order, t_arr, q_mg,
            p0=[qe_guess, 0.01], bounds=([0, 0], [np.inf, np.inf]),
        )
        popt2_mg, r2_2, mae2_mg = fit_kinetic(
            model_second_order, t_arr, q_mg,
            p0=[qe_guess, 0.001], bounds=([0, 0], [np.inf, np.inf]),
        )

        # qₑ scales with the basis, k₁ is invariant, k₂ scales inversely.
        # R² is scale-free; MAE carries the capacity unit.
        popt1 = ([popt1_mg[0] * sc, popt1_mg[1]]
                 if popt1_mg is not None else None)
        popt2 = ([popt2_mg[0] * sc, popt2_mg[1] / sc]
                 if popt2_mg is not None else None)
        mae1 = None if mae1_mg is None else mae1_mg * sc
        mae2 = None if mae2_mg is None else mae2_mg * sc

        t_fit = np.linspace(0, t_arr.max() * 1.05, 500) if len(t_arr) else np.array([])

        # experimental points
        has_err = bool((q_std > 0).any())
        fig_qt.add_trace(go.Scatter(
            x=t_arr, y=q_arr,
            error_y=dict(type="data", array=q_std, visible=has_err),
            mode="markers",
            name=f"{label} — Data",
            marker=dict(color=color, size=9, symbol="circle"),
        ))

        if popt1 is not None and len(t_fit):
            qe1, k1 = popt1
            fig_qt.add_trace(go.Scatter(
                x=t_fit, y=model_first_order(t_fit, qe1, k1), mode="lines",
                name=f"{label} — Pseudo-1st",
                line=dict(color=color, width=2),
            ))
        if popt2 is not None and len(t_fit):
            qe2, k2 = popt2
            fig_qt.add_trace(go.Scatter(
                x=t_fit, y=model_second_order(t_fit, qe2, k2), mode="lines",
                name=f"{label} — Pseudo-2nd",
                line=dict(color=color, width=2, dash="dash"),
            ))

        # linearised 1st order
        if popt1 is not None and len(t_arr) >= 2:
            qe1, k1 = popt1
            sl1, ic1, qe_safe = linear_first_order(t_arr, q_arr, qe1)
            y_lin1  = np.log(qe_safe - q_arr)
            t_range = np.linspace(t_arr.min(), t_arr.max(), 200)
            fig_lin1.add_trace(go.Scatter(
                x=t_arr, y=y_lin1, mode="markers",
                name=label, marker=dict(color=color, size=8),
            ))
            if sl1 is not None:
                fig_lin1.add_trace(go.Scatter(
                    x=t_range, y=ic1 + sl1 * t_range, mode="lines",
                    line=dict(color=color, width=2), showlegend=False,
                ))

        # linearised 2nd order
        if popt2 is not None and len(t_arr) >= 2:
            mask2 = q_arr > 1e-12
            t_m2  = t_arr[mask2]
            if len(t_m2) >= 2:
                y_m2  = t_m2 / q_arr[mask2]
                sl2, ic2 = linear_second_order(t_arr, q_arr)
                t_range2 = np.linspace(t_m2.min(), t_m2.max(), 200)
                fig_lin2.add_trace(go.Scatter(
                    x=t_m2, y=y_m2, mode="markers",
                    name=label, marker=dict(color=color, size=8),
                ))
                if sl2 is not None:
                    fig_lin2.add_trace(go.Scatter(
                        x=t_range2, y=ic2 + sl2 * t_range2, mode="lines",
                        line=dict(color=color, width=2), showlegend=False,
                    ))

        # per-series parameter panel
        with st.expander(f"📊  Series: **{label}**", expanded=True):
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Pseudo-1st-order:**")
                if popt1 is not None:
                    qe1, k1 = popt1
                    st.markdown(f"- qₑ = **{qe1:.5f}** {q_unit}")
                    st.markdown(f"- k₁ = **{k1:.6f}** min⁻¹")
                    st.markdown(f"- R² = **{r2_1:.5f}**")
                    st.markdown(f"- MAE = **{mae1:.5f}** {q_unit}")
                else:
                    st.warning("Fit did not converge.")
            with pc2:
                st.markdown("**Pseudo-2nd-order:**")
                if popt2 is not None:
                    qe2, k2 = popt2
                    st.markdown(f"- qₑ = **{qe2:.5f}** {q_unit}")
                    st.markdown(f"- k₂ = **{k2:.6f}** {k2_unit(series_basis)}")
                    st.markdown(f"- R² = **{r2_2:.5f}**")
                    st.markdown(f"- MAE = **{mae2:.5f}** {q_unit}")
                else:
                    st.warning("Fit did not converge.")

        # Values at the most stable point (longest contact time) — the reference
        # the competitiveness indicators use for kinetics.
        q_last_mg  = float(q_mg[-1])  if len(q_mg)  else None
        Ct_last_mg = float(Ct_mg[-1]) if len(Ct_mg) else None

        res = {"label": label, "Metal": metal, "System": system,
               "Solution": sol, "N": N_val,
               "AdsCode": ads, "AdsConc": adsc, "parity": parity,
               "pH": ph_val,
               "basis": series_basis, "q_unit": q_unit, "c_unit": c_unit,
               "t": t_arr, "q": q_arr,
               "q_mg": q_mg, "Ct_mg": Ct_mg, "q_std_mg": q_std_mg,
               "q_last_mg": q_last_mg, "Ct_last_mg": Ct_last_mg,
               "t_last": float(t_arr[-1]) if len(t_arr) else None}
        if popt1 is not None:
            qe1, k1 = popt1
            res["1st"] = {"qe": qe1, "k1": k1, "r2": r2_1, "mae": mae1,
                          "qe_mg": float(popt1_mg[0]),
                          "mae_mg": mae1_mg}
        if popt2 is not None:
            qe2, k2 = popt2
            res["2nd"] = {"qe": qe2, "k2": k2, "r2": r2_2, "mae": mae2,
                          "qe_mg": float(popt2_mg[0]),
                          "k2_mg": float(popt2_mg[1]),
                          "mae_mg": mae2_mg}
        all_series_results.append(res)

    # ── Combined plots ─────────────────────────────────────────────────────────
    q_axis = cap_unit(basis)

    st.divider()
    st.subheader("q vs. time (all series)")
    fig_qt.update_layout(
        xaxis_title="t  (min)", yaxis_title=f"q  ({q_axis})",
        legend=dict(), template="plotly_white", height=520,
    )
    st.plotly_chart(fig_qt, width='stretch')

    st.subheader("Linearised Forms")
    lc1, lc2 = st.columns(2)
    with lc1:
        fig_lin1.update_layout(title="1st order: ln(qₑ − q) vs. t",
                               xaxis_title="t (min)", yaxis_title="ln(qₑ − q(t))",
                               template="plotly_white", height=380)
        st.plotly_chart(fig_lin1, width='stretch')
    with lc2:
        fig_lin2.update_layout(title="2nd order: t/q vs. t",
                               xaxis_title="t (min)",
                               yaxis_title=f"t / q  (min·g·{q_axis.split('/')[0]}⁻¹)",
                               template="plotly_white", height=380)
        st.plotly_chart(fig_lin2, width='stretch')

    # ── Grouped comparison plots (multi-element datasets) ──────────────────────
    kin_records = _build_records(all_series_results)

    if multi_solution:
        render_grouped_comparison(
            all_series_results, kin_records,
            x_field="t", y_field="q",
            x_title="t  (min)", y_title=f"q  ({q_axis})",
            key_prefix="kin",
        )


    # ══════════════════════════════════════════════════════════════════════════
    # Extra kinetic models — user-selectable
    # ══════════════════════════════════════════════════════════════════════════

    def _nth_order(t, qe, kn, n):
        t = np.asarray(t, dtype=float)
        n = float(n)
        if abs(n - 1.0) < 1e-4:
            return qe * (1.0 - np.exp(-kn * t))
        if abs(n - 2.0) < 1e-4:
            return kn * qe**2 * t / (1.0 + kn * qe * t)
        base = qe ** (1.0 - n) - kn * (1.0 - n) * t
        return qe - np.maximum(base, 1e-300) ** (1.0 / (1.0 - n))

    def _elovich(t, alpha, beta):
        return (1.0 / beta) * np.log(np.maximum(1.0 + alpha * beta * np.asarray(t, float), 1e-300))

    def _avrami(t, qe, kAV, nAV):
        return qe * (1.0 - np.exp(-np.maximum(kAV * np.asarray(t, float), 0.0) ** nAV))

    def _fit_extra_kin(fn, t, q, p0, bounds):
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score, mean_absolute_error as _mae
        try:
            popt, _ = curve_fit(fn, t, q, p0=p0, bounds=bounds, maxfev=30_000)
            pred = fn(t, *popt)
            return popt, float(r2_score(q, pred)), float(_mae(q, pred))
        except Exception:
            return None, float("nan"), float("nan")

    def _ipd_linear(t_arr, q_arr):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_absolute_error as _mae
        X = np.sqrt(t_arr).reshape(-1, 1)
        reg = LinearRegression().fit(X, q_arr)
        pred = reg.predict(X)
        return (float(reg.coef_[0]), float(reg.intercept_),
                float(r2_score(q_arr, pred)), float(_mae(q_arr, pred)))

    def _fit_segmented_ipd(t_arr, q_arr):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_absolute_error as _mae
        n = len(t_arr)
        if n < 5:
            return None
        sqt = np.sqrt(t_arr)
        best_sse, best = np.inf, None
        for sp in range(2, n - 2):
            r1  = LinearRegression().fit(sqt[:sp].reshape(-1, 1), q_arr[:sp])
            r2_ = LinearRegression().fit(sqt[sp:].reshape(-1, 1), q_arr[sp:])
            sse = (np.sum((q_arr[:sp]  - r1.predict(sqt[:sp].reshape(-1, 1))) ** 2) +
                   np.sum((q_arr[sp:]  - r2_.predict(sqt[sp:].reshape(-1, 1))) ** 2))
            if sse < best_sse:
                best_sse = sse
                best = (sp, float(r1.coef_[0]), float(r1.intercept_),
                        float(r2_.coef_[0]), float(r2_.intercept_))
        if best is None:
            return None
        sp, k1, c1, k2, c2 = best
        pred_all = np.concatenate([k1 * sqt[:sp] + c1, k2 * sqt[sp:] + c2])
        return {
            "k1": k1, "C1": c1, "k2": k2, "C2": c2,
            "t1": float(t_arr[sp]),
            "r2":  float(r2_score(q_arr, pred_all)),
            "mae": float(_mae(q_arr, pred_all)),
        }

    _EXTRA_KIN_MODELS = {
        "Pseudo-nth-order": {
            "latex":        r"q_t = q_e - \!\left(q_e^{1-n} - k_n(1-n)\,t\right)^{\!\frac{1}{1-n}}",
            "params_label": "qₑ, kₙ, n",
        },
        "Elovich": {
            "latex":        r"q_t = \frac{1}{\beta}\ln\!\left(1+\alpha\beta t\right)",
            "params_label": "α, β",
        },
        "Avrami fraccionary": {
            "latex":        r"q_t = q_e\!\left[1 - e^{-(k_{AV}t)^{n_{AV}}}\right]",
            "params_label": "qₑ, kAV, nAV",
        },
        "IntraParticle Diffusion (IPD)": {
            "latex":        r"q_t = k_{id}\,t^{1/2} + C",
            "params_label": "kid, C",
        },
        "Segmented IPD": {
            "latex":        r"q_t = k_1\,t^{1/2}+C_1\;(\text{phase 1}), \quad q_t = k_2\,t^{1/2}+C_2\;(\text{phase 2})",
            "params_label": "k₁, C₁, k₂, C₂, t₁",
        },
    }

    st.divider()
    st.subheader("🔬 Additional kinetics models")
    st.markdown(
        "**PFO** and **PSO** models are the default (parameters obtained above). "
        "Select here any additional model to fit:"
    )

    with st.expander("📋 Available additional models", expanded=False):
        for _mn, _mi in _EXTRA_KIN_MODELS.items():
            _ca, _cb = st.columns([1, 2])
            _ca.markdown(f"**{_mn}**")
            _ca.caption(f"parameters: {_mi['params_label']}")
            _cb.latex(_mi["latex"])

    selected_extra_kin = st.multiselect(
        "Additional models to fit:",
        options=list(_EXTRA_KIN_MODELS.keys()),
        default=st.session_state.get("_kin_extra_sel", []),
        key="_kin_extra_sel",
    )

    extra_kin_results = []

    if selected_extra_kin:
        st.markdown("---")
        for _si, _sr in series_keys.iterrows():
            _metal  = _sr["Metal"]
            _system = _sr["System"]
            _sol    = _sr["Solution"]
            _N      = _sr["N (mg/L)"]
            _ads    = _sr["AdsCode"]
            _adsc   = _sr["AdsConc (g/L)"]
            _par    = _sr["Parity"] if use_parity else None
            _ph_val = _sr["pH"] if multi_ph else None

            _mask = (
                (avg_df["System"]         == _system) &
                (avg_df["Metal"]          == _metal) &
                (avg_df["N (mg/L)"]       == _N)     &
                (avg_df["AdsCode"]        == _ads)   &
                (avg_df["AdsConc (g/L)"]  == _adsc)
            )
            if use_parity:
                _mask &= (avg_df["Parity"] == _par)
            if multi_ph:
                _mask &= (avg_df["pH"] == _ph_val)

            _sub   = avg_df[_mask].sort_values("t (min)")
            _tarr  = _sub["t (min)"].values.astype(float)

            _sc = _scale(_metal, basis)
            _series_basis = basis if _sc is not None else "mg"
            if _sc is None:
                _sc = 1.0
            # Optional models are fitted on the mass basis, exactly like PFO/PSO,
            # so their parameters never depend on the selected display basis.
            _qarr_mg = _sub["q (mg/g)"].values.astype(float)
            _qarr    = _qarr_mg * _sc
            _qunit = cap_unit(_series_basis)
            _qmass = _qunit.split("/")[0]

            def _disp(model_name, params_mg, _f=_sc):
                """Mass-basis parameters → active display basis."""
                return scale_kinetic_params(model_name, params_mg, _f)

            _sys_tag   = f" [{_system}]" if _sol != "Mono" else ""
            _ph_prefix = f"pH={_ph_val} | " if multi_ph else ""
            _par_sfx   = f" | {_par} reps" if use_parity else ""
            _slbl = (f"{_ph_prefix}{_metal}{_sys_tag} ({_sol}, N={_N}) | "
                     f"{_ads} {_adsc:.4g} g/L{_par_sfx}")

            _color  = series_color(_si)
            _qe_ref = max(_qarr_mg.max() * 1.1, 1e-6) if len(_qarr_mg) else 1.0
            _tplot  = np.linspace(max(_tarr.min() * 0.5, 0.1), _tarr.max() * 1.05, 500) if len(_tarr) else np.array([])

            with st.expander(f"📊  Series: **{_slbl}**", expanded=True):
                _fig_qt  = go.Figure()
                _fig_ipd = None

                if len(_tarr):
                    _fig_qt.add_trace(go.Scatter(
                        x=_tarr, y=_qarr, mode="markers", name="Data",
                        marker=dict(color=_color, size=9, symbol="circle"),
                    ))

                _has_non_ipd = any(
                    m not in ("IntraParticle Diffusion (IPD)", "Segmented IPD")
                    for m in selected_extra_kin
                )

                for _mn in selected_extra_kin:
                    _rec = {
                        "model":    _mn,
                        "label":    _slbl,
                        "System":   _system,
                        "Solution": _sol,
                        "Metal":    _metal,
                        "N":       _N,
                        "AdsCode": _ads,
                        "AdsConc": _adsc,
                    }

                    if _mn == "Pseudo-nth-order":
                        _popt, _r2, _mae_mg = _fit_extra_kin(
                            _nth_order, _tarr, _qarr_mg,
                            p0=[_qe_ref, 0.01, 1.5],
                            bounds=([0.0, 1e-9, 0.01], [np.inf, np.inf, 10.0]),
                        )
                        if _popt is not None:
                            _qe, _kn, _n = _popt
                            _rec["params_mg"] = {"qe": _qe, "kn": _kn, "n": _n,
                                                 "h": _kn * _qe ** _n}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if len(_tplot):
                                _fig_qt.add_trace(go.Scatter(
                                    x=_tplot, y=_nth_order(_tplot, _qe, _kn, _n) * _sc,
                                    mode="lines", name=_mn, line=dict(width=2),
                                ))
                            st.markdown(
                                f"**{_mn}** — qₑ=`{_p['qe']:.5f}` {_qunit} | kₙ=`{_p['kn']:.6f}` | "
                                f"n=`{_p['n']:.4f}` | h=`{_p['h']:.5f}` | R²=`{_r2:.5f}` | "
                                f"MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Elovich":
                        _popt, _r2, _mae_mg = _fit_extra_kin(
                            _elovich, _tarr, _qarr_mg,
                            p0=[max(_qarr_mg[0], 1e-3) if len(_qarr_mg) else 1.0, 1.0],
                            bounds=([1e-9, 1e-9], [np.inf, np.inf]),
                        )
                        if _popt is not None:
                            _a, _b = _popt
                            _rec["params_mg"] = {"alpha": _a, "beta": _b}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if len(_tplot):
                                _fig_qt.add_trace(go.Scatter(
                                    x=_tplot, y=_elovich(_tplot, _a, _b) * _sc,
                                    mode="lines", name=_mn, line=dict(width=2, dash="dot"),
                                ))
                            st.markdown(
                                f"**{_mn}** — α=`{_p['alpha']:.6f}` {_qmass}·g⁻¹·min⁻¹ | "
                                f"β=`{_p['beta']:.6f}` g·{_qmass}⁻¹ | "
                                f"R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Avrami fraccionary":
                        _popt, _r2, _mae_mg = _fit_extra_kin(
                            _avrami, _tarr, _qarr_mg,
                            p0=[_qe_ref, 0.1, 1.0],
                            bounds=([0.0, 1e-9, 0.01], [np.inf, np.inf, 10.0]),
                        )
                        if _popt is not None:
                            _qe, _kAV, _nAV = _popt
                            _rec["params_mg"] = {"qe": _qe, "kAV": _kAV, "nAV": _nAV}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if len(_tplot):
                                _fig_qt.add_trace(go.Scatter(
                                    x=_tplot, y=_avrami(_tplot, _qe, _kAV, _nAV) * _sc,
                                    mode="lines", name=_mn, line=dict(width=2, dash="dashdot"),
                                ))
                            st.markdown(
                                f"**{_mn}** — qₑ=`{_p['qe']:.5f}` {_qunit} | kAV=`{_p['kAV']:.6f}` | "
                                f"nAV=`{_p['nAV']:.4f}` | R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "IntraParticle Diffusion (IPD)":
                        if len(_tarr) >= 2:
                            _kid, _C, _r2, _mae_mg = _ipd_linear(_tarr, _qarr_mg)
                            _rec["params_mg"] = {"kid": _kid, "C": _C}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if _fig_ipd is None:
                                _fig_ipd = go.Figure()
                                _fig_ipd.add_trace(go.Scatter(
                                    x=np.sqrt(_tarr), y=_qarr, mode="markers",
                                    name="Data", marker=dict(color=_color, size=9),
                                ))
                            if len(_tplot):
                                _sq_plot = np.sqrt(_tplot)
                                _fig_ipd.add_trace(go.Scatter(
                                    x=_sq_plot, y=(_kid * _sq_plot + _C) * _sc,
                                    mode="lines", name=_mn, line=dict(width=2),
                                ))
                            st.markdown(
                                f"**{_mn}** — kid=`{_p['kid']:.6f}` {_qmass}·g⁻¹·min⁻½ | "
                                f"C=`{_p['C']:.5f}` {_qunit} | R²=`{_r2:.5f}` | "
                                f"MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: need ≥2 data points.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Segmented IPD":
                        _seg = _fit_segmented_ipd(_tarr, _qarr_mg)
                        if _seg:
                            _rec["params_mg"] = {k: v for k, v in _seg.items()
                                                 if k not in ("r2", "mae")}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _seg["mae"]
                            _rec["r2"], _rec["mae"] = _seg["r2"], _seg["mae"] * _sc
                            _p = _rec["params"]
                            _sp  = np.searchsorted(_tarr, _seg["t1"])
                            _sq1 = np.sqrt(_tarr[:_sp])
                            _sq2 = np.sqrt(_tarr[_sp:])
                            if _fig_ipd is None:
                                _fig_ipd = go.Figure()
                                _fig_ipd.add_trace(go.Scatter(
                                    x=np.sqrt(_tarr), y=_qarr, mode="markers",
                                    name="Data", marker=dict(color=_color, size=9),
                                ))
                            _fig_ipd.add_trace(go.Scatter(
                                x=_sq1, y=(_seg["k1"]*_sq1+_seg["C1"]) * _sc,
                                mode="lines", name="IPD Seg. — phase 1", line=dict(width=2),
                            ))
                            _fig_ipd.add_trace(go.Scatter(
                                x=_sq2, y=(_seg["k2"]*_sq2+_seg["C2"]) * _sc,
                                mode="lines", name="IPD Seg. — phase 2",
                                line=dict(width=2, dash="dash"),
                            ))
                            st.markdown(
                                f"**{_mn}** — k₁=`{_p['k1']:.5f}` | C₁=`{_p['C1']:.4f}` | "
                                f"k₂=`{_p['k2']:.5f}` | C₂=`{_p['C2']:.4f}` | "
                                f"t₁=`{_p['t1']:.2f}` min | R²=`{_seg['r2']:.5f}` | "
                                f"MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: need ≥5 data points per series.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    extra_kin_results.append(_rec)

                if _has_non_ipd and len(_tarr):
                    _fig_qt.update_layout(
                        xaxis_title="t  (min)", yaxis_title=f"q  ({_qunit})",
                        template="plotly_white", height=380, legend=dict(),
                    )
                    st.plotly_chart(_fig_qt, width='stretch')

                if _fig_ipd is not None:
                    _fig_ipd.update_layout(
                        title="IntraParticle Diffusion: q vs √t",
                        xaxis_title="√t  (min^½)", yaxis_title=f"q  ({_qunit})",
                        template="plotly_white", height=380,
                    )
                    st.plotly_chart(_fig_ipd, width='stretch')

    # ── Persist for Download tab ───────────────────────────────────────────────
    lin1_rows, lin2_rows = [], []
    for s in all_series_results:
        t_arr = s["t"]
        q_arr = s["q"]
        lbl   = s["label"]
        qe1_val = s.get("1st", {}).get("qe")
        for ti, qi in zip(t_arr, q_arr):
            lin2_rows.append({"Series": lbl, "t_min": ti,
                              "t_over_q": ti / qi if qi > 1e-12 else None})
            if qe1_val is not None:
                safe = max(qe1_val - qi, 1e-12)
                lin1_rows.append({"Series": lbl, "t_min": ti, "ln_qe_minus_q": np.log(safe)})

    qt_cols = ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)",
               "t (min)", "Cₜ (mg/L)", "Cₜ std (mg/L)", "%R", "%R std", "q (mg/g)", "q std"]
    st.session_state["kin_results"] = {
        "use_parity": use_parity,
        "files":     [e["name"] for e in kin_entries] +
                     [e["name"] for e in kin_m2_entries],
        "df":        raw_df,
        "avg_df":    avg_df,
        "series":    all_series_results,
        "records":   kin_records,
        "basis":     basis,
        "solutions": sorted(solutions_present),
        "csv_qt":    avg_df[[c for c in qt_cols if c in avg_df.columns]].copy(),
        "csv_lin1":  pd.DataFrame(lin1_rows),
        "csv_lin2":  pd.DataFrame(lin2_rows),
        "extra_models": extra_kin_results,
    }
