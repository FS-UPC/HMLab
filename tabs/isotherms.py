"""
tabs/isotherms.py — Isotherms tab (HMLab).

Filename convention: AI{n}{Metal1}{C1}[{Metal2}{C2}…]{AdsCode}{AdsConc}R{rep}.csv
  e.g. AI1Cu50HIB6p6R1.csv      → 1 metal Cu at N=50 mg/L, HIB 6.6 g/L, R=1
       AI2Cu30Mn50HIB6p6R1.csv  → 2 metals Cu@30 + Mn@50 mg/L, HIB 6.6 g/L, R=1

Each CSV file (XRF output) contains the measured equilibrium concentration Cₑ
and its standard deviation for each metal.  Calibration is NOT required because
the instrument output is already in concentration units (mg/L = ppm).

Grouping:
  Series key : (System, Metal, AdsCode, AdsConc)
  Per series : data points are (Cₑ, qₑ) averaged over replicates with the
               same (Metal, N, AdsCode, AdsConc).
  qₑ = (N − Cₑ) / AdsConc   [mg/g]

Units: everything is computed once on the mass basis (mg/L, mg/g).  The Base
selector (Mass / Mol / Charge) is a presentation lens — Cₑ and qₑ are scaled by
1/M (mol) or |z|/M (charge) and K_L scales inversely, which is exactly
equivalent to re-fitting in the chosen basis.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

from utils import (
    list_csv_files,
    read_hmlab_csv,
    read_hmlab_csv_bytes,
    parse_hmlab_filename,
)
from units import (
    MASS_BASIS,
    basis_factor,
    scale_isotherm_params,
    cap_unit,
    conc_unit,
    kf_unit,
    kl_unit,
    unit_selector,
    warn_missing_for_basis,
)
from indicators import solution_type
from mode2 import list_mode2_files, read_mode2_file
from plots import render_grouped_comparison

# ── Isotherm models ────────────────────────────────────────────────────────────

def langmuir(Ce, qmax, KL):
    return (qmax * KL * Ce) / (1.0 + KL * Ce)

def freundlich(Ce, KF, n):
    return KF * np.power(np.abs(Ce), 1.0 / n)

def _fit(model_fn, Ce, qe, p0, bounds):
    try:
        popt, _ = curve_fit(model_fn, Ce, qe, p0=p0,
                            maxfev=20_000, bounds=bounds)
        q_pred = model_fn(Ce, *popt)
        return popt, float(r2_score(qe, q_pred)), float(mean_absolute_error(qe, q_pred))
    except Exception:
        return None, None, None

# ── Colour palette ─────────────────────────────────────────────────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

def _color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def _scale(metal: str, basis: str) -> float | None:
    """
    Factor converting a mass-based quantity (mg/L, mg/g) of *metal* to
    *basis* (1 for mass, 1/M for mol, |z|/M for charge).  None when the
    required molar mass / ionic charge is unknown.
    """
    return basis_factor(metal, basis)


def _build_records(series_results: list[dict]) -> list[dict]:
    """
    Condense the fitted series into the mass-basis records the competitiveness
    indicators consume.

    For isotherms the reference capacity is q_max from the Langmuir fit, as
    specified by the method; Cₑ is the highest equilibrium concentration of the
    series, i.e. the point closest to saturation.
    """
    records = []
    for s in series_results:
        lang = s.get("langmuir", {})
        Ce_arr = s.get("Ce_mg")
        records.append({
            "metal":    s["Metal"],
            "system":   s["System"],
            "solution": s["Solution"],
            "label":    s["label"],
            "ads":      s["Ads"],
            "adsconc":  s["Cads"],
            "qe_mg":    lang.get("qmax_mg"),
            "Ce_mg":    float(np.max(Ce_arr)) if Ce_arr is not None and len(Ce_arr) else None,
            "KL_mg":    lang.get("KL_mg"),
            "RL":       lang.get("RL"),
            "k1":       None,
            "k2_mg":    None,
        })
    return records

# ── Main render ────────────────────────────────────────────────────────────────

def render(isotherms_dir: str) -> None:

    st.info(
        "ℹ️ Equilibrium concentrations (Cₑ) are read directly from the XRF CSV "
        "files, which are already calibrated in mg/L (= ppm)."
    )

    basis = unit_selector("isotherms")

    st.divider()

    # ── File selector ──────────────────────────────────────────────────────────
    st.subheader("📂 Input Files — Isotherms")
    st.caption(
        "Two input modes can be combined: **Mode 1** — one XRF CSV per "
        "experimental point (`AI…R#.csv`); **Mode 2** — one Excel workbook "
        "per complete series (`SI<x><Metals>M<xx>[pH<y>]<Ads><conc>.xlsx`)."
    )

    source = st.radio("File source:", ["📁 Folder", "⬆️ Upload"],
                      horizontal=True, key="iso_source")

    iso_entries = []
    iso_m2_entries = []
    m2_errors = []

    if source == "📁 Folder":
        all_iso = list_csv_files(isotherms_dir, r"AI\d")
        all_m2 = list_mode2_files(isotherms_dir, "isotherm")
        if not all_iso and not all_m2:
            st.warning(
                f"No AI* CSV (Mode 1) nor SI* XLSX (Mode 2) files found "
                f"in **{isotherms_dir}**."
            )
            return
        iso_sel = []
        if all_iso:
            saved_iso = st.session_state.get("iso_selected_files", all_iso)
            iso_sel = st.multiselect(
                "Mode 1 — Isotherm point files (AI*.csv):",
                options=all_iso,
                default=[f for f in saved_iso if f in all_iso] or all_iso,
                key="iso_file_selector",
            )
            st.session_state["iso_selected_files"] = iso_sel
        m2_sel = []
        if all_m2:
            saved_m2 = st.session_state.get("iso_m2_selected_files", all_m2)
            m2_sel = st.multiselect(
                "Mode 2 — Isotherm series files (SI*.xlsx):",
                options=all_m2,
                default=[f for f in saved_m2 if f in all_m2] or all_m2,
                key="iso_m2_file_selector",
            )
            st.session_state["iso_m2_selected_files"] = m2_sel
        if not iso_sel and not m2_sel:
            st.info("Select at least one isotherm file.")
            return
        for fname in iso_sel:
            csv_data = read_hmlab_csv(os.path.join(isotherms_dir, fname))
            iso_entries.append({"name": fname, "csv_data": csv_data})
        for fname in m2_sel:
            try:
                data = read_mode2_file(os.path.join(isotherms_dir, fname), fname)
                iso_m2_entries.append({"name": fname, "data": data})
            except ValueError as exc:
                m2_errors.append(f"{fname}: {exc}")
    else:
        uploaded_iso = st.file_uploader(
            "Upload isotherm files — AI*.csv (Mode 1) and/or SI*.xlsx (Mode 2)",
            type=["csv", "xlsx"], accept_multiple_files=True, key="iso_uploader",
        )
        if not uploaded_iso:
            st.info("Upload at least one isotherm file.")
            return
        st.session_state["iso_selected_files"] = [f.name for f in uploaded_iso]
        for f in uploaded_iso:
            if f.name.lower().endswith(".xlsx"):
                try:
                    data = read_mode2_file(f, f.name)
                    iso_m2_entries.append({"name": f.name, "data": data})
                except ValueError as exc:
                    m2_errors.append(f"{f.name}: {exc}")
            else:
                csv_data = read_hmlab_csv_bytes(f)
                iso_entries.append({"name": f.name, "csv_data": csv_data})

    for msg in m2_errors:
        st.warning(f"Skipped Mode 2 file: {msg}")

    # ── Parse & compute (Cₑ, qₑ) for every file and every metal ──────────────
    raw_rows = []
    skipped  = []
    for e in iso_entries:
        meta = parse_hmlab_filename(e["name"])
        if meta is None:
            skipped.append(f"{e['name']} (cannot parse filename — expected AI prefix)")
            continue
        if meta.get("type") != "isotherm":
            skipped.append(f"{e['name']} (not an isotherm file — expected AI prefix)")
            continue
        if e["csv_data"] is None:
            skipped.append(f"{e['name']} (cannot read CSV)")
            continue
        csv_data = e["csv_data"]
        if len(csv_data) < len(meta["metals"]):
            skipped.append(
                f"{e['name']} (CSV has {len(csv_data)} metal column(s) "
                f"but filename specifies {len(meta['metals'])})"
            )
            continue
        system_label = "+".join(m["code"] for m in meta["metals"])
        sol_label    = solution_type(meta["n_metals"])
        for metal_idx, metal_info in enumerate(meta["metals"]):
            metal_code = metal_info["code"]
            N_metal    = metal_info["N"]
            Ce         = csv_data[metal_idx]["avg"]
            Ce_std     = csv_data[metal_idx]["std"]
            qe         = (N_metal - Ce) / meta["AdsConc"]
            raw_rows.append({
                "Filename":       e["name"],
                "pH":             meta.get("pH"),
                "System":         system_label,
                "Solution":       sol_label,
                "Metal":          metal_code,
                "N (mg/L)":       N_metal,
                "AdsCode":        meta["AdsCode"],
                "AdsConc (g/L)":  meta["AdsConc"],
                "R":              meta["R"],
                "Cₑ (mg/L)":     round(Ce, 4),
                "Cₑ std (mg/L)": round(Ce_std, 4),
                "%R":            round((N_metal - Ce) / N_metal * 100, 4) if N_metal > 1e-12 else None,
                "qₑ (mg/g)":     round(qe, 6),
            })

    # ── Mode 2 series files: one workbook already holds every (C0, rep) ───────
    for e in iso_m2_entries:
        meta = e["data"]["meta"]
        for w in e["data"]["warnings"]:
            st.warning(f"{e['name']}: {w}")
        system_label = "+".join(m["code"] for m in meta["metals"])
        sol_label    = solution_type(meta["n_metals"])
        for p in e["data"]["points"]:
            qe = (p["C0"] - p["C"]) / meta["AdsConc"]
            raw_rows.append({
                "Filename":       e["name"],
                "pH":             meta.get("pH"),
                "System":         system_label,
                "Solution":       sol_label,
                "Metal":          p["metal"],
                "N (mg/L)":       p["C0"],
                "AdsCode":        meta["AdsCode"],
                "AdsConc (g/L)":  meta["AdsConc"],
                "R":              p["rep"],
                "Cₑ (mg/L)":     round(p["C"], 4),
                "Cₑ std (mg/L)": round(p["C_std"], 4),
                "%R":            round((p["C0"] - p["C"]) / p["C0"] * 100, 4) if p["C0"] > 1e-12 else None,
                "qₑ (mg/g)":     round(qe, 6),
            })

    for s in skipped:
        st.warning(f"Skipped: {s}")

    if not raw_rows:
        st.error("No valid isotherm data could be processed.")
        return

    raw_df = (
        pd.DataFrame(raw_rows)
        .sort_values(["System", "Metal", "AdsCode", "AdsConc (g/L)", "N (mg/L)", "R"])
        .reset_index(drop=True)
    )

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
        st.dataframe(raw_df, width='stretch')

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

    # ── Average replicates ─────────────────────────────────────────────────────
    grp_avg    = ((["pH"] if multi_ph else []) +
                  ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)"])
    grp_series = ((["pH"] if multi_ph else []) +
                  ["System", "Solution", "Metal", "AdsCode", "AdsConc (g/L)"])

    avg_df = (
        raw_df.groupby(grp_avg, as_index=False)
        .agg(
            Ce_mean=("Cₑ (mg/L)",     "mean"),
            Ce_std =("Cₑ (mg/L)",     "std"),
            qe_mean=("qₑ (mg/g)",     "mean"),
            qe_std =("qₑ (mg/g)",     "std"),
            Reps   =("R",             "count"),
            R_mean =("%R",            "mean"),
            R_std  =("%R",            "std"),
        )
        .rename(columns={"Ce_mean": "Cₑ (mg/L)", "qe_mean": "qₑ (mg/g)",
                         "R_mean": "%R", "R_std": "%R std"})
        .sort_values((["pH"] if multi_ph else []) +
                     ["System", "Metal", "AdsCode", "AdsConc (g/L)", "N (mg/L)"])
        .reset_index(drop=True)
    )

    with st.expander("📋 Averaged data (mean per Metal × N × Ads × Cads group)", expanded=True):
        col_order_iso = ((["pH"] if multi_ph else []) +
                         ["System", "Solution", "Metal", "N (mg/L)", "AdsCode", "AdsConc (g/L)",
                          "Reps", "Cₑ (mg/L)", "Ce_std", "%R", "%R std", "qₑ (mg/g)", "qe_std"])
        st.dataframe(avg_df.round(5), width='stretch', column_order=col_order_iso)

    group_keys = (
        avg_df[grp_series]
        .drop_duplicates()
        .sort_values(grp_series)
        .reset_index(drop=True)
    )

    st.divider()
    st.subheader("Isotherm Model Fitting  —  per (Metal × Adsorbent × C_ads) series")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Langmuir:**")
        st.latex(r"q_e = \frac{q_{\max}\,K_L\,C_e}{1 + K_L\,C_e}, \qquad R_L=\frac{1}{1+K_L C_0}")
    with mc2:
        st.markdown("**Freundlich:**")
        st.latex(r"q_e = K_F\,C_e^{1/n}")

    fig_iso   = go.Figure()
    fig_lin_L = go.Figure()
    fig_lin_F = go.Figure()

    all_series_results = []

    for idx, grow in group_keys.iterrows():
        system_val = grow["System"]
        sol_val    = grow["Solution"]
        metal_code = grow["Metal"]
        ads_code   = grow["AdsCode"]
        cads_val   = grow["AdsConc (g/L)"]
        ph_val     = grow["pH"] if multi_ph else None

        mask_series = (
            (avg_df["System"]        == system_val) &
            (avg_df["Metal"]         == metal_code) &
            (avg_df["AdsCode"]       == ads_code)   &
            (avg_df["AdsConc (g/L)"] == cads_val)
        )
        if multi_ph:
            mask_series &= (avg_df["pH"] == ph_val)
        sub = avg_df[mask_series].sort_values("Cₑ (mg/L)")

        C0_ref_mg = float(sub["N (mg/L)"].max())
        Ce_mg_all = sub["Cₑ (mg/L)"].values.astype(float)
        qe_mg_all = sub["qₑ (mg/g)"].values.astype(float)
        q_std_mg  = sub["qe_std"].fillna(0).values.astype(float)

        # Unit lens: fall back to the mass basis when the molar mass is unknown
        sc = _scale(metal_code, basis)
        series_basis = basis if sc is not None else "mg"
        if sc is None:
            sc = 1.0

        C0_ref = C0_ref_mg * sc
        Ce     = Ce_mg_all * sc
        qe     = qe_mg_all * sc
        q_std  = q_std_mg  * sc

        q_unit = cap_unit(series_basis)
        c_unit = conc_unit(series_basis)

        mask   = (Ce > 0) & (qe > 0)
        Ce_m, qe_m = Ce[mask], qe[mask]
        Ce_m_mg, qe_m_mg = Ce_mg_all[mask], qe_mg_all[mask]

        color = _color(idx)
        ph_prefix = f"pH={ph_val} | " if multi_ph else ""
        sys_tag = f" [{system_val}]" if sol_val != "Mono" else ""
        label = (f"{ph_prefix}{metal_code}{sys_tag} ({sol_val}) | "
                 f"{ads_code} {cads_val:.4g} g/L")

        # ── Fits ──────────────────────────────────────────────────────────────
        # Always fitted on the mass basis, then converted, so that the numbers
        # never depend on which unit is selected.  Fitting the rescaled data
        # instead would be algebraically equivalent but not numerically so:
        # an ill-conditioned series can land on a different local optimum.
        qmax_guess = max(qe_m_mg.max() * 1.2, 1e-6) if len(qe_m_mg) else 25.0

        popt_L_mg, r2_L, mae_L_mg = _fit(
            langmuir, Ce_m_mg, qe_m_mg,
            p0=[qmax_guess, 1.0],
            bounds=([0, 0], [np.inf, np.inf]),
        )
        popt_F_mg, r2_F, mae_F_mg = _fit(
            freundlich, Ce_m_mg, qe_m_mg,
            p0=[10.0, 2.0],
            bounds=([0, 1e-6], [np.inf, np.inf]),
        )

        # q_max scales with the basis; K_L is a reciprocal concentration so it
        # scales inversely.  Freundlich: n is invariant and, from
        # sc·q = K_F·(Ce/sc)^(1/n), K_F scales by sc^(1−1/n).
        popt_L = ([popt_L_mg[0] * sc, popt_L_mg[1] / sc]
                  if popt_L_mg is not None else None)
        popt_F = ([popt_F_mg[0] * sc ** (1.0 - 1.0 / popt_F_mg[1]), popt_F_mg[1]]
                  if popt_F_mg is not None else None)
        mae_L = None if mae_L_mg is None else mae_L_mg * sc
        mae_F = None if mae_F_mg is None else mae_F_mg * sc

        Ce_fit = (np.linspace(max(Ce_m.min() * 0.5, 1e-3), Ce_m.max() * 1.1, 400)
                  if len(Ce_m) else np.array([]))

        # scatter with error bars
        has_err = bool((q_std > 0).any())
        fig_iso.add_trace(go.Scatter(
            x=Ce, y=qe,
            error_y=dict(type="data", array=q_std, visible=has_err),
            mode="markers",
            name=f"{label} — Data",
            marker=dict(color=color, size=9, symbol="circle"),
        ))
        if popt_L is not None and len(Ce_fit):
            fig_iso.add_trace(go.Scatter(
                x=Ce_fit, y=langmuir(Ce_fit, *popt_L), mode="lines",
                name=f"{label} — Langmuir",
                line=dict(color=color, width=2),
            ))
        if popt_F is not None and len(Ce_fit):
            fig_iso.add_trace(go.Scatter(
                x=Ce_fit, y=freundlich(Ce_fit, *popt_F), mode="lines",
                name=f"{label} — Freundlich",
                line=dict(color=color, width=2, dash="dash"),
            ))

        # Langmuir linearised: Cₑ/qₑ vs Cₑ
        if len(Ce_m) >= 2:
            y_LL  = Ce_m / qe_m
            reg_L = LinearRegression().fit(Ce_m.reshape(-1, 1), y_LL)
            sl_L, ic_L = float(reg_L.coef_[0]), float(reg_L.intercept_)
            Ce_r  = np.linspace(Ce_m.min(), Ce_m.max(), 200)
            fig_lin_L.add_trace(go.Scatter(x=Ce_m, y=y_LL, mode="markers",
                                           name=label, marker=dict(color=color, size=8)))
            fig_lin_L.add_trace(go.Scatter(x=Ce_r, y=ic_L + sl_L * Ce_r, mode="lines",
                                           line=dict(color=color, width=2), showlegend=False))

        # Freundlich linearised: ln(qₑ) vs ln(Cₑ)
        if len(Ce_m) >= 2:
            log_Ce = np.log(Ce_m)
            log_qe = np.log(qe_m)
            reg_F  = LinearRegression().fit(log_Ce.reshape(-1, 1), log_qe)
            sl_F, ic_F = float(reg_F.coef_[0]), float(reg_F.intercept_)
            lCe_r  = np.linspace(log_Ce.min(), log_Ce.max(), 200)
            fig_lin_F.add_trace(go.Scatter(x=log_Ce, y=log_qe, mode="markers",
                                           name=label, marker=dict(color=color, size=8)))
            fig_lin_F.add_trace(go.Scatter(x=lCe_r, y=ic_F + sl_F * lCe_r, mode="lines",
                                           line=dict(color=color, width=2), showlegend=False))

        # ── Per-series parameter panel ─────────────────────────────────────────
        n_pts       = len(Ce_m)
        n_reps_info = sub["Reps"].sum() if "Reps" in sub.columns else "?"
        with st.expander(
            f"📊  Series: **{label}**  ({n_pts} data points, {int(n_reps_info)} total replicas)",
            expanded=True,
        ):
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Langmuir:**")
                if popt_L is not None:
                    qmax_f, KL_f = popt_L
                    RL = 1.0 / (1.0 + KL_f * C0_ref)
                    st.markdown(f"- q_max = **{qmax_f:.5f}** {q_unit}")
                    st.markdown(f"- K_L   = **{KL_f:.6f}** {kl_unit(series_basis)}")
                    st.markdown(f"- R²    = **{r2_L:.5f}**")
                    st.markdown(f"- MAE   = **{mae_L:.5f}** {q_unit}")
                    st.markdown(f"- **R_L = {RL:.5f}**  (C₀ = {C0_ref:.4g} {c_unit})")
                    if RL > 1:
                        st.error("R_L > 1 → unfavorable adsorption")
                    elif abs(RL - 1) < 1e-4:
                        st.warning("R_L ≈ 1 → linear")
                    elif RL > 0:
                        st.success("0 < R_L < 1 → ✅ favorable adsorption")
                    else:
                        st.info("R_L ≈ 0 → irreversible")
                else:
                    st.warning("Langmuir fit did not converge.")
            with pc2:
                st.markdown("**Freundlich:**")
                if popt_F is not None:
                    KF_f, n_f = popt_F
                    inv_n = 1.0 / n_f
                    st.markdown(f"- K_F = **{KF_f:.5f}**  {kf_unit(series_basis)}")
                    st.markdown(f"- n   = **{n_f:.5f}**")
                    st.markdown(f"- 1/n = **{inv_n:.5f}**")
                    st.markdown(f"- R²  = **{r2_F:.5f}**")
                    st.markdown(f"- MAE = **{mae_F:.5f}** {q_unit}")
                    if inv_n < 1:
                        st.success("1/n < 1 → ✅ favorable adsorption")
                    elif abs(inv_n - 1) < 1e-3:
                        st.warning("1/n ≈ 1 → linear")
                    else:
                        st.error("1/n > 1 → unfavorable adsorption")
                else:
                    st.warning("Freundlich fit did not converge.")

        res = {
            "label":    label,
            "System":   system_val,
            "Solution": sol_val,
            "Metal":    metal_code,
            "Ads":      ads_code,
            "Cads":     cads_val,
            "Ce":       Ce_m,
            "qe":       qe_m,
            "Ce_mg":    Ce_m_mg,
            "qe_mg":    qe_m_mg,
            "pH":       ph_val,
            "C0_ref":   C0_ref,
            "C0_ref_mg": C0_ref_mg,
            "basis":    series_basis,
            "q_unit":   q_unit,
            "c_unit":   c_unit,
        }
        if popt_L is not None:
            qmax_f, KL_f = popt_L
            res["langmuir"] = {
                "qmax": qmax_f, "KL": KL_f,
                "RL": 1.0 / (1.0 + KL_f * C0_ref),
                "r2": r2_L, "mae": mae_L,
                # Mass-basis values feed the competitiveness indicators
                # and the basis-aware exports
                "qmax_mg": float(popt_L_mg[0]), "KL_mg": float(popt_L_mg[1]),
                "mae_mg": mae_L_mg, "C0_ref_mg": C0_ref_mg,
            }
        if popt_F is not None:
            KF_f, n_f = popt_F
            res["freundlich"] = {
                "KF": KF_f, "n": n_f, "inv_n": 1.0 / n_f,
                "r2": r2_F, "mae": mae_F,
                "KF_mg": float(popt_F_mg[0]), "mae_mg": mae_F_mg,
            }
        all_series_results.append(res)

    # ── Combined plots ─────────────────────────────────────────────────────────
    q_axis = cap_unit(basis)
    c_axis = conc_unit(basis)

    st.divider()
    st.subheader("Isotherm curves (all series)")
    fig_iso.update_layout(
        xaxis_title=f"Cₑ ({c_axis})", yaxis_title=f"qₑ ({q_axis})",
        template="plotly_white", height=480,
    )
    st.plotly_chart(fig_iso, width='stretch')

    st.subheader("Linearised Forms")
    lc1, lc2 = st.columns(2)
    with lc1:
        fig_lin_L.update_layout(
            title="Langmuir: Cₑ/qₑ vs Cₑ",
            xaxis_title=f"Cₑ ({c_axis})", yaxis_title="Cₑ / qₑ  (g/L)",
            template="plotly_white", height=360,
        )
        st.plotly_chart(fig_lin_L, width='stretch')
    with lc2:
        fig_lin_F.update_layout(
            title="Freundlich: ln(qₑ) vs ln(Cₑ)",
            xaxis_title="ln(Cₑ)", yaxis_title="ln(qₑ)",
            template="plotly_white", height=360,
        )
        st.plotly_chart(fig_lin_F, width='stretch')

    # ── Grouped comparison plots (multi-element datasets) ──────────────────────
    iso_records = _build_records(all_series_results)

    if multi_solution:
        render_grouped_comparison(
            all_series_results, iso_records,
            x_field="Ce", y_field="qe",
            x_title=f"Cₑ  ({c_axis})", y_title=f"qₑ  ({q_axis})",
            key_prefix="iso",
        )


    # ══════════════════════════════════════════════════════════════════════════
    # Extra isotherm models — user-selectable
    # ══════════════════════════════════════════════════════════════════════════

    def _sips(Ce, qm, KS, ne):
        Ce = np.asarray(Ce, float)
        return qm * (KS * Ce) ** ne / (1.0 + (KS * Ce) ** ne)

    def _toth(Ce, qm, KT, tp):
        Ce = np.asarray(Ce, float)
        return qm * KT * Ce / (1.0 + (KT * Ce) ** tp) ** (1.0 / tp)

    def _halsey(Ce, KH, nH):
        Ce = np.asarray(Ce, float)
        inner = np.log(1.0 / np.maximum(Ce, 1e-12))
        return np.where(inner > 1e-12,
                        (KH / np.maximum(inner, 1e-12)) ** (1.0 / nH),
                        np.nan)

    def _harkins_jura(Ce, A, B):
        Ce = np.asarray(Ce, float)
        inner = B / A - np.log10(np.maximum(Ce, 1e-12)) / A
        return np.where(inner > 1e-12,
                        1.0 / np.sqrt(np.maximum(inner, 1e-12)),
                        np.nan)

    def _janovics(Ce, qmax, KJ):
        Ce = np.asarray(Ce, float)
        return qmax * (1.0 - np.exp(-KJ * Ce))

    def _fit_extra_iso(fn, Ce, qe, p0, bounds):
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score, mean_absolute_error as _mae
        try:
            trial = np.asarray(fn(Ce, *p0), float)
            valid = np.isfinite(trial) & np.isfinite(qe)
            if valid.sum() < 2:
                return None, float("nan"), float("nan")
            Ce_v, qe_v = Ce[valid], qe[valid]
            popt, _ = curve_fit(fn, Ce_v, qe_v, p0=p0, bounds=bounds, maxfev=30_000)
            pred = fn(Ce_v, *popt)
            return popt, float(r2_score(qe_v, pred)), float(_mae(qe_v, pred))
        except Exception:
            return None, float("nan"), float("nan")

    _EXTRA_ISO_MODELS = {
        "Sips (Langmuir-Freundlich)": {
            "latex":        r"q_e = \frac{q_m(K_S C_e)^{n_e}}{1+(K_S C_e)^{n_e}}",
            "params_label": "qₘ, Kₛ, nₑ",
        },
        "Tóth": {
            "latex":        r"q_e = \frac{q_m K_T C_e}{\bigl[1+(K_T C_e)^t\bigr]^{1/t}}",
            "params_label": "qₘ, K_T, t",
        },
        "Halsey": {
            "latex":        r"q_e = \!\left(\frac{K_H}{\ln(1/C_e)}\right)^{\!1/n_H}",
            "params_label": "K_H, n_H",
            "note":         "Only valid for Cₑ < 1 mg/L; points out of range are omitted.",
        },
        "Harkins-Jura": {
            "latex":        r"\frac{1}{q_e^2}=\frac{B}{A}-\frac{1}{A}\log C_e",
            "params_label": "A, B",
        },
        "Janovics": {
            "latex":        r"q_e = q_{max}\!\left(1-e^{-K_J C_e}\right)",
            "params_label": "qₘₐₓ, K_J",
        },
    }

    st.divider()
    st.subheader("🔬 Additional Isotherm Models")
    st.markdown(
        "**Langmuir** and **Freundlich** models are fitted above. "
        "Select additional models to fit here:"
    )

    with st.expander("📋 Available additional models", expanded=False):
        for _mn, _mi in _EXTRA_ISO_MODELS.items():
            _ca, _cb = st.columns([1, 2])
            _ca.markdown(f"**{_mn}**")
            _ca.caption(f"parameters: {_mi['params_label']}")
            if "note" in _mi:
                _ca.caption(f"⚠️ {_mi['note']}")
            _cb.latex(_mi["latex"])

    selected_extra_iso = st.multiselect(
        "Additional models to fit:",
        options=list(_EXTRA_ISO_MODELS.keys()),
        default=st.session_state.get("_iso_extra_sel", []),
        key="_iso_extra_sel",
    )

    extra_iso_results = []

    if selected_extra_iso:
        st.markdown("---")
        for _gi, _gr in group_keys.iterrows():
            _system  = _gr["System"]
            _sol     = _gr["Solution"]
            _metal   = _gr["Metal"]
            _ads     = _gr["AdsCode"]
            _cads    = _gr["AdsConc (g/L)"]
            _ph_val  = _gr["pH"] if multi_ph else None

            _mask = (
                (avg_df["System"]        == _system) &
                (avg_df["Metal"]         == _metal) &
                (avg_df["AdsCode"]       == _ads)   &
                (avg_df["AdsConc (g/L)"] == _cads)
            )
            if multi_ph:
                _mask &= (avg_df["pH"] == _ph_val)
            _sub  = avg_df[_mask].sort_values("Cₑ (mg/L)")

            _sc = _scale(_metal, basis)
            _series_basis = basis if _sc is not None else "mg"
            if _sc is None:
                _sc = 1.0
            _qunit = cap_unit(_series_basis)
            _cunit = conc_unit(_series_basis)

            # Optional models are fitted on the mass basis, exactly like the
            # Langmuir/Freundlich pair, so their parameters never depend on the
            # selected display basis.
            _Ce_mg_all = _sub["Cₑ (mg/L)"].values.astype(float)
            _qe_mg_all = _sub["qₑ (mg/g)"].values.astype(float)
            _Ce   = _Ce_mg_all * _sc
            _qe   = _qe_mg_all * _sc
            _vmask = (_Ce_mg_all > 0) & (_qe_mg_all > 0)
            _Ce_m, _qe_m = _Ce[_vmask], _qe[_vmask]
            _Ce_m_mg, _qe_m_mg = _Ce_mg_all[_vmask], _qe_mg_all[_vmask]

            def _disp(model_name, params_mg, _f=_sc):
                """Mass-basis parameters → active display basis."""
                return scale_isotherm_params(model_name, params_mg, _f)

            _ph_prefix = f"pH={_ph_val} | " if multi_ph else ""
            _sys_tag = f" [{_system}]" if _sol != "Mono" else ""
            _label = (f"{_ph_prefix}{_metal}{_sys_tag} ({_sol}) | "
                      f"{_ads} {_cads:.4g} g/L")
            color_val = _color(_gi)
            _qm_ref   = max(_qe_m_mg.max() * 1.2, 1e-6) if len(_qe_m_mg) else 25.0
            _Ce_plot  = (np.linspace(max(_Ce_m.min() * 0.5, 1e-3),
                                     _Ce_m.max() * 1.1, 400)
                         if len(_Ce_m) else np.array([]))
            _Ce_plot_mg = _Ce_plot / _sc if len(_Ce_plot) else _Ce_plot

            with st.expander(f"📊  Series: **{_label}**", expanded=True):
                _fig_ex = go.Figure()
                _fig_ex.add_trace(go.Scatter(
                    x=_Ce_m, y=_qe_m, mode="markers", name="Data",
                    marker=dict(color=color_val, size=9, symbol="circle"),
                ))

                for _mn in selected_extra_iso:
                    _rec = {
                        "model":    _mn,
                        "label":    _label,
                        "System":   _system,
                        "Solution": _sol,
                        "Metal":    _metal,
                        "Ads":    _ads,
                        "Cads":   _cads,
                    }

                    if _mn == "Sips (Langmuir-Freundlich)":
                        _popt, _r2, _mae_mg = _fit_extra_iso(
                            _sips, _Ce_m_mg, _qe_m_mg,
                            p0=[_qm_ref, 1.0, 1.0],
                            bounds=([0.0, 1e-9, 0.01], [np.inf, np.inf, 10.0]),
                        )
                        if _popt is not None:
                            _qm, _KS, _ne = _popt
                            _rec["params_mg"] = {"qm": _qm, "KS": _KS, "ne": _ne}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if len(_Ce_plot):
                                _fig_ex.add_trace(go.Scatter(
                                    x=_Ce_plot,
                                    y=_sips(_Ce_plot_mg, _qm, _KS, _ne) * _sc,
                                    mode="lines", name=_mn, line=dict(width=2),
                                ))
                            st.markdown(
                                f"**{_mn}** — qₘ=`{_p['qm']:.5f}` {_qunit} | "
                                f"Kₛ=`{_p['KS']:.6f}` {kl_unit(_series_basis)} | "
                                f"nₑ=`{_p['ne']:.4f}` | "
                                f"R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Tóth":
                        _popt, _r2, _mae_mg = _fit_extra_iso(
                            _toth, _Ce_m_mg, _qe_m_mg,
                            p0=[_qm_ref, 1.0, 1.0],
                            bounds=([0.0, 1e-9, 0.01], [np.inf, np.inf, 20.0]),
                        )
                        if _popt is not None:
                            _qm, _KT, _tp = _popt
                            _rec["params_mg"] = {"qm": _qm, "KT": _KT, "t": _tp}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if len(_Ce_plot):
                                _fig_ex.add_trace(go.Scatter(
                                    x=_Ce_plot,
                                    y=_toth(_Ce_plot_mg, _qm, _KT, _tp) * _sc,
                                    mode="lines", name=_mn,
                                    line=dict(width=2, dash="dot"),
                                ))
                            st.markdown(
                                f"**{_mn}** — qₘ=`{_p['qm']:.5f}` {_qunit} | "
                                f"K_T=`{_p['KT']:.6f}` | t=`{_p['t']:.4f}` | "
                                f"R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Halsey":
                        _h_mask = (_Ce_m_mg < 1.0) & (_Ce_m_mg > 0) & (_qe_m_mg > 0)
                        if _h_mask.sum() < 2:
                            st.warning(
                                f"**{_mn}**: not enough points with Cₑ < 1 mg/L "
                                f"(found: {int(_h_mask.sum())})."
                            )
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")
                        else:
                            _popt, _r2, _mae_mg = _fit_extra_iso(
                                _halsey, _Ce_m_mg[_h_mask], _qe_m_mg[_h_mask],
                                p0=[1.0, 2.0],
                                bounds=([1e-9, 0.01], [np.inf, 20.0]),
                            )
                            if _popt is not None:
                                _KH, _nH = _popt
                                # q = f(ln C): no exact reprojection, so the
                                # parameters stay on the mass basis
                                _rec["params_mg"] = {"KH": _KH, "nH": _nH}
                                _rec["params"] = dict(_rec["params_mg"])
                                _rec["mae_mg"] = _mae_mg
                                _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                                _Ce_h_mg = np.linspace(
                                    1e-3, min(_Ce_m_mg[_h_mask].max() * 1.1, 0.99), 200)
                                _fig_ex.add_trace(go.Scatter(
                                    x=_Ce_h_mg * _sc,
                                    y=_halsey(_Ce_h_mg, _KH, _nH) * _sc,
                                    mode="lines", name=f"{_mn} (Cₑ<1 mg/L)",
                                    line=dict(width=2, dash="dashdot"),
                                ))
                                st.markdown(
                                    f"**{_mn}** (fit for Cₑ<1 mg/L; K_H and n_H on "
                                    f"the mass basis) — "
                                    f"K_H=`{_KH:.6f}` | n_H=`{_nH:.4f}` | "
                                    f"R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                                )
                            else:
                                st.warning(f"**{_mn}**: fit did not converge.")
                                _rec["params"] = _rec["params_mg"] = {}
                                _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Harkins-Jura":
                        _popt, _r2, _mae_mg = _fit_extra_iso(
                            _harkins_jura, _Ce_m_mg, _qe_m_mg,
                            p0=[10.0, 1.0],
                            bounds=([1e-9, -np.inf], [np.inf, np.inf]),
                        )
                        if _popt is not None:
                            _A, _B = _popt
                            # 1/q² = B/A − (1/A)·log C: no exact reprojection,
                            # so the parameters stay on the mass basis
                            _rec["params_mg"] = {"A": _A, "B": _B}
                            _rec["params"] = dict(_rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            if len(_Ce_plot):
                                _y_hj = _harkins_jura(_Ce_plot_mg, _A, _B) * _sc
                                _vhj  = np.isfinite(_y_hj)
                                if _vhj.any():
                                    _fig_ex.add_trace(go.Scatter(
                                        x=_Ce_plot[_vhj], y=_y_hj[_vhj],
                                        mode="lines", name=_mn,
                                        line=dict(width=2, dash="longdash"),
                                    ))
                            st.markdown(
                                f"**{_mn}** (A and B on the mass basis) — "
                                f"A=`{_A:.6f}` | B=`{_B:.6f}` | "
                                f"R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    elif _mn == "Janovics":
                        _popt, _r2, _mae_mg = _fit_extra_iso(
                            _janovics, _Ce_m_mg, _qe_m_mg,
                            p0=[_qm_ref, 0.1],
                            bounds=([0.0, 1e-9], [np.inf, np.inf]),
                        )
                        if _popt is not None:
                            _qmax, _KJ = _popt
                            _rec["params_mg"] = {"qmax": _qmax, "KJ": _KJ}
                            _rec["params"], _ = _disp(_mn, _rec["params_mg"])
                            _rec["mae_mg"] = _mae_mg
                            _rec["r2"], _rec["mae"] = _r2, _mae_mg * _sc
                            _p = _rec["params"]
                            if len(_Ce_plot):
                                _fig_ex.add_trace(go.Scatter(
                                    x=_Ce_plot,
                                    y=_janovics(_Ce_plot_mg, _qmax, _KJ) * _sc,
                                    mode="lines", name=_mn, line=dict(width=2),
                                ))
                            st.markdown(
                                f"**{_mn}** — qₘₐₓ=`{_p['qmax']:.5f}` {_qunit} | "
                                f"K_J=`{_p['KJ']:.6f}` {kl_unit(_series_basis)} | "
                                f"R²=`{_r2:.5f}` | MAE=`{_rec['mae']:.5f}`"
                            )
                        else:
                            st.warning(f"**{_mn}**: fit did not converge.")
                            _rec["params"] = _rec["params_mg"] = {}
                            _rec["r2"] = _rec["mae"] = _rec["mae_mg"] = float("nan")

                    extra_iso_results.append(_rec)

                _fig_ex.update_layout(
                    xaxis_title=f"Cₑ  ({_cunit})", yaxis_title=f"qₑ  ({_qunit})",
                    template="plotly_white", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                font=dict(size=11)),
                )
                st.plotly_chart(_fig_ex, width='stretch')

    # ── Persist for Download tab ───────────────────────────────────────────────
    iso_lin_L_rows, iso_lin_F_rows = [], []
    for s in all_series_results:
        Ce_m = s["Ce"]
        qe_m = s["qe"]
        lbl  = s["label"]
        for c, q in zip(Ce_m, qe_m):
            iso_lin_L_rows.append({"Series": lbl, "Ce_mg_per_L": c,
                                   "Ce_over_qe": c / q if q > 1e-12 else None})
            if c > 0 and q > 0:
                iso_lin_F_rows.append({"Series": lbl,
                                       "ln_Ce": np.log(c), "ln_qe": np.log(q)})

    _csv_iso_cols = ((["pH"] if multi_ph else []) +
                     ["System", "Solution", "Metal", "AdsCode", "AdsConc (g/L)", "N (mg/L)",
                      "Reps", "Cₑ (mg/L)", "Ce_std", "%R", "%R std", "qₑ (mg/g)", "qe_std"])
    st.session_state["iso_results"] = {
        "files":         [e["name"] for e in iso_entries] +
                         [e["name"] for e in iso_m2_entries],
        "raw_df":        raw_df,
        "avg_df":        avg_df,
        "series":        all_series_results,
        "records":       iso_records,
        "basis":         basis,
        "solutions":     sorted(solutions_present),
        "csv_iso":       avg_df[[c for c in _csv_iso_cols if c in avg_df.columns]].copy(),
        "csv_lin_lang":  pd.DataFrame(iso_lin_L_rows),
        "csv_lin_freun": pd.DataFrame(iso_lin_F_rows),
        "extra_models":  extra_iso_results,
    }
