"""
tabs/download.py — Download Results tab.

Assembles a human-readable TXT report from session_state and offers it
as a Streamlit download button.  Also provides per-dataset CSV downloads,
including extra kinetic / isotherm models when selected by the user.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from exports import (
    indicator_tables,
    kinetic_reference_note,
    non_exact_models,
    tables_for_basis,
    three_bases_tables,
)
from indicators import KIN_REF_FIT
from units import (
    BASIS_ORDER,
    MASS_BASIS,
    basis_factor,
    basis_name,
    cap_unit,
    conc_unit,
    k2_unit,
    kf_unit,
    kl_unit,
    resolve_basis,
    unit_selector,
)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _section(title: str) -> str:
    bar = "=" * 60
    return f"\n{bar}\n  {title}\n{bar}\n"


def _subsection(title: str) -> str:
    return f"\n--- {title} ---\n"


def _format_pretty_csv(df: pd.DataFrame) -> bytes:
    """
    Returns a fixed-width, human-readable text table.
    - Floats are rounded to 4 decimal places.
    - Every column is padded so header and values are left-aligned.
    """
    # Round all numeric columns to 4 decimal places
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].round(4)

    # Convert everything to strings so we can measure widths
    str_df = df.astype(str)

    # Column width = max of (header length, longest value in column)
    col_widths = {
        col: max(len(col), int(str_df[col].str.len().fillna(0).max()))
        for col in str_df.columns
    }
    #col_widths = {
    #    col: max(len(col), int(str_df[col].str.len().max() or 0))
    #    for col in str_df.columns
    #}

    def pad_row(values):
        return "  ".join(str(v).ljust(col_widths[col]) for col, v in zip(str_df.columns, values))

    header = pad_row(str_df.columns)
    separator = "  ".join("-" * col_widths[col] for col in str_df.columns)
    rows = [pad_row(row) for _, row in str_df.iterrows()]

    table = "\n".join([header, separator] + rows)
    return table.encode("utf-8")


def _csv_button(label: str, df, filename: str) -> None:
    """Renders two side-by-side download buttons for the same dataset:
    - Left  : pretty fixed-width text table  (*_pretty.csv)
    - Right : standard comma-separated CSV   (*.csv)
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        st.caption(f"_{label}: no data yet_")
        return

    # Derive pretty filename by inserting "_pretty" before the extension
    base, _ext = filename.rsplit(".", 1) if "." in filename else (filename, "csv")
    pretty_filename = f"{base}_pretty.{_ext}"

    col_pretty, col_std = st.columns(2)
    with col_pretty:
        st.download_button(
            label=f"⬇️  {label} (formatted)",
            data=_format_pretty_csv(df),
            file_name=pretty_filename,
            mime="text/plain",
        )
    with col_std:
        st.download_button(
            label=f"⬇️  {label} (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )

def _tri(value_mg, metal: str, unit_fn, fmt: str = "{:.6f}",
         inverse: bool = False) -> str:
    """
    A mass-basis quantity written on all three bases, e.g.
    ``73.000000 mg/g | 0.352317 mmol/g | 0.704634 meq/g``.

    *inverse* is for quantities that scale the other way (k₂, K_L), and a
    basis the element cannot be converted to is shown as ``n/a``.
    """
    if value_mg is None:
        return "n/a"
    parts = []
    for b in BASIS_ORDER:
        f = basis_factor(metal, b)
        unit = unit_fn(b)
        if f is None:
            parts.append(f"n/a {unit}")
        else:
            v = value_mg / f if inverse else value_mg * f
            parts.append(f"{fmt.format(v)} {unit}")
    return "  |  ".join(parts)


def _row_factor(basis: str):
    """Return metal → conversion factor for an already-resolved *basis*."""
    def factor(metal: str) -> float:
        f = basis_factor(metal, basis)
        return 1.0 if f is None else f
    return factor


def _basis_line(basis: str, blocked: list[str]) -> str:
    """The 'units are …' line printed above each converted table."""
    txt = (f"  Units: {conc_unit(basis)} for concentrations, "
           f"{cap_unit(basis)} for capacities.")
    if blocked:
        txt += (f"  ({', '.join(blocked)} cannot be converted, so this table "
                f"stays on the mass basis.)")
    return txt


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(basis: str = MASS_BASIS) -> str:
    """
    Return the full report as plain text.

    Measured data tables and fitted parameters are written on *basis*; a table
    whose elements cannot all be converted falls back to the mass basis and
    says so, so a single table never mixes units.
    """
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    lines.append("HMLab — HEAVY-METAL ADSORPTION ANALYSIS — RESULTS REPORT")
    lines.append(f"Generated: {ts}")
    lines.append(f"Reporting basis: {basis_name(basis).upper()} "
                 f"(concentrations in {conc_unit(basis)}, "
                 f"capacities in {cap_unit(basis)}).")
    lines.append("Fitted parameters are additionally given on all three bases — "
                 "Mass (mg) | Mol (mmol) | Charge (meq).")
    lines.append("Competitiveness indicators are dimensionless (basis-independent).")

    # ── Kinetics ───────────────────────────────────────────────────────────────
    lines.append(_section("KINETICS"))

    kin = st.session_state.get("kin_results", {})

    lines.append(_subsection("Parameters"))
    if kin:
        lines.append(f"  Formula : q(t) = (N − Cₜ) / AdsConc  [{cap_unit(basis)}]")
        lines.append("  N and AdsConc are read from each filename.")
    else:
        lines.append("  (kinetics tab not yet processed)")

    lines.append(_subsection("Processed data (t, Ct, q(t))"))
    df = kin.get("df")
    if df is not None and not df.empty:
        _b, _blocked = resolve_basis(df.get("Metal", []), basis)
        _f = _row_factor(_b)
        _cu, _qu = conc_unit(_b), cap_unit(_b)
        lines.append(_basis_line(_b, _blocked))
        lines.append(f"  {'Filename':<35} {'Metal':>6} {'t (min)':>8}"
                     f" {'Ct (' + _cu + ')':>14} {'q(t) (' + _qu + ')':>16}")
        lines.append("  " + "-" * 84)
        for _, row in df.iterrows():
            k = _f(row.get("Metal", ""))
            lines.append(
                f"  {row['Filename']:<35} {str(row.get('Metal','')):>6} "
                f"{row['t (min)']:>8.1f} "
                f"{row['Cₜ (mg/L)'] * k:>14.5f} {row['q(t) (mg/g)'] * k:>16.6f}"
            )
    else:
        lines.append("  (no data)")

    lines.append(_subsection("Averaged data (mean Cₜ and q per series)"))
    avg_df_rep = kin.get("avg_df")
    if avg_df_rep is not None and not avg_df_rep.empty:
        _b, _blocked = resolve_basis(avg_df_rep.get("Metal", []), basis)
        _f = _row_factor(_b)
        _cu, _qu = conc_unit(_b), cap_unit(_b)
        lines.append(_basis_line(_b, _blocked))
        lines.append(
            f"  {'Metal':>6} {'N (' + _cu + ')':>13} {'AdsCode':>10} {'AdsConc':>8}"
            f" {'t (min)':>8} {'Reps':>5}"
            f" {'Cₜ (' + _cu + ')':>14} {'Cₜ std':>12}"
            f" {'q (' + _qu + ')':>14} {'q std':>12}"
        )
        lines.append("  " + "-" * 112)
        for _, row in avg_df_rep.iterrows():
            k = _f(row.get("Metal", ""))
            lines.append(
                f"  {str(row.get('Metal', '')):>6}"
                f" {row.get('N (mg/L)', float('nan')) * k:>13.5f}"
                f" {str(row['AdsCode']):>10} {row.get('AdsConc (g/L)', float('nan')):>8.2f}"
                f" {row['t (min)']:>8.1f} {int(row['Reps']):>5}"
                f" {row['Cₜ (mg/L)'] * k:>14.5f}"
                f" {row.get('Cₜ std (mg/L)', float('nan')) * k:>12.5f}"
                f" {row['q (mg/g)'] * k:>14.6f}"
                f" {row.get('q std', float('nan')) * k:>12.6f}"
            )
    else:
        lines.append("  (no data)")

    lines.append(_subsection("Default kinetic models  —  Pseudo-1st & 2nd order  (per series)"))
    series_list = kin.get("series", [])
    if series_list:
        for s in series_list:
            lines.append(f"\n  Series: {s['label']}")
            lines.append(f"    Metal={s.get('Metal','')}  N={s['N']}  AdsCode={s['AdsCode']}  AdsConc={s['AdsConc']} g/L")
            _metal = s.get("Metal", "")
            m1 = s.get("1st")
            if m1:
                lines.append(f"    1st-order (PFO): k₁={m1['k1']:.8f} min⁻¹   "
                             f"R²={m1['r2']:.6f}   (k₁ is basis-invariant)")
                lines.append(f"       qₑ  = {_tri(m1['qe_mg'], _metal, cap_unit)}")
                lines.append(f"       MAE = {_tri(m1.get('mae_mg'), _metal, cap_unit)}")
            else:
                lines.append("    1st-order: fit did not converge")
            m2 = s.get("2nd")
            if m2:
                lines.append(f"    2nd-order (PSO): R²={m2['r2']:.6f}")
                lines.append(f"       qₑ  = {_tri(m2['qe_mg'], _metal, cap_unit)}")
                lines.append(f"       k₂  = {_tri(m2['k2_mg'], _metal, k2_unit, '{:.8f}', inverse=True)}")
                lines.append(f"       MAE = {_tri(m2.get('mae_mg'), _metal, cap_unit)}")
            else:
                lines.append("    2nd-order: fit did not converge")
    elif kin:
        lines.append("  (kinetics tab not yet processed)")
    else:
        lines.append("  (not available)")

    # ── Extra kinetic models ───────────────────────────────────────────────────
    lines.append(_subsection("Additional kinetic models  (user-selected)"))
    extra_kin = kin.get("extra_models", [])
    if extra_kin:
        # Group by model name
        seen_models: dict[str, list] = {}
        for rec in extra_kin:
            seen_models.setdefault(rec["model"], []).append(rec)
        for model_name, recs in seen_models.items():
            lines.append(f"\n  ── Model: {model_name} ──")
            for rec in recs:
                lines.append(f"    Series : {rec['label']}")
                params = rec.get("params_mg", rec.get("params", {}))
                if params:
                    for p_name, p_val in params.items():
                        try:
                            lines.append(f"      {p_name:<10} = {float(p_val):.6f}")
                        except (TypeError, ValueError):
                            lines.append(f"      {p_name:<10} = {p_val}")
                r2_  = rec.get("r2",  float("nan"))
                mae_ = rec.get("mae_mg", rec.get("mae", float("nan")))
                lines.append(f"      R²         = {r2_:.6f}")
                lines.append(f"      MAE        = {mae_:.6f}")
    else:
        lines.append("  (no additional kinetic models selected)")

    # ── Isotherms ──────────────────────────────────────────────────────────────
    lines.append(_section("ISOTHERMS"))

    iso = st.session_state.get("iso_results", {})

    lines.append(_subsection("Fixed parameters"))
    if iso:
        lines.append(f"  qₑ formula    = (N − Cₑ) / C_ads   [{cap_unit(basis)}]")
        lines.append("  Cₑ source     = XRF CSV file (average of replicates)")
        lines.append("  R_L formula   = 1 / (1 + K_L · N_max)  — N_max per series")
    else:
        lines.append("  (not available)")

    lines.append(_subsection("Raw data summary"))
    raw_df = iso.get("raw_df")
    if raw_df is not None and not raw_df.empty:
        _b, _blocked = resolve_basis(raw_df.get("Metal", []), basis)
        _f = _row_factor(_b)
        _cu, _qu = conc_unit(_b), cap_unit(_b)
        lines.append(_basis_line(_b, _blocked))
        lines.append(
            f"  {'Filename':<35} {'Metal':>6} {'N (' + _cu + ')':>13}"
            f" {'AdsCode':>8} {'Cads':>5} {'R':>3}"
            f" {'Ce (' + _cu + ')':>13} {'qe (' + _qu + ')':>14}"
        )
        lines.append("  " + "-" * 104)
        for _, row in raw_df.iterrows():
            k = _f(row.get("Metal", ""))
            lines.append(
                f"  {str(row['Filename']):<35} {str(row.get('Metal','')):>6}"
                f" {row['N (mg/L)'] * k:>13.5f} {str(row['AdsCode']):>8}"
                f" {row['AdsConc (g/L)']:>5.2f} {int(row['R']):>3}"
                f" {row['Cₑ (mg/L)'] * k:>13.5f} {row['qₑ (mg/g)'] * k:>14.6f}"
            )
    else:
        lines.append("  (no data)")

    lines.append(_subsection("Averaged data (mean Cₑ and qₑ per Metal × N × Ads × Cads)"))
    avg_df_iso = iso.get("avg_df")
    if avg_df_iso is not None and not avg_df_iso.empty:
        _b, _blocked = resolve_basis(avg_df_iso.get("Metal", []), basis)
        _f = _row_factor(_b)
        _cu, _qu = conc_unit(_b), cap_unit(_b)
        lines.append(_basis_line(_b, _blocked))
        lines.append(
            f"  {'Metal':>6} {'N (' + _cu + ')':>13} {'AdsCode':>8}"
            f" {'Cads (g/L)':>10} {'Reps':>5}"
            f" {'Cₑ (' + _cu + ')':>14} {'Ce std':>12}"
            f" {'qₑ (' + _qu + ')':>14} {'qe std':>12}"
        )
        lines.append("  " + "-" * 110)
        for _, row in avg_df_iso.iterrows():
            k = _f(row.get("Metal", ""))
            lines.append(
                f"  {str(row.get('Metal','')):>6} {row['N (mg/L)'] * k:>13.5f}"
                f" {str(row['AdsCode']):>8} {row['AdsConc (g/L)']:>10.2f} {int(row['Reps']):>5}"
                f" {row['Cₑ (mg/L)'] * k:>14.5f} {row.get('Ce_std', float('nan')) * k:>12.5f}"
                f" {row['qₑ (mg/g)'] * k:>14.6f} {row.get('qe_std', float('nan')) * k:>12.6f}"
            )
    else:
        lines.append("  (no data)")

    lines.append(_subsection("Default isotherm models  —  Langmuir & Freundlich  (per series)"))
    if not iso.get("series"):
        lines.append("  (isotherms tab not yet processed)")
    else:
        for s in iso["series"]:
            # Check if there is actually data to report
            if len(s['Ce_mg']) == 0:
                lines.append(f"\n  ── Series: {s['label']} ──")
                lines.append("      No data points available.")
                continue # Skip to the next series

            lines.append(f"\n  ── Series: {s['label']} ──")
            lines.append(f"     Data points : {len(s['Ce_mg'])} (after averaging replicates)")
            _metal = s.get("Metal", "")
            _b, _ = resolve_basis([_metal], basis)
            _k = _row_factor(_b)(_metal)
            lines.append(f"     Ce range    : {s['Ce_mg'].min() * _k:.4g} – "
                         f"{s['Ce_mg'].max() * _k:.4g} {conc_unit(_b)}")
            lines.append(f"     qe range    : {s['qe_mg'].min() * _k:.4g} – "
                         f"{s['qe_mg'].max() * _k:.4g} {cap_unit(_b)}")

            lm = s.get("langmuir")
            lines.append("")
            lines.append("     Langmuir model:  qe = qmax·KL·Ce / (1 + KL·Ce)")
            if lm:
                RL = lm["RL"]
                interp = ("unfavorable" if RL > 1 else
                          "linear"      if abs(RL - 1) < 1e-4 else
                          "favorable"   if RL > 0 else "irreversible")
                lines.append(f"       q_max  = {_tri(lm['qmax_mg'], _metal, cap_unit)}")
                lines.append(f"       K_L    = {_tri(lm['KL_mg'], _metal, kl_unit, '{:.8f}', inverse=True)}")
                lines.append(f"       C₀ (N_max for R_L) = {lm.get('C0_ref_mg', float('nan')):.4g} mg/L")
                lines.append(f"       R_L    = {lm['RL']:.6f}  → {interp}  (basis-invariant)")
                lines.append(f"       R²     = {lm['r2']:.6f}")
                lines.append(f"       MAE    = {_tri(lm.get('mae_mg'), _metal, cap_unit)}")
            else:
                lines.append("       Fit did not converge.")

            fm = s.get("freundlich")
            lines.append("")
            lines.append("     Freundlich model:  qe = KF·Ce^(1/n)")
            if fm:
                fav = ("favorable" if fm["inv_n"] < 1
                       else "linear" if abs(fm["inv_n"] - 1) < 1e-3
                       else "unfavorable")
                _kf = fm.get("KF_mg")
                _kf_tri = "n/a"
                if _kf is not None:
                    _p = []
                    for _b in BASIS_ORDER:
                        _f = basis_factor(_metal, _b)
                        # q = K_F·C^(1/n) ⇒ K_F scales as f^(1−1/n)
                        _p.append(f"{_kf * _f ** (1.0 - 1.0 / fm['n']):.6f} {kf_unit(_b)}"
                                  if _f is not None else f"n/a {kf_unit(_b)}")
                    _kf_tri = "  |  ".join(_p)
                lines.append(f"       K_F    = {_kf_tri}")
                lines.append(f"       n      = {fm['n']:.6f}  (basis-invariant)")
                lines.append(f"       1/n    = {fm['inv_n']:.6f}  → {fav}")
                lines.append(f"       R²     = {fm['r2']:.6f}")
                lines.append(f"       MAE    = {_tri(fm.get('mae_mg'), _metal, cap_unit)}")
            else:
                lines.append("       Fit did not converge.")

    # ── Extra isotherm models ──────────────────────────────────────────────────
    lines.append(_subsection("Additional isotherm models  (user-selected)"))
    extra_iso = iso.get("extra_models", [])
    if extra_iso:
        seen_iso: dict[str, list] = {}
        for rec in extra_iso:
            seen_iso.setdefault(rec["model"], []).append(rec)
        for model_name, recs in seen_iso.items():
            lines.append(f"\n  ── Model: {model_name} ──")
            for rec in recs:
                lines.append(f"    Series : {rec['label']}")
                params = rec.get("params_mg", rec.get("params", {}))
                if params:
                    for p_name, p_val in params.items():
                        try:
                            lines.append(f"      {p_name:<10} = {float(p_val):.6f}")
                        except (TypeError, ValueError):
                            lines.append(f"      {p_name:<10} = {p_val}")
                r2_  = rec.get("r2",  float("nan"))
                mae_ = rec.get("mae_mg", rec.get("mae", float("nan")))
                lines.append(f"      R²         = {r2_:.6f}")
                lines.append(f"      MAE        = {mae_:.6f}")
    else:
        lines.append("  (no additional isotherm models selected)")

    # ── Competitiveness indicators ─────────────────────────────────────────────
    lines.append(_section("COMPETITIVENESS INDICATORS"))

    _kin_ref = st.session_state.get("ind_kin_ref", KIN_REF_FIT)
    _ind = indicator_tables(st.session_state.get("kin_results", {}),
                            st.session_state.get("iso_results", {}),
                            _kin_ref)
    if not _ind:
        lines.append(
            "  (not computed — no multi-element dataset has been processed.)"
        )
    else:
        _note = kinetic_reference_note(st.session_state.get("kin_results", {}),
                                       _kin_ref)
        if _note:
            lines.append(f"  Kinetic reference capacity : {_note}")
        lines.append("  Isotherm reference capacity: q_max from the Langmuir fit")
        lines.append("  Indicators are dimensionless ratios (basis-independent).")
        for _name, _table in _ind.items():
            if _table is None or _table.empty:
                continue
            lines.append(_subsection(_name[len("indicators_"):-4].replace("_", " ")))
            lines.append(_table.round(5).to_string(index=False))

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60 + "\n")

    return "\n".join(lines)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    """Main entry-point; call from app.py inside the Download Results tab context."""
    st.subheader("📥 Download Results")
    st.markdown(
        "This tab assembles a plain-text summary of everything computed so far "
        "— kinetic and isotherm parameters, the competitiveness indicators and "
        "the processed data tables — and lets you download it as a `.txt` file."
    )

    export_basis = unit_selector("download")
    st.caption(
        "The kinetic and isotherm tables below are exported on the selected "
        "basis, which appears in the file name, in the column headers and in a "
        "`Basis` column. Switch the basis and download again to obtain the same "
        "tables on another basis. The competitiveness indicators are "
        "dimensionless ratios and are therefore basis-independent."
    )

    # ── Status summary ─────────────────────────────────────────────────────────
    has_kin  = "kin_results"        in st.session_state
    has_iso  = "iso_results"        in st.session_state
    has_ind  = "indicator_results"  in st.session_state

    col1, col2, col3 = st.columns(3)
    with col1:
        if has_kin:
            n_pts = len(st.session_state["kin_results"].get("df", []))
            extra_kin_n = len(st.session_state["kin_results"].get("extra_models", []))
            label_kin = f"✅ Kinetics ({n_pts} pts"
            label_kin += f", {extra_kin_n} extra)" if extra_kin_n else ")"
            st.success(label_kin)
        else:
            st.warning("⚠️ Kinetics not processed")
    with col2:
        if has_iso:
            n_ptsiso = len(st.session_state["iso_results"].get("raw_df", []))
            extra_iso_n = len(st.session_state["iso_results"].get("extra_models", []))
            label_iso = f"✅ Isotherms ({n_ptsiso} pts"
            label_iso += f", {extra_iso_n} extra)" if extra_iso_n else ")"
            st.success(label_iso)
        else:
            st.warning("⚠️ Isotherms not processed")
    with col3:
        if has_ind:
            ind_d = st.session_state["indicator_results"]
            st.success(
                f"✅ Indicators ({ind_d['study']}, "
                f"{len(ind_d.get('tables', {}))} tables)"
            )
        else:
            st.warning("⚠️ Indicators not computed")

    st.divider()

    # ── Preview ────────────────────────────────────────────────────────────────
    report_text = build_report(export_basis)

    with st.expander("👁  Preview report", expanded=False):
        st.code(report_text, language="")

    # ── Download TXT ───────────────────────────────────────────────────────────
    ts_file  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"HM_adsorption_results_{ts_file}.txt"

    st.download_button(
        label="⬇️  Download results as TXT",
        data=report_text.encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        type="primary",
        width='stretch',
    )
    st.caption(f"File will be saved as: `{filename}`")

    # ── CSV Downloads ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Download chart data as CSV")


    # ── Kinetics & Isotherms — one set of files per selected basis ────────────
    kin = st.session_state.get("kin_results", {})
    iso = st.session_state.get("iso_results", {})

    if not kin and not iso:
        st.caption(
            "_Kinetic and isotherm tables appear here once the **Kinetics** "
            "or **Isotherms** tab has been run._"
        )
    else:
        _KIN_LABELS = {
            "q_vs_time":        "q vs. time — all series (chart data)",
            "model_parameters": "PFO & PSO fitted parameters",
            "fitted_curves":    "PFO & PSO fitted curves (plotted lines)",
            "linear_1st_order": "Linearised 1st order — ln(qₑ − q) vs. t",
            "linear_2nd_order": "Linearised 2nd order — t/q vs. t",
            "extra_models_params": "Additional kinetic models — parameters",
        }
        _ISO_LABELS = {
            "qe_vs_ce":         "qₑ vs. Cₑ — all series (chart data)",
            "model_parameters": "Langmuir & Freundlich fitted parameters",
            "fitted_curves":    "Langmuir & Freundlich fitted curves (plotted lines)",
            "linear_langmuir":  "Linearised Langmuir — Cₑ/qₑ vs. Cₑ",
            "linear_freundlich": "Linearised Freundlich — ln qₑ vs. ln Cₑ",
            "extra_models_params": "Additional isotherm models — parameters",
        }

        def _pretty(fname: str) -> str:
            """Map an export file name to its human-readable button label."""
            stem = fname[:-4]                     # drop '.csv'
            for prefix, labels in (("kinetics_", _KIN_LABELS),
                                   ("isotherms_", _ISO_LABELS)):
                if stem.startswith(prefix):
                    rest = stem[len(prefix):]
                    for tag in ("_mass", "_mol", "_charge"):
                        if rest.endswith(tag):
                            rest = rest[: -len(tag)]
                            break
                    return labels.get(rest, rest.replace("_", " "))
            return stem.replace("_", " ")

        tables = tables_for_basis(kin, iso, export_basis)

        st.markdown(
            f"**{basis_name(export_basis)} basis** — "
            f"capacities in `{cap_unit(export_basis)}`, "
            f"concentrations in `{conc_unit(export_basis)}`"
        )

        kin_files = {k: v for k, v in tables.items() if k.startswith("kinetics_")}
        iso_files = {k: v for k, v in tables.items() if k.startswith("isotherms_")}

        if kin_files:
            st.caption("Kinetics")
            for fname, df in kin_files.items():
                _csv_button(_pretty(fname), df, fname)
        if iso_files:
            st.caption("Isotherms")
            for fname, df in iso_files.items():
                _csv_button(_pretty(fname), df, fname)

        three = three_bases_tables(kin, iso)
        if three:
            st.caption("All three bases side by side (three-panel figures)")
            for fname, df in three.items():
                _csv_button(
                    ("Kinetics" if fname.startswith("kinetics") else "Isotherms")
                    + " — fitted curves on the mass, molar and charge bases",
                    df, fname,
                )

        _inexact = non_exact_models(kin, iso, export_basis)
        if _inexact:
            st.caption(
                f"ℹ️ **{', '.join(_inexact)}** contain a logarithm of the "
                f"concentration, so their parameters have no exact "
                f"reprojection; those rows are always exported on the mass "
                f"basis and their `Basis` column says so."
            )

    # ── Competitiveness indicators ─────────────────────────────────────────────
    kin_ref = st.session_state.get("ind_kin_ref", KIN_REF_FIT)
    ind_files = indicator_tables(kin, iso, kin_ref)

    if ind_files:
        st.markdown("**Competitiveness indicators**")
        note = kinetic_reference_note(kin, kin_ref)
        st.caption(
            "Dimensionless ratios — the same on every basis, so these files "
            "carry no basis suffix. Computed for both studies, independently of "
            "the study selected in the Indicators tab."
            + (f" Kinetic reference capacity: {note}." if note else "")
        )
        _labels_ind = {
            "level1":               "Level 1 — apparent competition (mixture only)",
            "level2":               "Level 2 — competitive effect (mixture vs mono)",
            "level3":               "Level 3 — q across matrices",
            "level3_pairwise":      "Level 3 — pairwise suppression S_i(j)",
            "level3_reciprocity":   "Level 3 — pair dominance (reciprocity)",
            "level3_cooperativity": "Level 3 — binary→ternary cooperativity C_i",
            "level3_competitors":   "Level 3 — strongest competitor per metal",
            "charge_balance":       "Total adsorbed amount per system (CEC signature)",
        }

        def _ind_label(fname: str) -> str:
            stem = fname[:-4]
            if stem == "indicators_cross_axis":
                return "Cross-axis — kinetic vs thermodynamic competition"
            rest = stem[len("indicators_"):]
            for _study in ("kinetics", "isotherms"):
                if rest.startswith(_study + "_"):
                    key = rest[len(_study) + 1:]
                    return f"{_study.capitalize()} — {_labels_ind.get(key, key)}"
            return rest.replace("_", " ")

        for fname, table in ind_files.items():
            _csv_button(_ind_label(fname), table, fname)
    else:
        st.caption(
            "Competitiveness indicators appear here once a multi-element "
            "dataset has been processed in the Kinetics or Isotherms tab."
        )
