"""
exports.py — build the downloadable tables for any reporting basis.

The analysis tabs store their results on the **mass basis** (mg/L, mg/g), which
is the basis every model is fitted on.  This module reprojects those canonical
results onto whichever basis the user asks for in the Download tab, so the same
analysis can be exported as Mass, Mol and/or Charge without re-running it.

Every table carries a ``Basis`` column naming the basis its numbers are
actually in.  That column is not decoration: when an element has no molar mass
(or no ionic charge on the charge basis) its rows fall back to the mass basis,
and the column is what tells the reader which rows those are.

The plotted quantities are all covered here:

    q(t) and qₑ(Cₑ) points ....... *_data_<basis>.csv
    fitted model curves .......... *_fitted_curves_<basis>.csv
    fitted model parameters ...... *_model_parameters_<basis>.csv
    linearised diagnostics ....... *_linear_*_<basis>.csv
    optional-model curves/params . *_extra_models_*_<basis>.csv

The grouped Mono/Bi/Tri comparison figures re-plot the same series, so their
data is the ``*_data_<basis>.csv`` table filtered by its ``Metal``, ``System``
and ``Solution`` columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    KIN_REF_FIT,
    apply_kinetic_reference,
    availability,
    best_global_model,
    charge_balance,
    cross_axis,
    level1,
    level2,
    level3,
    level3_cooperativity,
    level3_pairwise,
    level3_reciprocity,
    strongest_competitors,
)
from units import (
    CHARGE_BASIS,
    MASS_BASIS,
    MOLAR_BASIS,
    NON_COVARIANT_MODELS,
    basis_factor,
    basis_name,
    basis_tag,
    cap_unit,
    conc_unit,
    k2_unit,
    kl_unit,
    scale_isotherm_params,
    scale_kinetic_params,
)
from utils import model_first_order, model_second_order

_N_CURVE = 200          # points per fitted curve


def _factor(metal: str, basis: str) -> tuple[float, str]:
    """
    Conversion factor for *metal* and the basis actually applied.

    Falls back to the mass basis when the molar mass (or the ionic charge, on
    the charge basis) is unknown, so a missing table entry can never silently
    produce a wrong number.
    """
    f = basis_factor(metal, basis)
    if f is None:
        return 1.0, MASS_BASIS
    return f, basis


# ══════════════════════════════════════════════════════════════════════════════
# Kinetics
# ══════════════════════════════════════════════════════════════════════════════

def kinetics_tables(kin: dict, basis: str) -> dict[str, pd.DataFrame]:
    """Return ``{filename: DataFrame}`` for the kinetics tab on *basis*."""
    if not kin:
        return {}

    tag = basis_tag(basis)
    out: dict[str, pd.DataFrame] = {}
    series = kin.get("series", [])

    q_u, c_u = cap_unit(basis), conc_unit(basis)

    # ── q(t) points, i.e. the main chart and the grouped comparison charts ──
    rows = []
    for s in series:
        f, used = _factor(s["Metal"], basis)
        for i, t in enumerate(np.asarray(s["t"], dtype=float)):
            rows.append({
                "Basis":            basis_name(used),
                "Series":           s["label"],
                "System":           s["System"],
                "Solution":         s["Solution"],
                "Metal":            s["Metal"],
                "pH":               s.get("pH"),
                "N (mg/L)":         s["N"],
                "AdsCode":          s["AdsCode"],
                "AdsConc (g/L)":    s["AdsConc"],
                "t (min)":          t,
                f"Cₜ ({c_u})":      s["Ct_mg"][i] * f,
                f"q ({q_u})":       s["q_mg"][i] * f,
                f"q std ({q_u})":   (s["q_std_mg"][i] * f
                                     if s.get("q_std_mg") is not None else None),
            })
    if rows:
        out[f"kinetics_q_vs_time_{tag}.csv"] = pd.DataFrame(rows)

    # ── PFO / PSO fitted parameters ────────────────────────────────────────
    rows = []
    for s in series:
        f, used = _factor(s["Metal"], basis)
        base = {
            "Basis":    basis_name(used),
            "Series":   s["label"],
            "System":   s["System"],
            "Solution": s["Solution"],
            "Metal":    s["Metal"],
            "AdsCode":  s["AdsCode"],
            "AdsConc (g/L)": s["AdsConc"],
        }
        if "1st" in s:
            fit = s["1st"]
            rows.append({**base, "Model": "PFO (pseudo-1st-order)",
                         f"qe ({q_u})": fit["qe_mg"] * f,
                         "k1 (min⁻¹)": fit["k1"],
                         "k2": None, "R2": fit["r2"],
                         f"MAE ({q_u})": (fit.get("mae_mg") or 0.0) * f})
        if "2nd" in s:
            fit = s["2nd"]
            rows.append({**base, "Model": "PSO (pseudo-2nd-order)",
                         f"qe ({q_u})": fit["qe_mg"] * f,
                         "k1 (min⁻¹)": None,
                         f"k2 ({k2_unit(used)})": fit["k2_mg"] / f,
                         "R2": fit["r2"],
                         f"MAE ({q_u})": (fit.get("mae_mg") or 0.0) * f})
    if rows:
        out[f"kinetics_model_parameters_{tag}.csv"] = pd.DataFrame(rows)

    # ── Fitted PFO / PSO curves (the lines drawn on the chart) ─────────────
    rows = []
    for s in series:
        t_arr = np.asarray(s["t"], dtype=float)
        if not len(t_arr):
            continue
        f, used = _factor(s["Metal"], basis)
        t_fit = np.linspace(0.0, t_arr.max() * 1.05, _N_CURVE)
        pfo = (model_first_order(t_fit, s["1st"]["qe_mg"], s["1st"]["k1"]) * f
               if "1st" in s else np.full(_N_CURVE, np.nan))
        pso = (model_second_order(t_fit, s["2nd"]["qe_mg"], s["2nd"]["k2_mg"]) * f
               if "2nd" in s else np.full(_N_CURVE, np.nan))
        for t, a, b in zip(t_fit, pfo, pso):
            rows.append({
                "Basis": basis_name(used), "Series": s["label"],
                "Metal": s["Metal"], "System": s["System"],
                "Solution": s["Solution"], "t (min)": t,
                f"q PFO ({q_u})": a, f"q PSO ({q_u})": b,
            })
    if rows:
        out[f"kinetics_fitted_curves_{tag}.csv"] = pd.DataFrame(rows)

    # ── Linearised diagnostics ─────────────────────────────────────────────
    lin1, lin2 = [], []
    for s in series:
        f, used = _factor(s["Metal"], basis)
        q = np.asarray(s["q_mg"], dtype=float) * f
        t = np.asarray(s["t"], dtype=float)
        qe1 = s.get("1st", {}).get("qe_mg")
        for ti, qi in zip(t, q):
            lin2.append({"Basis": basis_name(used), "Series": s["label"],
                         "Metal": s["Metal"], "t (min)": ti,
                         f"t/q (min·g·{q_u.split('/')[0]}⁻¹)":
                             ti / qi if qi > 1e-12 else None})
            if qe1 is not None:
                lin1.append({"Basis": basis_name(used), "Series": s["label"],
                             "Metal": s["Metal"], "t (min)": ti,
                             "ln(qe − q)": float(np.log(max(qe1 * f - qi, 1e-12)))})
    if lin1:
        out[f"kinetics_linear_1st_order_{tag}.csv"] = pd.DataFrame(lin1)
    if lin2:
        out[f"kinetics_linear_2nd_order_{tag}.csv"] = pd.DataFrame(lin2)

    # ── Optional models ────────────────────────────────────────────────────
    extra = kin.get("extra_models", [])
    rows = []
    for rec in extra:
        f, used = _factor(rec.get("Metal", ""), basis)
        params, exact = scale_kinetic_params(
            rec.get("model", ""), rec.get("params_mg", {}), f)
        row = {
            "Basis":    basis_name(used) if exact else basis_name(MASS_BASIS),
            "Model":    rec.get("model"),
            "Series":   rec.get("label"),
            "System":   rec.get("System"),
            "Solution": rec.get("Solution"),
            "Metal":    rec.get("Metal"),
        }
        row.update(params)
        row["R2"] = rec.get("r2")
        row[f"MAE ({q_u})"] = (rec.get("mae_mg") or float("nan")) * f
        rows.append(row)
    if rows:
        out[f"kinetics_extra_models_params_{tag}.csv"] = pd.DataFrame(rows)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Isotherms
# ══════════════════════════════════════════════════════════════════════════════

def _langmuir(Ce, qmax, KL):
    return (qmax * KL * Ce) / (1.0 + KL * Ce)


def _freundlich(Ce, KF, n):
    return KF * np.power(np.abs(Ce), 1.0 / n)


def isotherm_tables(iso: dict, basis: str) -> dict[str, pd.DataFrame]:
    """Return ``{filename: DataFrame}`` for the isotherms tab on *basis*."""
    if not iso:
        return {}

    tag = basis_tag(basis)
    out: dict[str, pd.DataFrame] = {}
    series = iso.get("series", [])

    q_u, c_u = cap_unit(basis), conc_unit(basis)

    # ── (Cₑ, qₑ) points — main chart and grouped comparison charts ─────────
    rows = []
    for s in series:
        f, used = _factor(s["Metal"], basis)
        for ce, qe in zip(np.asarray(s["Ce_mg"], dtype=float),
                          np.asarray(s["qe_mg"], dtype=float)):
            rows.append({
                "Basis":        basis_name(used),
                "Series":       s["label"],
                "System":       s["System"],
                "Solution":     s["Solution"],
                "Metal":        s["Metal"],
                "pH":           s.get("pH"),
                "AdsCode":      s["Ads"],
                "AdsConc (g/L)": s["Cads"],
                f"Cₑ ({c_u})":  ce * f,
                f"qₑ ({q_u})":  qe * f,
            })
    if rows:
        out[f"isotherms_qe_vs_ce_{tag}.csv"] = pd.DataFrame(rows)

    # ── Langmuir / Freundlich parameters ───────────────────────────────────
    rows = []
    for s in series:
        f, used = _factor(s["Metal"], basis)
        base = {
            "Basis": basis_name(used), "Series": s["label"],
            "System": s["System"], "Solution": s["Solution"],
            "Metal": s["Metal"], "AdsCode": s["Ads"],
            "AdsConc (g/L)": s["Cads"],
        }
        if "langmuir" in s:
            L = s["langmuir"]
            rows.append({
                **base, "Model": "Langmuir",
                f"q_max ({q_u})": L["qmax_mg"] * f,
                f"K_L ({kl_unit(used)})": L["KL_mg"] / f,
                "R_L": L["RL"], f"C0 ref ({c_u})": L.get("C0_ref_mg", np.nan) * f,
                "K_F": None, "n": None, "1/n": None,
                "R2": L["r2"], f"MAE ({q_u})": (L.get("mae_mg") or 0.0) * f,
            })
        if "freundlich" in s:
            F = s["freundlich"]
            n = F["n"]
            # q = K_F·C^(1/n) ⇒ K_F scales as f^(1−1/n)
            KF = F.get("KF_mg", np.nan) * f ** (1.0 - 1.0 / n)
            rows.append({
                **base, "Model": "Freundlich",
                f"q_max ({q_u})": None, f"K_L ({kl_unit(used)})": None,
                "R_L": None, f"C0 ref ({c_u})": None,
                "K_F": KF, "n": n, "1/n": 1.0 / n,
                "R2": F["r2"], f"MAE ({q_u})": (F.get("mae_mg") or 0.0) * f,
            })
    if rows:
        out[f"isotherms_model_parameters_{tag}.csv"] = pd.DataFrame(rows)

    # ── Fitted Langmuir / Freundlich curves ────────────────────────────────
    rows = []
    for s in series:
        ce = np.asarray(s["Ce_mg"], dtype=float)
        if not len(ce):
            continue
        f, used = _factor(s["Metal"], basis)
        ce_fit = np.linspace(max(ce.min() * 0.5, 1e-3), ce.max() * 1.1, _N_CURVE)
        lang = (_langmuir(ce_fit, s["langmuir"]["qmax_mg"], s["langmuir"]["KL_mg"]) * f
                if "langmuir" in s else np.full(_N_CURVE, np.nan))
        freu = (_freundlich(ce_fit, s["freundlich"].get("KF_mg", np.nan),
                            s["freundlich"]["n"]) * f
                if "freundlich" in s else np.full(_N_CURVE, np.nan))
        for c, a, b in zip(ce_fit * f, lang, freu):
            rows.append({
                "Basis": basis_name(used), "Series": s["label"],
                "Metal": s["Metal"], "System": s["System"],
                "Solution": s["Solution"], f"Cₑ ({c_u})": c,
                f"qₑ Langmuir ({q_u})": a, f"qₑ Freundlich ({q_u})": b,
            })
    if rows:
        out[f"isotherms_fitted_curves_{tag}.csv"] = pd.DataFrame(rows)

    # ── Linearised diagnostics ─────────────────────────────────────────────
    lin_l, lin_f = [], []
    for s in series:
        f, used = _factor(s["Metal"], basis)
        for ce, qe in zip(np.asarray(s["Ce_mg"], dtype=float) * f,
                          np.asarray(s["qe_mg"], dtype=float) * f):
            lin_l.append({"Basis": basis_name(used), "Series": s["label"],
                          "Metal": s["Metal"], f"Cₑ ({c_u})": ce,
                          "Cₑ/qₑ (g/L)": ce / qe if qe > 1e-12 else None})
            if ce > 0 and qe > 0:
                lin_f.append({"Basis": basis_name(used), "Series": s["label"],
                              "Metal": s["Metal"],
                              "ln(Cₑ)": float(np.log(ce)),
                              "ln(qₑ)": float(np.log(qe))})
    if lin_l:
        out[f"isotherms_linear_langmuir_{tag}.csv"] = pd.DataFrame(lin_l)
    if lin_f:
        out[f"isotherms_linear_freundlich_{tag}.csv"] = pd.DataFrame(lin_f)

    # ── Optional models ────────────────────────────────────────────────────
    extra = iso.get("extra_models", [])
    rows = []
    for rec in extra:
        f, used = _factor(rec.get("Metal", ""), basis)
        params, exact = scale_isotherm_params(
            rec.get("model", ""), rec.get("params_mg", {}), f)
        row = {
            "Basis":    basis_name(used) if exact else basis_name(MASS_BASIS),
            "Model":    rec.get("model"),
            "Series":   rec.get("label"),
            "System":   rec.get("System"),
            "Solution": rec.get("Solution"),
            "Metal":    rec.get("Metal"),
        }
        row.update(params)
        row["R2"] = rec.get("r2")
        row[f"MAE ({q_u})"] = (rec.get("mae_mg") or float("nan")) * f
        rows.append(row)
    if rows:
        out[f"isotherms_extra_models_params_{tag}.csv"] = pd.DataFrame(rows)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# All tables for one basis
# ══════════════════════════════════════════════════════════════════════════════

def tables_for_basis(kin: dict, iso: dict, basis: str) -> dict[str, pd.DataFrame]:
    """Every kinetics and isotherm export table for a single *basis*."""
    tables = kinetics_tables(kin, basis)
    tables.update(isotherm_tables(iso, basis))
    return tables


def non_exact_models(kin: dict, iso: dict, basis: str) -> list[str]:
    """
    Optional models whose parameters cannot be reprojected onto *basis*
    (they contain a logarithm of the concentration) and are therefore always
    reported on the mass basis.
    """
    if basis == MASS_BASIS:
        return []
    used = [rec.get("model") for rec in kin.get("extra_models", [])] + \
           [rec.get("model") for rec in iso.get("extra_models", [])]
    names = []
    for m in used:
        if m in NON_COVARIANT_MODELS and m not in names:
            names.append(m)
    return names


# ══════════════════════════════════════════════════════════════════════════════
# Indicator tables — computed for BOTH studies
# ══════════════════════════════════════════════════════════════════════════════

def _study_records(res: dict, study: str, kin_ref: str) -> list[dict]:
    recs = [r for r in res.get("records", []) if r.get("qe_mg") is not None]
    if study == "kinetics":
        recs = apply_kinetic_reference(recs, kin_ref)
    return recs


def indicator_tables(kin: dict, iso: dict,
                     kin_ref: str = KIN_REF_FIT) -> dict[str, pd.DataFrame]:
    """
    Every competitiveness table, for the kinetic *and* the isotherm study.

    These are the numbers behind the Indicators-tab figures, so the export does
    not depend on which study happens to be selected there — nor on that tab
    having been opened at all.  The indicators are dimensionless ratios, hence
    no basis suffix; the capacities they derive from are given on all three
    bases inside the Level 1 and charge-balance tables.
    """
    out: dict[str, pd.DataFrame] = {}
    per_study: dict[str, list[dict]] = {}

    for study, res in (("kinetics", kin), ("isotherms", iso)):
        recs = _study_records(res, study, kin_ref)
        if not recs:
            continue
        per_study[study] = recs
        av = availability(recs)
        mono = [r for r in recs if r.get("solution") == "Mono"]
        mix = [r for r in recs if r.get("solution") != "Mono"]

        if av["level1"]["ok"]:
            frames = []
            for system in sorted({r["system"] for r in mix}):
                sub = [r for r in recs if r["system"] == system]
                if len(sub) < 2:
                    continue
                frame = level1(sub)
                frame.insert(0, "System", system)
                frames.append(frame)
            if frames:
                out[f"indicators_{study}_level1.csv"] = pd.concat(
                    frames, ignore_index=True)

        if av["level2"]["ok"]:
            out[f"indicators_{study}_level2.csv"] = level2(mix, mono)

        if av["level3"]["ok"]:
            out[f"indicators_{study}_level3.csv"] = level3(recs).reset_index()
            pw = level3_pairwise(recs)
            if not pw.empty:
                out[f"indicators_{study}_level3_pairwise.csv"] = pw
                rec_t = level3_reciprocity(pw)
                if not rec_t.empty:
                    out[f"indicators_{study}_level3_reciprocity.csv"] = rec_t
            coop = level3_cooperativity(recs)
            if not coop.empty:
                out[f"indicators_{study}_level3_cooperativity.csv"] = coop
            comp = strongest_competitors(recs)
            if not comp.empty:
                out[f"indicators_{study}_level3_competitors.csv"] = comp

        cb = charge_balance(recs)
        if not cb.empty:
            out[f"indicators_{study}_charge_balance.csv"] = cb

    # Cross-axis needs both studies on matching matrices
    if "kinetics" in per_study and "isotherms" in per_study:
        k, i = per_study["kinetics"], per_study["isotherms"]
        cross = cross_axis(
            level2([r for r in k if r.get("solution") != "Mono"],
                   [r for r in k if r.get("solution") == "Mono"]),
            level2([r for r in i if r.get("solution") != "Mono"],
                   [r for r in i if r.get("solution") == "Mono"]),
        )
        if not cross.empty:
            out["indicators_cross_axis.csv"] = cross

    return out


def kinetic_reference_note(kin: dict, kin_ref: str = KIN_REF_FIT) -> str:
    """One line naming the kinetic reference capacity behind the indicators."""
    recs = _study_records(kin, "kinetics", kin_ref)
    if not recs:
        return ""
    if kin_ref == KIN_REF_FIT:
        return (f"fitted q_e of the {best_global_model(recs)} model "
                f"(better of PFO/PSO over this dataset)")
    return "q measured at the most stable point (longest contact time)"


# ══════════════════════════════════════════════════════════════════════════════
# The three bases side by side (the data behind the three-panel figures)
# ══════════════════════════════════════════════════════════════════════════════

def three_bases_tables(kin: dict, iso: dict) -> dict[str, pd.DataFrame]:
    """
    Fitted curves of every series with q given on all three bases at once.

    This is the table form of the three-panel figures: one row per point, with
    the mass, molar and charge capacities side by side.  For the isotherm the
    liquid-phase concentration stays in mg/L across the three columns, so the
    table isolates the effect of the basis on the adsorbed amount.
    """
    out: dict[str, pd.DataFrame] = {}

    rows = []
    for s in kin.get("series", []):
        t_arr = np.asarray(s["t"], dtype=float)
        fit = s.get("2nd") or s.get("1st")
        if not len(t_arr) or fit is None:
            continue
        t_fit = np.linspace(0.0, t_arr.max() * 1.02, _N_CURVE)
        q_mass = (model_second_order(t_fit, fit["qe_mg"], fit["k2_mg"])
                  if "k2_mg" in fit
                  else model_first_order(t_fit, fit["qe_mg"], fit["k1"]))
        fm = basis_factor(s["Metal"], MOLAR_BASIS)
        fc = basis_factor(s["Metal"], CHARGE_BASIS)
        for t, q in zip(t_fit, q_mass):
            rows.append({
                "Series": s["label"], "Metal": s["Metal"],
                "System": s["System"], "Solution": s["Solution"],
                "t (min)": t,
                "q (mg/g)": q,
                "q (mmol/g)": q * fm if fm is not None else None,
                "q (meq/g)": q * fc if fc is not None else None,
            })
    if rows:
        out["kinetics_three_bases.csv"] = pd.DataFrame(rows)

    rows = []
    for s in iso.get("series", []):
        ce = np.asarray(s["Ce_mg"], dtype=float)
        lang = s.get("langmuir")
        if not len(ce) or not lang:
            continue
        ce_fit = np.linspace(max(ce.min() * 0.2, 1e-3), ce.max() * 1.05, _N_CURVE)
        q_mass = _langmuir(ce_fit, lang["qmax_mg"], lang["KL_mg"])
        fm = basis_factor(s["Metal"], MOLAR_BASIS)
        fc = basis_factor(s["Metal"], CHARGE_BASIS)
        for c, q in zip(ce_fit, q_mass):
            rows.append({
                "Series": s["label"], "Metal": s["Metal"],
                "System": s["System"], "Solution": s["Solution"],
                "Cₑ (mg/L)": c,
                "qₑ (mg/g)": q,
                "qₑ (mmol/g)": q * fm if fm is not None else None,
                "qₑ (meq/g)": q * fc if fc is not None else None,
            })
    if rows:
        out["isotherms_three_bases.csv"] = pd.DataFrame(rows)

    return out
