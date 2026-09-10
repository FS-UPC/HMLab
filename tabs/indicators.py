"""
tabs/indicators.py — Indicators tab (HMLab).

Collects the competitiveness indicators for the kinetics and isotherm analyses
that have already been run in their own tabs, and presents them in the three
levels defined by the method:

  Level 1 — apparent competition, from the mixture alone
  Level 2 — real competitive effect, mixture vs mono
  Level 3 — pairwise competition, needs binary matrices

Each level is shown as the indicator tables plus the corresponding figures of
the project report (model fits mono vs mixture, Level 2/3 ratio charts, the
cross reading, and the three-bases comparison); see :mod:`indicator_plots`.

The reference quantity differs per study, as specified by the method:
kinetics uses the values measured at the most stable point (the longest contact
time), isotherms use q_max from the Langmuir fit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from indicators import (
    KIN_REF_FIT,
    KIN_REF_LAST,
    apply_kinetic_reference,
    availability,
    best_global_model,
    charge_balance,
    cross_axis,
    henry_selectivity_matrix,
    level1,
    level2,
    level3,
    level3_cooperativity,
    level3_pairwise,
    level3_reciprocity,
    poor_fit_metals,
    selectivity_matrix,
    strongest_competitors,
)
from indicator_plots import (
    fig_cross_axis,
    fig_fractions,
    fig_isotherm_fits,
    fig_kinetic_fits,
    fig_level2_isotherm,
    fig_level2_kinetics,
    fig_level3_pairwise,
    fig_three_bases_isotherm,
    fig_three_bases_kinetics,
    fig_total_charge,
)
from units import missing_charges, missing_molar_masses, unit_selector


def _show(fig, key: str) -> None:
    """Render *fig* when the data allowed it to be built."""
    if fig is not None:
        st.plotly_chart(fig, width="stretch", key=key)

_STUDIES = {
    "Kinetics":  ("kin_results", "q at the most stable point (longest contact time)"),
    "Isotherms": ("iso_results", "q_max from the Langmuir fit"),
}


def _fmt(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(decimals)
    return out


def _level_badge(label: str, info: dict) -> None:
    if info["ok"]:
        st.success(f"**{label}** — available")
    else:
        st.info(f"**{label}** — not available: {info['why']}")


def render() -> None:
    st.subheader("🏁 Competitiveness Indicators")
    st.markdown(
        "Indicators quantifying whether adsorption is more favourable from "
        "mono-, bi- or tri-element solutions, derived from the PFO / PSO "
        "(kinetics) and Langmuir (isotherm) parameters already fitted in the "
        "other tabs."
    )

    basis = unit_selector("indicators")
    st.caption(
        "The **Base** sets the units of the model-fit charts below. The "
        "indicators themselves are ratios or fractions and are therefore "
        "dimensionless — identical on every basis: the adsorbed fractions are "
        "always reported on all three bases side by side (f_mass, f_mol, "
        "f_charge), and the capacities they derive from are listed per basis "
        "(mg/g, mmol/g, meq/g) in the Level 1 tables."
    )

    available = {
        name: st.session_state.get(key)
        for name, (key, _) in _STUDIES.items()
    }
    ready = [name for name, res in available.items()
             if res and res.get("records")]

    if not ready:
        st.warning(
            "⚠️ No analysis has been run yet. Open the **Kinetics** and/or "
            "**Isotherms** tab, select your files, and come back here."
        )
        return

    study = st.radio("Study", ready, horizontal=True, key="ind_study")
    records = available[study]["records"]
    _, ref_desc = _STUDIES[study]

    # For kinetics the reference capacity can be the fitted q_e (what the method
    # document and the project report use) or the value actually measured at the
    # most stable point — the two differ whenever the curve has not levelled off.
    if study == "Kinetics":
        _REF_OPTIONS = {
            KIN_REF_FIT:  "Fitted qₑ (best model)",
            KIN_REF_LAST: "q at the most stable point",
        }
        kin_ref = st.radio(
            "Kinetic reference capacity",
            options=list(_REF_OPTIONS),
            format_func=lambda k: _REF_OPTIONS[k],
            horizontal=True,
            key="ind_kin_ref",
            help="The fitted qₑ extrapolates to equilibrium and is what the "
                 "method document prescribes; the measured value makes no "
                 "extrapolation but underestimates qₑ when the curve is still "
                 "rising at the last time point.",
        )
        records = apply_kinetic_reference(records, kin_ref)
        if kin_ref == KIN_REF_FIT:
            ref_desc = (f"fitted qₑ of the {best_global_model(records)} model "
                        f"(the better of PFO/PSO over this dataset)")
        else:
            ref_desc = "q measured at the most stable point (longest contact time)"

    st.caption(f"Reference quantity for {study.lower()}: **{ref_desc}**.")

    # Drop records whose reference capacity could not be determined
    usable = [r for r in records if r.get("qe_mg") is not None]
    dropped = len(records) - len(usable)
    if dropped:
        st.warning(
            f"⚠️ {dropped} series excluded — the reference capacity could not "
            f"be determined (the model fit did not converge)."
        )
    if not usable:
        st.error("No series with a usable reference capacity.")
        return

    # ── Data availability ──────────────────────────────────────────────────────
    avail = availability(usable)
    st.divider()
    st.markdown("#### Dataset coverage")
    st.markdown("Solution types loaded: **" + ", ".join(avail["solutions"]) + "**")

    c1, c2, c3 = st.columns(3)
    with c1:
        _level_badge("Level 1 — within mixture", avail["level1"])
    with c2:
        _level_badge("Level 2 — mixture vs mono", avail["level2"])
    with c3:
        _level_badge("Level 3 — pairwise", avail["level3"])

    metals = sorted({r["metal"] for r in usable})
    no_mass = missing_molar_masses(metals)
    if no_mass:
        st.warning(
            f"⚠️ Molar mass not available for **{', '.join(no_mass)}** — the "
            f"molar and charge fractions cannot be computed for the matrices "
            f"containing it. Add a row in `data/iupac_molar_masses.csv`."
        )
    no_charge = [m for m in missing_charges(metals) if m not in no_mass]
    if no_charge:
        st.info(
            f"ℹ️ No ionic charge defined for **{', '.join(no_charge)}** — the "
            f"charge fraction is omitted for the matrices containing it."
        )

    # ── Reference mixture used by the figures ──────────────────────────────────
    mixtures = sorted(
        {r["system"] for r in usable if r.get("solution") != "Mono"},
        key=lambda s: (-len(s.split("+")), s),
    )
    ref_mixture = None
    if mixtures:
        st.divider()
        ref_mixture = st.selectbox(
            "Reference mixture for the figures",
            options=mixtures,
            key="ind_ref_mixture",
            help="The matrix compared against the mono-element references in "
                 "the charts below. Defaults to the most complex mixture.",
        )

    # ── Model fits — mono vs mixture ───────────────────────────────────────────
    kin_res = st.session_state.get("kin_results", {})
    iso_res = st.session_state.get("iso_results", {})

    if ref_mixture and (kin_res.get("series") or iso_res.get("series")):
        st.divider()
        st.markdown("### Model fits — mono vs mixture")
        st.markdown(
            "The same metal fitted alone (solid) and inside the mixture "
            "(dashed). The gap between the two curves *is* the competition "
            "that the indicators below quantify."
        )
        if kin_res.get("series"):
            _show(fig_kinetic_fits(kin_res["series"], ref_mixture, basis),
                  "ind_fit_kin")
        if iso_res.get("series"):
            _show(fig_isotherm_fits(iso_res["series"], ref_mixture, basis),
                  "ind_fit_iso")

    # ── Level 1 ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Level 1 — apparent competition (mixture only)")
    st.markdown(
        "Who dominates *inside* the mixture. "
        r"$f_i^{basis} = q_{e,i} / \sum_j q_{e,j}$ on the mass, molar and "
        r"charge-equivalent bases; $K_{d,i} = q_{e,i}/C_{e,i}$; "
        r"$\alpha_{i/j} = K_{d,i}/K_{d,j}$."
    )

    if not avail["level1"]["ok"]:
        st.info(f"Not available: {avail['level1']['why']}.")
    else:
        systems = sorted(
            {r["system"] for r in usable if r.get("solution") != "Mono"},
            key=lambda s: (len(s.split("+")), s),
        )
        for system in systems:
            recs = [r for r in usable if r["system"] == system]
            if len(recs) < 2:
                continue
            st.markdown(f"**Matrix `{system}`**")
            l1 = level1(recs)
            cols = [c for c in ["Metal", "qe (mg/g)", "qe (mmol/g)", "qe (meq/g)",
                                "f_mass", "f_mol", "f_charge", "Ce (mg/L)",
                                "Kd (L/g)", "rank_qe_mass", "rank_qe_mol",
                                "rank_qe_charge", "rank_k1", "rank_k2"]
                    if c in l1.columns]
            st.dataframe(_fmt(l1[cols]), width='stretch', hide_index=True)

            alpha = selectivity_matrix(recs)
            if not alpha.empty:
                st.caption(
                    "Selectivity factor α(i/j) = K_d,i/K_d,j — row *i* over "
                    "column *j*; α > 1 means the adsorbent prefers *i* over *j*."
                )
                st.dataframe(_fmt(alpha), width='stretch')

            if study == "Isotherms":
                alpha_h = henry_selectivity_matrix(recs)
                if not alpha_h.empty:
                    st.caption(
                        "α(i/j) in the Henry form (q_max·K_L ratios) — "
                        "concentration-independent, the recommended form for "
                        "isotherms."
                    )
                    st.dataframe(_fmt(alpha_h), width='stretch')

            _show(fig_fractions(l1, system), f"ind_frac_{system}")

    # ── Level 2 ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Level 2 — real competitive effect (mixture vs mono)")
    st.markdown(
        "How much the co-solutes changed each metal. "
        r"$R_q = q_e^{mix}/q_e^{mono}$, $S_q = 1 - R_q$, and the same ratios "
        r"for the PFO and PSO rate constants. "
        "R > 1 synergy · R ≈ 1 non-interaction · R < 1 antagonism."
    )

    if not avail["level2"]["ok"]:
        st.info(f"Not available: {avail['level2']['why']}.")
    else:
        mono = [r for r in usable if r.get("solution") == "Mono"]
        mix  = [r for r in usable if r.get("solution") != "Mono"]
        l2 = level2(mix, mono)
        cols = [c for c in ["Metal", "System", "Solution", "qe mono (mg/g)",
                            "qe mix (mg/g)", "R_q", "S_q = 1 − R_q", "S_q (%)",
                            "R_k1", "S_k1", "R_k2", "S_k2",
                            "best model", "R_v", "R_h",
                            "t½ mono (min)", "t½ mix (min)", "t½ ratio",
                            "R_K", "R_L mono", "R_L mix", "Effect",
                            "min R² PFO", "min R² PSO", "Note"]
                if c in l2.columns]
        st.dataframe(_fmt(l2[cols]), width='stretch', hide_index=True)

        if study == "Kinetics":
            st.caption(
                "R_v is the rate-constant ratio of the better-fitting model "
                "(R_k1 for PFO, R_k2 for PSO); R_h = h^mix/h^mono with "
                "h = k₂·qe² (PSO initial rate); t½ is the half-adsorption "
                "time — a t½ ratio > 1 means the competition delays the metal."
            )
        else:
            st.caption(
                "R_K = K_L^mix / K_L^mono (affinity ratio): combined with R_q "
                "it separates 'fewer sites' (q_max falls) from 'binds worse' "
                "(K_L falls). R_L pairs show whether adsorption stays "
                "favourable under competition (each at its own series C₀)."
            )

        flagged = poor_fit_metals(l2)
        for model, entries in flagged.items():
            if entries:
                st.warning(
                    f"⚠️ The {model} rate-constant ratio for "
                    f"**{', '.join(entries)}** rests on a fit with R² < 0.9. "
                    f"R_q (capacity) is unaffected, but treat the "
                    f"{model} ratio as unreliable for those series."
                )

        if study == "Isotherms":
            st.caption(
                "PFO / PSO rate-constant ratios are kinetic quantities and are "
                "blank for isotherm data — switch the study selector to "
                "**Kinetics** to see them."
            )

        if ref_mixture:
            fig_l2 = (fig_level2_isotherm(l2, ref_mixture) if study == "Isotherms"
                      else fig_level2_kinetics(l2, ref_mixture))
            _show(fig_l2, f"ind_l2_{study}")
            if fig_l2 is not None:
                st.caption(
                    "Bars below the dashed line at R = 1 mean the mixture "
                    "suppressed that quantity; above it, the mixture enhanced it."
                    + ("  R_h = k₂·qe² combines the capacity and rate effects, "
                       "so it falls faster than either alone."
                       if study == "Kinetics" else
                       "  R_q falling with R_K tells apart 'fewer sites' from "
                       "'weaker binding'.")
                )

    # ── Level 3 ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Level 3 — pairwise competition")
    st.markdown(
        "Comparing the same metal across mono → each binary → ternary reveals "
        "*which* co-solute competes with it — something the full mixture alone "
        "cannot separate."
    )

    if not avail["level3"]["ok"]:
        st.info(f"Not available: {avail['level3']['why']}.")
    else:
        pivot = level3(usable)
        st.markdown("**q (mg/g) across matrices**")
        st.dataframe(_fmt(pivot), width='stretch')

        pairwise = level3_pairwise(usable)
        if not pairwise.empty:
            st.markdown(
                "**Pairwise suppression** — effect of co-ion *j* on metal *i* "
                "in the binary: R_q,i(j) = q_i^(i+j)/q_i^mono, "
                "S_i(j) = 1 − R_q,i(j); rank 1 = strongest competitor of *i*."
            )
            st.dataframe(_fmt(pairwise), width='stretch', hide_index=True)

            _show(fig_level3_pairwise(pairwise), "ind_l3_pairwise")

            recip = level3_reciprocity(pairwise)
            if not recip.empty:
                st.markdown(
                    "**Reciprocity within each pair** — if S_j(i) > S_i(j), "
                    "metal *i* dominates *j* (consistent with α(i/j) > 1)."
                )
                st.dataframe(_fmt(recip), width='stretch', hide_index=True)

        coop = level3_cooperativity(usable)
        if not coop.empty:
            st.markdown(
                "**Binary → ternary cooperativity** — predicted ternary "
                "capacity assuming independent (multiplicative) binary "
                "suppressions, vs measured: C_i = q_meas/q_pred. "
                "C ≈ 1 independent · C < 1 competitors reinforce · "
                "C > 1 competitors overlap."
            )
            st.dataframe(_fmt(coop), width='stretch', hide_index=True)

        worst = strongest_competitors(usable)
        if not worst.empty:
            st.markdown("**Strongest competitor per metal** (lowest R_q, any matrix)")
            st.dataframe(_fmt(worst), width='stretch', hide_index=True)

    # ── Cross-axis: kinetics vs isotherm ───────────────────────────────────────
    st.divider()
    st.markdown("### Cross-axis reading — kinetic vs thermodynamic competition")
    st.markdown(
        "Crossing the **velocity** effect (R_v, kinetics) with the "
        "**capacity/affinity** effect (R_q from q_max and R_K, isotherm) "
        "tells whether competition is a transport bottleneck, a fight for "
        "sites, both, or absent."
    )

    kin_res = st.session_state.get("kin_results", {})
    iso_res = st.session_state.get("iso_results", {})
    kin_recs = [r for r in kin_res.get("records", []) if r.get("qe_mg") is not None]
    iso_recs = [r for r in iso_res.get("records", []) if r.get("qe_mg") is not None]

    cross = pd.DataFrame()
    if kin_recs and iso_recs:
        kin_l2 = level2([r for r in kin_recs if r.get("solution") != "Mono"],
                        [r for r in kin_recs if r.get("solution") == "Mono"])
        iso_l2 = level2([r for r in iso_recs if r.get("solution") != "Mono"],
                        [r for r in iso_recs if r.get("solution") == "Mono"])
        cross = cross_axis(kin_l2, iso_l2)

    if cross.empty:
        st.info(
            "Needs both studies: run **Kinetics** and **Isotherms** on "
            "matching matrices (same metals and systems) to enable the "
            "cross-axis table."
        )
    else:
        st.dataframe(_fmt(cross), width='stretch', hide_index=True)
        _show(fig_cross_axis(cross, ref_mixture), "ind_cross")
        st.caption(
            "R_v < 1 with R_q^iso ≈ 1 → kinetic-only competition; "
            "R_v ≈ 1 with R_q^iso < 1 → thermodynamic competition; "
            "both < 1 → strong antagonism; both > 1 → synergy. "
            "A conclusion is more defensible when it appears on both axes."
        )

    # ── The three bases ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### The three bases — mass, molar and charge")
    st.markdown(
        "The same adsorbed amounts read as mass (mg/g), as number of ions "
        "(mmol/g) and as ionic charge (meq/g). With mixed valences the three "
        "readings disagree, and the metal that *dominates* the adsorbed phase "
        "can change from one basis to the next."
    )

    if ref_mixture:
        if iso_res.get("series"):
            _show(fig_three_bases_isotherm(iso_res["series"], ref_mixture),
                  "ind_bases_iso")
        if kin_res.get("series"):
            _show(fig_three_bases_kinetics(kin_res["series"], ref_mixture),
                  "ind_bases_kin")

    fig_q = fig_total_charge(usable, study)
    _show(fig_q, "ind_total_charge")
    if fig_q is not None:
        st.caption(
            "Σ q in meq/g per system. Mixtures converging on a common value "
            "while the mono systems sit below it is the signature of a shared "
            "pool of exchangeable charge — i.e. the adsorbent's cation "
            "exchange capacity being divided among the ions, which supports an "
            "ion-exchange mechanism."
        )
    elif study:
        st.caption(
            "_The total-charge chart needs the ionic charge of every metal in "
            "the lookup table._"
        )

    # ── Persist for the Download tab ───────────────────────────────────────────
    tables: dict[str, pd.DataFrame] = {}
    if avail["level1"]["ok"]:
        frames = []
        for system in sorted({r["system"] for r in usable
                              if r.get("solution") != "Mono"}):
            recs = [r for r in usable if r["system"] == system]
            if len(recs) < 2:
                continue
            frame = level1(recs)
            frame.insert(0, "System", system)
            frames.append(frame)
        if frames:
            tables["level1"] = pd.concat(frames, ignore_index=True)
    if avail["level2"]["ok"]:
        tables["level2"] = level2(
            [r for r in usable if r.get("solution") != "Mono"],
            [r for r in usable if r.get("solution") == "Mono"],
        )
    if avail["level3"]["ok"]:
        tables["level3"] = level3(usable).reset_index()
        pw = level3_pairwise(usable)
        if not pw.empty:
            tables["level3_pairwise"] = pw
            rec_t = level3_reciprocity(pw)
            if not rec_t.empty:
                tables["level3_reciprocity"] = rec_t
        coop_t = level3_cooperativity(usable)
        if not coop_t.empty:
            tables["level3_cooperativity"] = coop_t
        tables["level3_competitors"] = strongest_competitors(usable)
    if not cross.empty:
        tables["cross_axis"] = cross
    _cb = charge_balance(usable)
    if not _cb.empty:
        tables["charge_balance"] = _cb

    st.session_state["indicator_results"] = {
        "study":     study,
        "reference": ref_desc,
        "solutions": avail["solutions"],
        "tables":    tables,
    }
