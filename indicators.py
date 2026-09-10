"""
indicators.py — competitiveness indicators for multi-element adsorption.

Implements the "implicit competition" framework used by the project: every
metal is fitted with the same single-component models inside each matrix
(mono, binary, ternary …), and competition is read afterwards by comparing the
fitted parameters between matrices.

Three levels, in increasing order of robustness:

  Level 1 — apparent competition, from the mixture alone
            f_i(mass/mol/charge), K_d, α_i/j, capacity and rate rankings
  Level 2 — real competitive effect, mixture vs mono
            R_q, S_q = 1 − R_q, R_k1, S_k1, R_k2, S_k2
  Level 3 — pairwise competition, needs binary matrices
            q of one metal across mono → each binary → ternary

The module is deliberately free of Streamlit imports so it can be unit-tested
and reused outside the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from units import ionic_charge, molar_mass

# ── Solution types ─────────────────────────────────────────────────────────────

SOLUTION_NAMES = {1: "Mono", 2: "Bi", 3: "Tri"}
SOLUTION_ORDER = ["Mono", "Bi", "Tri"]


def solution_type(n_metals: int) -> str:
    """1 → 'Mono', 2 → 'Bi', 3 → 'Tri', n → 'n-element'."""
    try:
        n = int(n_metals)
    except (TypeError, ValueError):
        return "?"
    return SOLUTION_NAMES.get(n, f"{n}-element")


def solution_sort_key(name: str) -> tuple[int, str]:
    """Sort key placing Mono < Bi < Tri < anything else (alphabetical)."""
    if name in SOLUTION_ORDER:
        return (SOLUTION_ORDER.index(name), name)
    return (len(SOLUTION_ORDER), name)


# ── Basis conversion of a record ───────────────────────────────────────────────

def _q_bases(qe_mg: float, metal: str) -> dict[str, float | None]:
    """Return {mass, mol, charge} versions of a capacity given in mg/g."""
    M = molar_mass(metal)
    z = ionic_charge(metal)
    q_mol = None if (M is None or M <= 0 or qe_mg is None) else qe_mg / M
    q_chg = None if (q_mol is None or z is None) else q_mol * abs(z)
    return {"mass": qe_mg, "mol": q_mol, "charge": q_chg}


def _safe_ratio(num, den):
    """num/den, or None when either side is missing or the denominator is ~0."""
    if num is None or den is None:
        return None
    try:
        num, den = float(num), float(den)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-15:
        return None
    return num / den


# ── Level 1 — apparent competition, within one mixture ─────────────────────────

def level1(records: list[dict]) -> pd.DataFrame:
    """
    Indicators computed from one multi-element matrix alone.

    *records* is a list of per-metal dicts for a **single** matrix, each with
    at least ``metal``, ``qe_mg`` (mg/g) and ``Ce_mg`` (mg/L); ``k1`` and
    ``k2_mg`` are optional.

    Returns one row per metal with the adsorbed fractions on the three bases,
    the distribution coefficient and the capacity / rate rankings.
    """
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        metal = rec["metal"]
        q = _q_bases(rec.get("qe_mg"), metal)
        rows.append({
            "Metal":            metal,
            "qe (mg/g)":        q["mass"],
            "qe (mmol/g)":      q["mol"],
            "qe (meq/g)":       q["charge"],
            "Ce (mg/L)":        rec.get("Ce_mg"),
            "k1 (min⁻¹)":       rec.get("k1"),
            "k2 (g·mg⁻¹·min⁻¹)": rec.get("k2_mg"),
        })
    df = pd.DataFrame(rows)

    # Adsorbed fractions — each basis needs every metal to be convertible,
    # otherwise the sum in the denominator would be incomplete.
    for basis, col, out in (("mass",   "qe (mg/g)",   "f_mass"),
                            ("mol",    "qe (mmol/g)", "f_mol"),
                            ("charge", "qe (meq/g)",  "f_charge")):
        vals = df[col]
        total = vals.sum(skipna=False)
        if pd.notna(total) and abs(total) > 1e-15:
            df[out] = vals / total
        else:
            df[out] = np.nan

    # Distribution coefficient K_d = qe/Ce.  Basis-invariant for a given metal
    # (the molar mass cancels), so the mass basis is used.
    df["Kd (L/g)"] = [
        _safe_ratio(q, c) for q, c in zip(df["qe (mg/g)"], df["Ce (mg/L)"])
    ]

    # Rankings — 1 = highest
    for col, out in (("qe (mg/g)",           "rank_qe_mass"),
                     ("qe (mmol/g)",         "rank_qe_mol"),
                     ("qe (meq/g)",          "rank_qe_charge"),
                     ("k1 (min⁻¹)",          "rank_k1"),
                     ("k2 (g·mg⁻¹·min⁻¹)",   "rank_k2")):
        df[out] = df[col].rank(ascending=False, method="min").astype("Int64")

    return df


def selectivity_matrix(records: list[dict]) -> pd.DataFrame:
    """
    Selectivity factors α_i/j = K_d,i / K_d,j for one matrix.

    Rows are metal *i*, columns metal *j*; α > 1 means the adsorbent is more
    selective for *i* than for *j*.  α is basis-invariant.
    """
    kd = {}
    for rec in records:
        kd[rec["metal"]] = _safe_ratio(rec.get("qe_mg"), rec.get("Ce_mg"))

    metals = list(kd)
    if len(metals) < 2:
        return pd.DataFrame()

    data = {
        j: [_safe_ratio(kd[i], kd[j]) for i in metals]
        for j in metals
    }
    return pd.DataFrame(data, index=metals)


# ── Derived kinetic quantities (guide §3.3 / §3.4) ─────────────────────────────

def initial_rate_h(rec: dict):
    """PSO initial rate h = k₂·qe² (mass basis), from the PSO fit."""
    k2 = rec.get("k2_mg")
    qe = rec.get("qe_pso_mg", rec.get("qe_mg"))
    if k2 is None or qe is None:
        return None
    return k2 * qe * qe


def half_time(rec: dict, model: str):
    """
    Half-adsorption time t½ in minutes: ln2/k₁ (PFO) or 1/(k₂·qe) (PSO).
    Basis-invariant (k₂·qe is), computed from the mass-basis parameters.
    """
    if model == "PFO":
        k1 = rec.get("k1")
        return None if not k1 else float(np.log(2.0)) / k1
    k2 = rec.get("k2_mg")
    qe = rec.get("qe_pso_mg", rec.get("qe_mg"))
    if not k2 or not qe:
        return None
    return 1.0 / (k2 * qe)


def best_kinetic_model(mix: dict, mono: dict) -> str | None:
    """
    PFO or PSO — whichever fits the *pair* better, judged by the worse of the
    two R²s (the guide's rule: always compare the same model between
    matrices, chosen as the best one).
    """
    pfo = _min_r2(mix.get("r2_pfo"), mono.get("r2_pfo"))
    pso = _min_r2(mix.get("r2_pso"), mono.get("r2_pso"))
    if pfo is None and pso is None:
        return None
    if pso is None:
        return "PFO"
    if pfo is None:
        return "PSO"
    return "PFO" if pfo >= pso else "PSO"


# ── Kinetic reference capacity ─────────────────────────────────────────────────

KIN_REF_FIT = "fit"
KIN_REF_LAST = "last"


def best_global_model(records: list[dict]) -> str:
    """
    'PFO' or 'PSO', whichever describes the whole dataset better on average.

    The choice is global on purpose: a capacity ratio only compares like with
    like if the same model was used for the mixture and for its mono-element
    reference, as the method requires.
    """
    def mean_r2(key):
        vals = [r[key] for r in records
                if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("-inf")
    return "PFO" if mean_r2("r2_pfo") > mean_r2("r2_pso") else "PSO"


def apply_kinetic_reference(records: list[dict], mode: str) -> list[dict]:
    """
    Return *records* with ``qe_mg`` set to the chosen kinetic reference.

    ``KIN_REF_FIT``  — the fitted q_e of the globally better model (PFO/PSO);
                       this is what the method document and the project report
                       use, and it is the capacity extrapolated to equilibrium.
    ``KIN_REF_LAST`` — the value actually measured at the most stable point
                       (the longest contact time), which makes no extrapolation
                       but underestimates q_e if the curve has not levelled off.
    """
    if mode != KIN_REF_FIT:
        return records
    model = best_global_model(records)
    key = "qe_pfo_mg" if model == "PFO" else "qe_pso_mg"
    out = []
    for r in records:
        q = r.get(key)
        rec = dict(r)
        if q is not None and np.isfinite(q):
            rec["qe_mg"] = float(q)
            rec["qe_source"] = f"{model} fit"
        else:
            rec["qe_source"] = "most stable point (no fit)"
        out.append(rec)
    return out


# ── Level 2 — real competitive effect, mixture vs mono ─────────────────────────

def level2(mix_records: list[dict], mono_records: list[dict]) -> pd.DataFrame:
    """
    Compare each metal in a mixture against its own mono-element reference.

    Ratios R are basis-invariant for capacity (the molar mass cancels between
    numerator and denominator of the same metal), so they are computed once
    from the mass basis.  k₁ is basis-invariant by definition; k₂ is converted
    consistently on both sides, which again cancels.

    Kinetic records additionally get the guide's derived indicators: the
    velocity ratio R_v (rate-constant ratio of the better-fitting model), the
    PSO initial-rate ratio R_h = R_k2·R_q², and the half-time ratio.
    Isotherm records get the affinity ratio R_K = K_L^mix / K_L^mono and the
    R_L pair.
    """
    mono_by_metal = {r["metal"]: r for r in mono_records}

    rows = []
    for rec in mix_records:
        metal = rec["metal"]
        ref = mono_by_metal.get(metal)
        if ref is None:
            rows.append({
                "Metal": metal,
                "System": rec.get("system"),
                "Note": "no mono-element reference",
            })
            continue

        R_q  = _safe_ratio(rec.get("qe_mg"), ref.get("qe_mg"))
        R_k1 = _safe_ratio(rec.get("k1"),    ref.get("k1"))
        R_k2 = _safe_ratio(rec.get("k2_mg"), ref.get("k2_mg"))

        row = {
            "Metal":            metal,
            "System":           rec.get("system"),
            "Solution":         rec.get("solution"),
            "qe mono (mg/g)":   ref.get("qe_mg"),
            "qe mix (mg/g)":    rec.get("qe_mg"),
            "R_q":              R_q,
            "S_q = 1 − R_q":    None if R_q is None else 1.0 - R_q,
            "S_q (%)":          None if R_q is None else 100.0 * (1.0 - R_q),
            "R_k1":             R_k1,
            "S_k1":             None if R_k1 is None else 1.0 - R_k1,
            "R_k2":             R_k2,
            "S_k2":             None if R_k2 is None else 1.0 - R_k2,
            "Effect":           interpret_rq(R_q),
            "min R² PFO":       _min_r2(rec.get("r2_pfo"), ref.get("r2_pfo")),
            "min R² PSO":       _min_r2(rec.get("r2_pso"), ref.get("r2_pso")),
        }

        # Kinetic extras — velocity ratio, initial rate, half-time
        if rec.get("k1") is not None or rec.get("k2_mg") is not None:
            model = best_kinetic_model(rec, ref)
            R_v = R_k1 if model == "PFO" else (R_k2 if model == "PSO" else None)
            row["best model"] = model
            row["R_v"] = R_v
            row["R_h"] = _safe_ratio(initial_rate_h(rec), initial_rate_h(ref))
            t_mix = half_time(rec, model) if model else None
            t_mono = half_time(ref, model) if model else None
            row["t½ mono (min)"] = t_mono
            row["t½ mix (min)"] = t_mix
            row["t½ ratio"] = _safe_ratio(t_mix, t_mono)

        # Isotherm extras — affinity and favourability
        if rec.get("KL_mg") is not None or ref.get("KL_mg") is not None:
            row["R_K"] = _safe_ratio(rec.get("KL_mg"), ref.get("KL_mg"))
            row["R_L mono"] = ref.get("RL")
            row["R_L mix"] = rec.get("RL")

        rows.append(row)

    return pd.DataFrame(rows)


def _min_r2(a, b):
    """Worst of the two fits behind a ratio — the quality that limits it."""
    vals = [v for v in (a, b)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
    return min(vals) if vals else None


def poor_fit_metals(l2: pd.DataFrame, threshold: float = 0.9) -> dict[str, list]:
    """
    Rows of *l2* whose rate-constant ratios rest on a fit worse than
    *threshold*, so the UI can warn instead of presenting a spurious number.
    """
    flagged = {"PFO": [], "PSO": []}
    if l2.empty:
        return flagged
    for model, col, ratio in (("PFO", "min R² PFO", "R_k1"),
                              ("PSO", "min R² PSO", "R_k2")):
        if col not in l2.columns or ratio not in l2.columns:
            continue
        bad = l2[(l2[col].notna()) & (l2[col] < threshold) & (l2[ratio].notna())]
        flagged[model] = [
            f"{r.Metal} in {r.System}" for r in bad.itertuples(index=False)
        ]
    return flagged


def interpret_rq(R_q, tol: float = 0.05) -> str:
    """R_q > 1 synergy · R_q ≈ 1 non-interaction · R_q < 1 antagonism."""
    if R_q is None or (isinstance(R_q, float) and not np.isfinite(R_q)):
        return "—"
    if R_q > 1.0 + tol:
        return "synergy"
    if R_q < 1.0 - tol:
        return "antagonism"
    return "non-interaction"


# ── Level 3 — pairwise competition across matrices ─────────────────────────────

def level3(records: list[dict]) -> pd.DataFrame:
    """
    Capacity of each metal across every matrix it appears in, so that the
    strongest competitor can be identified.

    Requires at least one binary matrix to be informative: comparing
    q(mono) → q(each binary) → q(ternary) is what separates "who competes with
    whom" from the overall suppression seen in the full mixture.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "Metal":    r["metal"],
        "System":   r.get("system"),
        "Solution": r.get("solution"),
        "qe (mg/g)": r.get("qe_mg"),
    } for r in records])

    pivot = df.pivot_table(index="Metal", columns="System",
                           values="qe (mg/g)", aggfunc="mean")

    # Order columns Mono → Bi → Tri, then by label
    sol_of = {r.get("system"): r.get("solution") for r in records}
    pivot = pivot[sorted(pivot.columns,
                         key=lambda s: solution_sort_key(sol_of.get(s, "?")) + (s,))]
    return pivot


def level3_pairwise(records: list[dict]) -> pd.DataFrame:
    """
    Guide §6.1–6.2: effect of each co-ion j on metal i, measured in the
    binary (i+j) against the mono of i:

        R_q,i(j) = q_i^(i+j) / q_i^mono        S_i(j) = 1 − R_q,i(j)

    Returns one row per (i, j) with the competitor rank per metal
    (1 = strongest competitor, i.e. the largest S).
    """
    mono = {r["metal"]: r.get("qe_mg") for r in records
            if r.get("solution") == "Mono" and r.get("qe_mg") is not None}

    rows = []
    for r in records:
        if r.get("solution") != "Bi" or r.get("qe_mg") is None:
            continue
        i = r["metal"]
        others = [m for m in str(r.get("system", "")).split("+") if m and m != i]
        if len(others) != 1 or i not in mono:
            continue
        j = others[0]
        R = _safe_ratio(r["qe_mg"], mono[i])
        if R is None:
            continue
        rows.append({
            "Metal i":        i,
            "Co-ion j":       j,
            "System":         r.get("system"),
            "q mono (mg/g)":  mono[i],
            "q binary (mg/g)": r["qe_mg"],
            "R_q,i(j)":       R,
            "S_i(j)":         1.0 - R,
            "Effect of j on i": interpret_rq(R),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["competitor rank"] = (
            df.groupby("Metal i")["S_i(j)"]
              .rank(ascending=False, method="min").astype(int)
        )
        df = df.sort_values(["Metal i", "competitor rank"]).reset_index(drop=True)
    return df


def level3_reciprocity(pairwise: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    """
    Guide §6.3: inside each binary pair compare S_i(j) with S_j(i).
    S_j(i) > S_i(j) means i dominates j (i takes more capacity from j than
    the other way round) — the direct link to the selectivity factor α.
    """
    if pairwise.empty:
        return pd.DataFrame()

    s = {(r["Metal i"], r["Co-ion j"]): r["S_i(j)"]
         for _, r in pairwise.iterrows()}
    done, rows = set(), []
    for (i, j) in s:
        if (j, i) not in s or (j, i) in done:
            continue
        done.update({(i, j), (j, i)})
        S_ij, S_ji = s[(i, j)], s[(j, i)]
        if abs(S_ji - S_ij) <= tol:
            verdict = "mutual / symmetric competition"
        elif S_ji > S_ij:
            verdict = f"{i} dominates {j}"
        else:
            verdict = f"{j} dominates {i}"
        rows.append({
            "Pair":                f"{i}+{j}",
            "S_i(j) — loss of first metal":  S_ij,
            "S_j(i) — loss of second metal": S_ji,
            "Dominance":           verdict,
        })
    return pd.DataFrame(rows)


def level3_cooperativity(records: list[dict]) -> pd.DataFrame:
    """
    Guide §6.4: predict each metal's capacity in the ternary assuming the
    binary suppressions act independently (multiplicatively), then compare
    with the measured value:

        q_i^(tern,pred) = q_i^mono · Π_j R_q,i(j)
        C_i = q_i^(tern,meas) / q_i^(tern,pred)

    C ≈ 1 independent effects · C < 1 competitors reinforce each other ·
    C > 1 competitors overlap (less suppression than predicted).
    """
    pw = level3_pairwise(records)
    if pw.empty:
        return pd.DataFrame()
    mono = {r["metal"]: r.get("qe_mg") for r in records
            if r.get("solution") == "Mono" and r.get("qe_mg") is not None}

    rows = []
    for r in records:
        if r.get("solution") != "Tri" or r.get("qe_mg") is None:
            continue
        i = r["metal"]
        co_ions = [m for m in str(r.get("system", "")).split("+")
                   if m and m != i]
        if i not in mono or not co_ions:
            continue
        ratios = []
        for j in co_ions:
            hit = pw[(pw["Metal i"] == i) & (pw["Co-ion j"] == j)]
            if hit.empty:
                ratios = None
                break
            ratios.append(float(hit["R_q,i(j)"].iloc[0]))
        if ratios is None:
            rows.append({"Metal": i, "System": r.get("system"),
                         "Note": "not all binaries of this metal available"})
            continue
        pred = mono[i]
        for R in ratios:
            pred *= R
        C = _safe_ratio(r["qe_mg"], pred)
        if C is None:
            continue
        if C > 1.05:
            verdict = "competitors overlap (less suppression than predicted)"
        elif C < 0.95:
            verdict = "competitors reinforce (more suppression than predicted)"
        else:
            verdict = "independent effects (binaries explain the ternary)"
        rows.append({
            "Metal":               i,
            "System":              r.get("system"),
            "q predicted (mg/g)":  pred,
            "q measured (mg/g)":   r["qe_mg"],
            "C_i":                 C,
            "Reading":             verdict,
        })
    return pd.DataFrame(rows)


def henry_selectivity_matrix(records: list[dict]) -> pd.DataFrame:
    """
    Guide §5.2, Henry form: α_i/j = (q_max,i·K_L,i)/(q_max,j·K_L,j) from the
    Langmuir parameters of one matrix — concentration-independent, hence the
    recommended form for isotherms.  Rows are metal i, columns metal j.
    """
    b = {}
    for r in records:
        qmax, KL = r.get("qe_mg"), r.get("KL_mg")
        if qmax is not None and KL is not None:
            b[r["metal"]] = qmax * KL
    if len(b) < 2:
        return pd.DataFrame()
    metals = list(b)
    return pd.DataFrame(
        {j: [_safe_ratio(b[i], b[j]) for i in metals] for j in metals},
        index=metals,
    )


def cross_axis(kin_l2: pd.DataFrame, iso_l2: pd.DataFrame,
               tol: float = 0.05) -> pd.DataFrame:
    """
    Guide §8: cross the velocity effect (R_v, kinetics) with the capacity /
    affinity effect (R_q^iso, R_K, isotherm) to tell whether the competition
    is kinetic (transport), thermodynamic (sites/affinity), both, or absent.
    Joined on (Metal, System) — the same matrix must exist in both studies.
    """
    if kin_l2.empty or iso_l2.empty:
        return pd.DataFrame()

    def klass(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        if v > 1.0 + tol:
            return "up"
        if v < 1.0 - tol:
            return "down"
        return "flat"

    kin = kin_l2.set_index(["Metal", "System"])
    rows = []
    for _, r in iso_l2.iterrows():
        key = (r.get("Metal"), r.get("System"))
        if key not in kin.index:
            continue
        k = kin.loc[key]
        R_v = k.get("R_v")
        R_qi = r.get("R_q")
        cv, cq = klass(R_v), klass(R_qi)
        if cv is None or cq is None:
            reading = "—"
        elif cv == "flat" and cq == "flat":
            reading = "no relevant competition (non-interaction)"
        elif cv == "down" and cq == "flat":
            reading = "KINETIC competition only: slower, same final capacity"
        elif cv == "flat" and cq == "down":
            reading = "THERMODYNAMIC competition: same speed, less capacity/affinity"
        elif cv == "down" and cq == "down":
            reading = "STRONG antagonism: slower and with less capacity"
        elif cv == "up" and cq == "up":
            reading = "synergy: faster and with more capacity"
        else:
            reading = "mixed effect — check both axes and the uncertainty"
        rows.append({
            "Metal":         key[0],
            "System":        key[1],
            "R_v (kinetics)": R_v,
            "R_q iso (q_max)": R_qi,
            "R_K (affinity)": r.get("R_K"),
            "Interpretation": reading,
        })
    return pd.DataFrame(rows)


def strongest_competitors(records: list[dict]) -> pd.DataFrame:
    """
    For every metal, the matrix with the lowest R_q relative to its mono value
    — i.e. the co-solute that suppressed it most.
    """
    mono = {r["metal"]: r["qe_mg"] for r in records
            if r.get("solution") == "Mono" and r.get("qe_mg") is not None}

    rows = []
    for metal, q_mono in mono.items():
        worst_sys, worst_R = None, None
        for r in records:
            if r["metal"] != metal or r.get("solution") == "Mono":
                continue
            R = _safe_ratio(r.get("qe_mg"), q_mono)
            if R is None:
                continue
            if worst_R is None or R < worst_R:
                worst_R, worst_sys = R, r.get("system")
        if worst_sys is not None:
            rows.append({
                "Metal":                metal,
                "qe mono (mg/g)":       q_mono,
                "Strongest competitor": worst_sys,
                "R_q":                  worst_R,
                "S_q (%)":              100.0 * (1.0 - worst_R),
            })
    return pd.DataFrame(rows)


def charge_balance(records: list[dict]) -> pd.DataFrame:
    """
    Total adsorbed amount per system on the three bases (the data behind the
    CEC-signature figure).  Σ meq/g staying roughly constant across the
    mixtures, above the mono systems, indicates a shared pool of exchangeable
    charge.
    """
    totals: dict[str, dict] = {}
    for r in records:
        sys_ = r.get("system")
        q = r.get("qe_mg")
        if sys_ is None or q is None:
            continue
        t = totals.setdefault(sys_, {"Solution": r.get("solution"),
                                     "Metals": 0, "mass": 0.0,
                                     "mol": 0.0, "charge": 0.0, "ok": True})
        bases = _q_bases(q, r["metal"])
        t["Metals"] += 1
        t["mass"] += bases["mass"]
        if bases["mol"] is None or bases["charge"] is None:
            t["ok"] = False
        else:
            t["mol"] += bases["mol"]
            t["charge"] += bases["charge"]

    rows = []
    for sys_, t in totals.items():
        rows.append({
            "System":        sys_,
            "Solution":      t["Solution"],
            "Metals":        t["Metals"],
            "Σ q (mg/g)":    t["mass"],
            "Σ q (mmol/g)":  t["mol"] if t["ok"] else None,
            "Σ q (meq/g)":   t["charge"] if t["ok"] else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Metals", "System"]).reset_index(drop=True)
    return df


# ── Availability of each level ─────────────────────────────────────────────────

def availability(records: list[dict]) -> dict:
    """
    Report which indicator levels the supplied dataset can support.

    Returns a dict with the solution types present and a per-level flag plus a
    short reason, so the UI can explain why a level is unavailable rather than
    silently omitting it.
    """
    solutions = {r.get("solution") for r in records if r.get("solution")}
    has_mono  = "Mono" in solutions
    has_bi    = "Bi" in solutions
    has_multi = bool(solutions - {"Mono"})

    return {
        "solutions": sorted(solutions, key=solution_sort_key),
        "level1": {
            "ok": has_multi,
            "why": "" if has_multi else
                   "needs at least one multi-element (Bi/Tri) matrix",
        },
        "level2": {
            "ok": has_mono and has_multi,
            "why": "" if (has_mono and has_multi) else
                   "needs both a mono-element reference and a multi-element matrix",
        },
        "level3": {
            "ok": has_bi and has_mono,
            "why": "" if (has_bi and has_mono) else
                   "needs binary (Bi) matrices alongside the mono references",
        },
    }
