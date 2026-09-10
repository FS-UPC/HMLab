"""
units.py — IUPAC molar-mass lookup table and mg ↔ mol unit handling.

Concentrations always enter the app in mg/L (the XRF instrument output) and
capacities are always computed first in mg/g.  This module turns those mass
based values into the molar and charge-equivalent bases used to compare
different elements fairly:

    C [mmol/L] = C [mg/L] / M          q [mmol/g] = q [mg/g] / M
    C [meq/L]  = C [mmol/L] · |z|      q [meq/g]  = q [mmol/g] · |z|

The per-tab "Base" selector offers all three: Mass (mg), Mol (mmol) and
Charge (meq, using the ionic charge z of the assumed aqueous species).

The lookup table lives in ``data/iupac_molar_masses.csv`` and is keyed by
**symbol**, not by atomic number or element name, so that non-elemental
species (compounds with a known molar mass) can be added later by appending a
row with a new symbol.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

# ── Lookup table ───────────────────────────────────────────────────────────────

MASS_TABLE_PATH = os.path.join(os.path.dirname(__file__), "data",
                               "iupac_molar_masses.csv")

MASS_TABLE_COLUMNS = {
    "symbol":      "Symbol",
    "Z":           "Z",
    "name":        "Element",
    "species":     "Assumed species",
    "charge":      "Ionic charge (z)",
    "molar_mass":  "Molar mass (g/mol)",
    "uncertainty": "Uncertainty (±)",
}


@st.cache_data(show_spinner=False)
def load_mass_table() -> pd.DataFrame:
    """
    Return the molar-mass lookup table as a DataFrame.

    Columns: symbol, Z, name, species, charge, molar_mass, uncertainty.
    ``molar_mass`` and ``charge`` are NaN where IUPAC defines no value
    (e.g. Tc, Pm, Ra have no standard atomic weight; noble gases have no
    ionic charge).
    """
    if not os.path.isfile(MASS_TABLE_PATH):
        return pd.DataFrame(columns=list(MASS_TABLE_COLUMNS))

    df = pd.read_csv(MASS_TABLE_PATH)
    df["molar_mass"] = pd.to_numeric(df["molar_mass"], errors="coerce")
    df["charge"]     = pd.to_numeric(df["charge"],     errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _mass_index() -> dict[str, dict]:
    """Symbol (case-folded) → {M, z, species, name} for O(1) lookups."""
    df = load_mass_table()
    index: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        sym = str(row.symbol).strip()
        if not sym:
            continue
        index[sym.lower()] = {
            "symbol":  sym,
            "M":       None if pd.isna(row.molar_mass) else float(row.molar_mass),
            "z":       None if pd.isna(row.charge)     else int(row.charge),
            "species": getattr(row, "species", ""),
            "name":    getattr(row, "name", ""),
        }
    return index


def molar_mass(symbol: str) -> float | None:
    """Molar mass of *symbol* in g/mol (= mg/mmol), or None if unknown."""
    entry = _mass_index().get(str(symbol).strip().lower())
    return entry["M"] if entry else None


def ionic_charge(symbol: str) -> int | None:
    """Signed ionic charge z of the assumed aqueous species, or None."""
    entry = _mass_index().get(str(symbol).strip().lower())
    return entry["z"] if entry else None


def assumed_species(symbol: str) -> str:
    entry = _mass_index().get(str(symbol).strip().lower())
    return entry["species"] if entry else ""


def missing_molar_masses(symbols) -> list[str]:
    """Return the subset of *symbols* that have no usable molar mass."""
    missing, seen = [], set()
    for sym in symbols:
        key = str(sym).strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        if molar_mass(key) is None:
            missing.append(key)
    return missing


def warn_missing_masses(symbols, context: str = "") -> list[str]:
    """
    Show a Streamlit warning naming every symbol in *symbols* whose molar mass
    is unknown, and return that list.  Callers use a non-empty return value to
    fall back to the mass basis instead of producing wrong numbers.
    """
    missing = missing_molar_masses(symbols)
    if missing:
        where = f" {context}" if context else ""
        st.warning(
            f"⚠️ Molar mass not available for **{', '.join(missing)}** — "
            f"the molar/charge based calculation{where} cannot be performed. "
            f"Add a row for it in `data/iupac_molar_masses.csv` "
            f"(see the table at the end of the **Help** tab)."
        )
    return missing


def missing_charges(symbols) -> list[str]:
    """Return the subset of *symbols* that have no ionic charge defined."""
    missing, seen = [], set()
    for sym in symbols:
        key = str(sym).strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        if ionic_charge(key) is None:
            missing.append(key)
    return missing


# ── Unit basis ─────────────────────────────────────────────────────────────────

MASS_BASIS = "mg"
MOLAR_BASIS = "mol"
CHARGE_BASIS = "charge"

_BASIS_OPTIONS = {
    MASS_BASIS:   "Mass  (mg/L · mg/g)",
    MOLAR_BASIS:  "Mol  (mmol/L · mmol/g)",
    CHARGE_BASIS: "Charge  (meq/L · meq/g)",
}

_GLOBAL_BASIS_KEY = "unit_basis"

_UNIT_PREFIX = {MASS_BASIS: "mg", MOLAR_BASIS: "mmol", CHARGE_BASIS: "meq"}

BASIS_ORDER = [MASS_BASIS, MOLAR_BASIS, CHARGE_BASIS]

_BASIS_NAMES = {MASS_BASIS: "Mass", MOLAR_BASIS: "Mol", CHARGE_BASIS: "Charge"}


def basis_name(basis: str) -> str:
    """Human-readable name of *basis*: 'Mass', 'Mol' or 'Charge'."""
    return _BASIS_NAMES.get(basis, basis)


def basis_tag(basis: str) -> str:
    """Lower-case slug of *basis*, for file names and column suffixes."""
    return basis_name(basis).lower()


def conc_unit(basis: str) -> str:
    """Concentration unit label for *basis*."""
    return f"{_UNIT_PREFIX.get(basis, 'mg')}/L"


def cap_unit(basis: str) -> str:
    """Adsorption-capacity unit label for *basis*."""
    return f"{_UNIT_PREFIX.get(basis, 'mg')}/g"


def k2_unit(basis: str) -> str:
    """Pseudo-2nd-order rate-constant unit label for *basis*."""
    return f"g·{_UNIT_PREFIX.get(basis, 'mg')}⁻¹·min⁻¹"


def kf_unit(basis: str) -> str:
    """Freundlich K_F unit label for *basis*."""
    u = _UNIT_PREFIX.get(basis, "mg")
    return f"({u}/g)·(L/{u})^(1/n)"


def kl_unit(basis: str) -> str:
    """Langmuir K_L unit label for *basis*."""
    return f"L/{_UNIT_PREFIX.get(basis, 'mg')}"


def unit_selector(tab_key: str, label: str = "Base") -> str:
    """
    Render the Mass / Mol / Charge basis selector for one tab and return the
    chosen basis.

    Each tab owns its own widget (Streamlit keys must be unique) but the last
    choice is remembered globally, so switching the basis in one tab becomes
    the default in the others.
    """
    global_basis = st.session_state.get(_GLOBAL_BASIS_KEY, MASS_BASIS)
    options = list(BASIS_ORDER)
    index = options.index(global_basis) if global_basis in options else 0

    basis = st.radio(
        label,
        options=options,
        index=index,
        format_func=lambda b: _BASIS_OPTIONS[b],
        horizontal=True,
        key=f"unit_basis_{tab_key}",
        help="Mass compares removed mass (engineering, regulatory limits); "
             "Mol compares the number of adsorbed ions — the fair basis when "
             "molar masses differ; Charge (mol × |z|) compares the ionic "
             "charge carried — relevant when the mechanism is ion exchange "
             "or surface-charge neutralisation.",
    )
    st.session_state[_GLOBAL_BASIS_KEY] = basis
    return basis


def basis_factor(symbol: str, basis: str) -> float | None:
    """
    Multiplier that converts a mass-based quantity (mg/L or mg/g) of *symbol*
    to *basis*: 1 (mass), 1/M (mol), |z|/M (charge).

    Returns None when the conversion is impossible — molar mass unknown, or
    ionic charge unknown/zero on the charge basis — so callers can fall back
    to the mass basis instead of producing a wrong number.
    """
    if basis == MASS_BASIS:
        return 1.0
    M = molar_mass(symbol)
    if M is None or M <= 0:
        return None
    if basis == MOLAR_BASIS:
        return 1.0 / M
    if basis == CHARGE_BASIS:
        z = ionic_charge(symbol)
        if z is None or z == 0:
            return None
        return abs(z) / M
    return None


def resolve_basis(symbols, basis: str) -> tuple[str, list[str]]:
    """
    The basis a whole table or chart can actually use, plus the symbols that
    blocked it.

    If any element involved lacks the molar mass (or the valence, on the charge
    basis) the whole table falls back to the mass basis: a table whose header
    says mmol/g but whose rows mix mmol/g and mg/g would be worse than one that
    simply stays on the mass basis and says so.
    """
    if basis == MASS_BASIS:
        return MASS_BASIS, []
    seen, blocked = set(), []
    for sym in symbols:
        key = str(sym).strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        if basis_factor(key, basis) is None:
            blocked.append(key)
    return (MASS_BASIS if blocked else basis), sorted(blocked)


def warn_missing_for_basis(symbols, basis: str, context: str = "") -> list[str]:
    """
    Warn about every symbol that cannot be converted to *basis* (missing molar
    mass, or missing ionic charge on the charge basis) and return that list.
    """
    if basis == MASS_BASIS:
        return []

    missing, seen = [], set()
    for sym in symbols:
        key = str(sym).strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        if basis_factor(key, basis) is None:
            missing.append(key)

    if missing:
        need = ("molar mass" if basis == MOLAR_BASIS
                else "molar mass and/or ionic charge (z)")
        where = f" {context}" if context else ""
        st.warning(
            f"⚠️ The {need} is not available for **{', '.join(missing)}** — "
            f"the {_BASIS_OPTIONS[basis].split('  ')[0].lower()}-basis "
            f"calculation{where} cannot be performed. Add the value in "
            f"`data/iupac_molar_masses.csv` (see the table at the end of the "
            f"**Help** tab)."
        )
    return missing


# ── Conversions ────────────────────────────────────────────────────────────────

def to_basis(value_mg, symbol: str, basis: str):
    """
    Convert a mass-based quantity (mg/L or mg/g) to *basis* — ÷M for mol,
    ×|z|/M for charge.  The factor is the same for concentrations and
    capacities, so this one helper serves both.  Returns None when the
    conversion data for *symbol* is missing.
    """
    if basis == MASS_BASIS:
        return value_mg
    f = basis_factor(symbol, basis)
    if f is None or value_mg is None:
        return None
    return value_mg * f


def to_charge_basis(value_mg, symbol: str):
    """
    Convert a mass-based quantity (mg/L or mg/g) to the charge-equivalent
    basis (meq/L or meq/g):  value / M · |z|.  None if M or z is unknown.
    """
    M = molar_mass(symbol)
    z = ionic_charge(symbol)
    if M is None or M <= 0 or z is None or value_mg is None:
        return None
    return value_mg / M * abs(z)


def convert_series(values, symbol: str, basis: str):
    """Vectorised :func:`to_basis` for a pandas Series / numpy array."""
    if basis == MASS_BASIS:
        return values
    f = basis_factor(symbol, basis)
    if f is None:
        return None
    return values * f


def convert_k2(k2_mass, symbol: str, basis: str):
    """
    Convert a pseudo-2nd-order rate constant from the mass basis
    (g·mg⁻¹·min⁻¹) to *basis*: k₂ scales inversely with the capacity factor,
    so k2 → k2/f (k2·M for mol, k2·M/|z| for charge).

    k₁ (PFO) is basis-invariant and needs no conversion.
    """
    if basis == MASS_BASIS or k2_mass is None:
        return k2_mass
    f = basis_factor(symbol, basis)
    if f is None:
        return None
    return k2_mass / f


# Optional models whose parameters contain a logarithm of the concentration and
# therefore have no exact reprojection between bases.
NON_COVARIANT_MODELS = {"Halsey", "Harkins-Jura"}


def scale_kinetic_params(model: str, params: dict, f: float) -> tuple[dict, bool]:
    """
    Convert optional-kinetic-model parameters fitted on the mass basis to a
    basis whose capacity factor is *f* (i.e. q_new = f·q_mass).

    Returns ``(converted, exact)``; ``exact`` is False for models whose
    parameters are not scale-covariant, in which case the mass-basis values are
    returned unchanged and must be labelled as such.
    """
    p = dict(params or {})
    exact = (f == 1.0) or (model not in NON_COVARIANT_MODELS)
    if not p or f == 1.0:
        return p, exact

    if model == "Pseudo-nth-order":
        # q_t = q_e − (q_e^{1−n} − k_n(1−n)t)^{1/(1−n)} ⇒ k_n scales as f^{1−n}
        n = p.get("n")
        if "qe" in p:
            p["qe"] *= f
        if n is not None and "kn" in p:
            p["kn"] *= f ** (1.0 - n)
        if "h" in p:
            p["h"] *= f
        return p, True

    if model == "Elovich":
        # q_t = (1/β)·ln(1+αβt) is invariant under α→αf, β→β/f
        if "alpha" in p:
            p["alpha"] *= f
        if "beta" in p:
            p["beta"] /= f
        return p, True

    if model == "Avrami fraccionary":
        if "qe" in p:
            p["qe"] *= f            # k_AV and n_AV are basis-invariant
        return p, True

    if model == "IntraParticle Diffusion (IPD)":
        for k in ("kid", "C"):
            if k in p:
                p[k] *= f
        return p, True

    if model == "Segmented IPD":
        for k in ("k1", "C1", "k2", "C2"):
            if k in p:
                p[k] *= f           # the breakpoint t1 is a time, invariant
        return p, True

    return p, False


def scale_isotherm_params(model: str, params: dict, f: float) -> tuple[dict, bool]:
    """
    Convert optional-isotherm-model parameters fitted on the mass basis to a
    basis whose factor is *f* (q_new = f·q_mass and C_new = f·C_mass).

    The Halsey and Harkins--Jura models contain a logarithm of the
    concentration and are therefore not scale-covariant: their parameters are
    returned unchanged with ``exact = False``.
    """
    p = dict(params or {})
    exact = (f == 1.0) or (model not in NON_COVARIANT_MODELS)
    if not p or f == 1.0:
        return p, exact

    if model == "Sips (Langmuir-Freundlich)":
        if "qm" in p:
            p["qm"] *= f
        if "KS" in p:
            p["KS"] /= f
        return p, True

    if model == "Tóth":
        if "qm" in p:
            p["qm"] *= f
        if "KT" in p:
            p["KT"] /= f
        return p, True

    if model == "Janovics":
        if "qmax" in p:
            p["qmax"] *= f
        if "KJ" in p:
            p["KJ"] /= f
        return p, True

    # Halsey, Harkins–Jura: q = g(log C) — no exact reprojection
    return p, False


def convert_kl(KL_mass, symbol: str, basis: str):
    """
    Convert a Langmuir affinity constant from L/mg to *basis*: K_L is the
    reciprocal of a concentration, so it scales inversely with the factor
    (K_L·M for mol, K_L·M/|z| for charge).
    """
    if basis == MASS_BASIS or KL_mass is None:
        return KL_mass
    f = basis_factor(symbol, basis)
    if f is None:
        return None
    return KL_mass / f
