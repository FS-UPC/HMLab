"""
utils.py — shared helpers: XRF CSV reading, filename parsing and kinetic models.

HMLab accepts two complementary input formats, which may be mixed in one
analysis:

    Mode 1  one XRF CSV report per experimental point   (AK… / AI… files)
    Mode 2  one Excel workbook per complete series      (SK… / SI… files)

This module covers Mode 1 and the model definitions shared by both; the Mode 2
workbook reader lives in :mod:`mode2`.
"""

import os
import re

import numpy as np
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression


# ═════════════════════════════════════════════════════════════════════════════
# p-decimal helpers
# ═════════════════════════════════════════════════════════════════════════════
#
# Filenames cannot contain a dot outside the extension, so decimals are written
# with the letter "p": 6p6 → 6.6.

def _p_to_float(s: str) -> float:
    """Convert a p-decimal token to float: '10p5' → 10.5,  '10' → 10.0."""
    return float(s.replace("p", "."))


def _p_time_to_minutes(s: str) -> float:
    """
    Convert a time token to minutes.

    Tokens containing 'p' represent hours → multiplied by 60
      e.g. '24p0' → 24.0 h → 1440 min,  '0p25' → 0.25 h → 15 min.
    Plain integer tokens represent minutes directly
      e.g. '120' → 120 min,  '5' → 5 min.
    """
    val = _p_to_float(s)
    return val * 60.0 if "p" in s.lower() else val


# ═════════════════════════════════════════════════════════════════════════════
# UI helper
# ═════════════════════════════════════════════════════════════════════════════

def folder_picker(label, default, key):
    """
    Sidebar folder picker: text input + live subfolder selector.
    Returns the chosen path.
    """
    path = st.sidebar.text_input(label, value=default, key=f"{key}_input")

    if os.path.isdir(path):
        try:
            subdirs = sorted(
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
            )
        except PermissionError:
            subdirs = []

        if subdirs:
            choice = st.sidebar.selectbox(
                f"Subfolders of `{os.path.basename(path) or path}`",
                options=["— stay here —"] + subdirs,
                key=f"{key}_sub",
            )
            if choice != "— stay here —":
                path = os.path.join(path, choice)
                try:
                    subdirs2 = sorted(
                        d for d in os.listdir(path)
                        if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
                    )
                except PermissionError:
                    subdirs2 = []
                if subdirs2:
                    choice2 = st.sidebar.selectbox(
                        f"Subfolders of `{os.path.basename(path)}`",
                        options=["— stay here —"] + subdirs2,
                        key=f"{key}_sub2",
                    )
                    if choice2 != "— stay here —":
                        path = os.path.join(path, choice2)
        else:
            st.sidebar.caption("No subfolders found.")
    else:
        st.sidebar.caption("Path does not exist.")

    if path != st.session_state.get(f"{key}_input", path):
        st.sidebar.caption(f"{path}")

    return path


# ═════════════════════════════════════════════════════════════════════════════
# Kinetic models
# ═════════════════════════════════════════════════════════════════════════════

def model_first_order(t, qe, k1):
    """Pseudo-first-order:  q(t) = qe · (1 − exp(−k1·t))."""
    return qe * (1.0 - np.exp(-k1 * t))


def model_second_order(t, qe, k2):
    """Pseudo-second-order:  q(t) = k2·qe²·t / (1 + k2·qe·t)."""
    denom = 1.0 + k2 * qe * t
    return (k2 * qe**2 * t) / np.where(denom == 0, 1e-12, denom)


def fit_kinetic(model_fn, t_arr, q_arr, p0, bounds):
    """Wrapper around curve_fit; returns (popt, r2, mae) or (None, None, None)."""
    try:
        popt, _ = curve_fit(model_fn, t_arr, q_arr,
                            p0=p0, maxfev=20_000, bounds=bounds)
        q_pred = model_fn(t_arr, *popt)
        return (popt,
                float(r2_score(q_arr, q_pred)),
                float(mean_absolute_error(q_arr, q_pred)))
    except Exception:
        return None, None, None


def linear_first_order(t_arr, q_arr, qe):
    """
    Linear form of pseudo-first-order:  ln(qe − q) = ln(qe) − k1·t
    Uses a qe slightly above max(q_arr) when qe leaves no room.
    """
    qe_safe = max(qe, q_arr.max() * 1.001) * 1.001
    y   = np.log(qe_safe - q_arr)
    reg = LinearRegression().fit(t_arr.reshape(-1, 1), y)
    return float(reg.coef_[0]), float(reg.intercept_), qe_safe


def linear_second_order(t_arr, q_arr):
    """Linear form of pseudo-second-order:  t/q = 1/(k2·qe²) + t/qe."""
    mask = q_arr > 1e-12
    if mask.sum() < 2:
        return None, None
    y   = t_arr[mask] / q_arr[mask]
    reg = LinearRegression().fit(t_arr[mask].reshape(-1, 1), y)
    return float(reg.coef_[0]), float(reg.intercept_)


# ═════════════════════════════════════════════════════════════════════════════
# Mode 1 — XRF CSV file reader
# ═════════════════════════════════════════════════════════════════════════════

_CSV_EXTENSIONS = (".csv",)


def _extract_csv_fields(line: str) -> list:
    """
    Split a raw XRF CSV line by comma and strip all double-quote characters.

    Each line in the XRF export is wrapped in outer double-quotes and uses
    doubled inner quotes for escaping.  Removing all quote chars after
    splitting gives clean field values (e.g. 'Average' at col 0, '23.4' at col 3).
    """
    return [part.replace('"', '').strip() for part in line.split(',')]


def _parse_hmlab_csv_content(content: str) -> list | None:
    """
    Parse XRF CSV text content and return per-metal concentration data.

    The file has two sections separated by a blank line:
      Section 1 — concentration in ppm (the one we need)
      Section 2 — raw counts in cps/μA (ignored)

    Within section 1:
      Line 0 : metal names  at columns 3, 5, 7 …
      Line 1 : units        at columns 3, 5, 7 …
      …data rows…
      Row starting with 'Average'  → average concentration per metal
      Row starting with 'Std dev.' → standard deviation per metal

    Returns a list of dicts (one per metal):
        [{"name": str, "unit": str, "avg": float, "std": float}, …]
    Returns None if the content cannot be parsed.
    """
    # Use only the first section (stop at first blank line)
    first_section: list[str] = []
    for line in content.splitlines():
        if not line.strip():
            if first_section:
                break
        else:
            first_section.append(line)

    if len(first_section) < 3:
        return None

    names_fields = _extract_csv_fields(first_section[0])
    units_fields = _extract_csv_fields(first_section[1])

    # Metal names / units are at positions 3, 5, 7, …
    metal_names: list[str] = []
    metal_units: list[str] = []
    idx = 3
    while idx < len(names_fields):
        name = names_fields[idx]
        unit = units_fields[idx] if idx < len(units_fields) else "ppm"
        if name:
            metal_names.append(name)
            metal_units.append(unit or "ppm")
        idx += 2

    if not metal_names:
        return None

    avg_vals: list = [None] * len(metal_names)
    std_vals: list = [0.0]  * len(metal_names)

    for line in first_section:
        fields = _extract_csv_fields(line)
        if not fields:
            continue
        label = fields[0].lower()

        if label.startswith("average"):
            for i in range(len(metal_names)):
                pos = 3 + 2 * i
                try:
                    val_str = fields[pos] if pos < len(fields) else ""
                    if val_str and val_str.upper() != "ND":
                        avg_vals[i] = float(val_str)
                    else:
                        avg_vals[i] = 0.0   # ND = not detected → 0
                except (ValueError, IndexError):
                    pass

        elif label.startswith("std dev"):
            for i in range(len(metal_names)):
                pos = 3 + 2 * i
                try:
                    val_str = fields[pos] if pos < len(fields) else ""
                    if val_str and val_str.upper() != "ND":
                        std_vals[i] = float(val_str)
                except (ValueError, IndexError):
                    pass

    if all(v is None for v in avg_vals):
        return None

    return [
        {
            "name":  metal_names[i],
            "unit":  metal_units[i],
            "avg":   avg_vals[i] if avg_vals[i] is not None else 0.0,
            "std":   std_vals[i],
        }
        for i in range(len(metal_names))
    ]


def read_hmlab_csv(path: str) -> list | None:
    """
    Read an XRF CSV file from *path* and return per-metal concentration data.
    Returns a list of dicts or None on failure.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        return None
    return _parse_hmlab_csv_content(content)


def read_hmlab_csv_bytes(uploaded_file) -> list | None:
    """Version of :func:`read_hmlab_csv` for Streamlit uploaded files."""
    try:
        raw = uploaded_file.read()
        content = raw.decode("utf-8", errors="replace")
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    except Exception:
        return None
    return _parse_hmlab_csv_content(content)


def list_csv_files(folder: str, pattern=None) -> list:
    """
    Return sorted CSV files in *folder* whose name matches *pattern*
    (optional regex, case-insensitive).
    """
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(_CSV_EXTENSIONS)]
    if pattern:
        files = [f for f in files if re.search(pattern, f, re.IGNORECASE)]
    return sorted(files)


# ═════════════════════════════════════════════════════════════════════════════
# Mode 1 — filename parsers
# ═════════════════════════════════════════════════════════════════════════════
#
# Filename conventions:
#
#   Kinetics : AK{n}{Metal1}{C1}[{Metal2}{C2}…]{AdsCode}{AdsConc}t{time}R{rep}.csv
#              e.g.  AK1Cu30HIB6p6t005R1.csv    → 1 metal Cu@30 ppm, HIB@6.6 g/L, 5 min
#                    AK2Pb100Cr20MXB2t030R1.csv → 2 metals Pb@100 + Cr@20, MXB@2 g/L, 30 min
#
#   Isotherms: AI{n}{Metal1}{C1}[{Metal2}{C2}…]{AdsCode}{AdsConc}R{rep}.csv
#              e.g.  AI1Cu30HIB6p6R1.csv
#                    AI2Cu10Mn20HIB6p6R1.csv
#
#   n       = number of metals (single digit)
#   MetalX  = letter-only metal code (e.g. Cu, Pb, Mn)
#   CX      = initial metal concentration (mg/L), p-decimal allowed (6p6 → 6.6)
#   AdsCode = letter-only adsorbent code
#   AdsConc = adsorbent concentration (g/L), p-decimal allowed
#   time    = contact time (minutes, zero-padded integer)
#   rep     = replica number
#
# An optional matrix code M{xx} and pH token pH{y} may appear between the
# metals and the adsorbent; both are stripped by _strip_m_ph before the
# metal/adsorbent pairs are read.

_METAL_PAIR_RE = re.compile(r"([A-Za-z]+)(\d+(?:p\d+)?)")
_M_FIELD_RE    = re.compile(r"M(\d+(?:p\d+)?)", re.IGNORECASE)
_PH_FIELD_RE   = re.compile(r"pH(\d+(?:p\d+)?)", re.IGNORECASE)


def _strip_m_ph(middle: str):
    """
    Extract and strip the optional M{matrix} and pH{value} fields from the
    middle section of a Mode 1 filename (between the prefix/metals and the
    adsorbent+time+replica suffix).

    Returns (clean_middle, m_code, ph_val) where m_code is a string or None
    and ph_val is a float or None.

    M must be followed immediately by a digit (e.g. M01, M1), so metal codes
    like Mn or Mg are never mis-identified as the matrix field.
    """
    clean = middle

    m_match = _M_FIELD_RE.search(clean)
    m_code  = m_match.group(1) if m_match else None
    if m_match:
        clean = clean[:m_match.start()] + clean[m_match.end():]

    ph_match = _PH_FIELD_RE.search(clean)
    ph_val   = _p_to_float(ph_match.group(1)) if ph_match else None
    if ph_match:
        clean = clean[:ph_match.start()] + clean[ph_match.end():]

    return clean, m_code, ph_val


def parse_hmlab_filename(fname: str) -> dict | None:
    """
    Parse a Mode 1 CSV filename and return a parameter dict.

    The returned dict always contains:
        "type"      : "kinetics" or "isotherm"
        "n_metals"  : int   — number of metals encoded in the filename
        "metals"    : list of {"code": str, "N": float}  (one entry per metal)
        "N"         : float — initial concentration of the *first* metal (mg/L)
        "metal"     : str   — code of the first metal
        "AdsCode"   : str   — adsorbent identifier
        "AdsConc"   : float — adsorbent concentration (g/L)
        "R"         : int   — replica number
        "M"         : str or None   — matrix code
        "pH"        : float or None — pH

    Kinetics dicts additionally contain:
        "t"    : float — contact time (minutes)

    Isotherm dicts additionally contain:
        "Cads" : float — same value as AdsConc (alias kept for compatibility)

    Returns None if the filename does not match.
    """
    base = os.path.basename(fname)
    name = os.path.splitext(base)[0]   # strip .csv

    # ── Kinetics: AK{n}…t{time}R{rep} ────────────────────────────────────────
    m_ak = re.match(r"AK(\d)", name, re.IGNORECASE)
    if m_ak:
        n = int(m_ak.group(1))
        m_tr = re.search(r"t(\d+(?:p\d+)?)R(\d+)$", name, re.IGNORECASE)
        if not m_tr:
            return None
        time_str = m_tr.group(1)
        rep      = int(m_tr.group(2))
        middle   = name[len(m_ak.group(0)) : m_tr.start()]
        clean, m_code, ph_val = _strip_m_ph(middle)
        pairs    = _METAL_PAIR_RE.findall(clean)
        if len(pairs) < n + 1:
            return None
        metals   = [{"code": p[0], "N": _p_to_float(p[1])} for p in pairs[:n]]
        ads_code = pairs[n][0]
        ads_conc = _p_to_float(pairs[n][1])
        return {
            "type":     "kinetics",
            "n_metals": n,
            "metals":   metals,
            "N":        metals[0]["N"],
            "metal":    metals[0]["code"],
            "AdsCode":  ads_code,
            "AdsConc":  ads_conc,
            "t":        _p_time_to_minutes(time_str),
            "R":        rep,
            "M":        m_code,
            "pH":       ph_val,
        }

    # ── Isotherms: AI{n}…R{rep} ───────────────────────────────────────────────
    m_ai = re.match(r"AI(\d)", name, re.IGNORECASE)
    if m_ai:
        n = int(m_ai.group(1))
        m_rep = re.search(r"R(\d+)$", name, re.IGNORECASE)
        if not m_rep:
            return None
        rep    = int(m_rep.group(1))
        middle = name[len(m_ai.group(0)) : m_rep.start()]
        clean, m_code, ph_val = _strip_m_ph(middle)
        pairs  = _METAL_PAIR_RE.findall(clean)
        if len(pairs) < n + 1:
            return None
        metals   = [{"code": p[0], "N": _p_to_float(p[1])} for p in pairs[:n]]
        ads_code = pairs[n][0]
        ads_conc = _p_to_float(pairs[n][1])
        return {
            "type":     "isotherm",
            "n_metals": n,
            "metals":   metals,
            "N":        metals[0]["N"],
            "metal":    metals[0]["code"],
            "AdsCode":  ads_code,
            "AdsConc":  ads_conc,
            "Cads":     ads_conc,
            "R":        rep,
            "M":        m_code,
            "pH":       ph_val,
        }

    return None
