"""
mode2.py — Mode 2 input: one Excel workbook per complete series.

While Mode 1 keeps one XRF CSV per experimental point, a Mode 2 file gathers
every point of one kinetic or isotherm series (all times or all C0 levels,
with replicates) in a single .xlsx, plus a metadata block.

Filename convention (LEIAME sheet of the sample files):

    SK{x}{Metals}M{xx}[pH{y}]{Ads}{conc}.xlsx      kinetic series
    SI{x}{Metals}M{xx}[pH{y}]{Ads}{conc}.xlsx      isotherm series

    x       number of metals; then x chemical symbols concatenated with NO
            concentrations between them (SK2PbCdM10BTO1 → Pb, Cd)
    Mxx     matrix code (NaCl molarity, 2 digits); pH y optional
    Ads     3-letter adsorbent code + concentration in g/L (BTO1 = 1 g/L)

Workbook layout — sheet "Dados":

    metadata rows (label in col A, value in col B)
    …
    TABELA DE DADOS
    t_min | rep | Pb_C0_mgL | Pb_C0_std | Pb_Ct_mgL | Pb_Ct_std | …   kinetics
    ponto | rep | Pb_C0_mgL | Pb_C0_std | Pb_Ce_mgL | Pb_Ce_std | …   isotherm

The "Formato de incerteza (std)" metadata controls the _std columns:
SIM-ABS = absolute values (mg/L), SIM-PCT = percentages, NAO = no
uncertainty (columns ignored).  Concentrations are always mg/L; q and the
molar basis are derived downstream from the adsorbent dose (g/L = m/V).
"""

from __future__ import annotations

import os
import re
import unicodedata

import pandas as pd

from utils import _p_to_float

MODE2_EXTENSIONS = (".xlsx",)

_MODE2_RE = re.compile(
    r"^S([KI])"                 # SK kinetic / SI isotherm
    r"(\d)"                     # number of metals
    r"((?:[A-Z][a-z]?)+?)"      # metal symbols, concatenated
    r"M(\d+(?:p\d+)?)"          # matrix code
    r"(?:pH(\d+(?:p\d+)?))?"    # optional pH
    r"([A-Za-z]{3})"            # adsorbent code (3 letters)
    r"(\d+(?:p\d+)?)$",         # adsorbent concentration (g/L)
)

_SYMBOL_RE = re.compile(r"[A-Z][a-z]?")


def parse_mode2_filename(fname: str) -> dict | None:
    """
    Parse a Mode 2 series filename; returns None when it does not match.

    The dict mirrors the Mode 1 parser keys where they overlap ("type",
    "n_metals", "metals", "AdsCode", "AdsConc", "M", "pH") so the tabs can
    treat both modes uniformly.  Per-metal concentrations are NOT in the
    name — they live inside the workbook.
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    m = _MODE2_RE.match(base)
    if not m:
        return None

    symbols = _SYMBOL_RE.findall(m.group(3))
    n = int(m.group(2))
    if len(symbols) != n:
        return None

    return {
        "type":     "kinetics" if m.group(1).upper() == "K" else "isotherm",
        "mode":     2,
        "n_metals": n,
        "metals":   [{"code": s} for s in symbols],
        "M":        m.group(4),
        "pH":       _p_to_float(m.group(5)) if m.group(5) else None,
        "AdsCode":  m.group(6).upper(),
        "AdsConc":  _p_to_float(m.group(7)),
    }


def list_mode2_files(folder: str, study: str) -> list[str]:
    """Sorted Mode 2 workbooks in *folder* for 'kinetics' or 'isotherm'."""
    prefix = "SK" if study == "kinetics" else "SI"
    if not os.path.isdir(folder):
        return []
    out = []
    for f in os.listdir(folder):
        if not f.lower().endswith(MODE2_EXTENSIONS) or f.startswith("~$"):
            continue
        meta = parse_mode2_filename(f)
        if meta and f.upper().startswith(prefix):
            out.append(f)
    return sorted(out)


# ── Workbook reader ────────────────────────────────────────────────────────────

def _ascii(s: str) -> str:
    """Casefold + strip accents, for tolerant metadata-label matching."""
    return (unicodedata.normalize("NFKD", str(s))
            .encode("ascii", "ignore").decode().lower())


def _std_mode(value: str) -> str:
    """Map the 'Formato de incerteza' metadata value to abs / pct / none."""
    v = _ascii(value)
    if "abs" in v:
        return "abs"
    if "pct" in v or "%" in v:
        return "pct"
    return "none"


def read_mode2_file(path_or_buffer, fname: str) -> dict:
    """
    Read a Mode 2 workbook and return::

        {"meta": {...}, "points": [ {metal, x, rep, C0, C0_std, C, C_std}, … ],
         "warnings": [str, …]}

    ``x`` is the contact time in minutes for kinetics and the point index for
    isotherms; C is Cₜ or Cₑ accordingly, always in mg/L.  Raises ValueError
    when the file cannot be interpreted, with a message meant for the user.
    """
    meta = parse_mode2_filename(fname)
    if meta is None:
        raise ValueError(
            "filename does not match the Mode 2 convention "
            "S<K|I><n><Metals>M<xx>[pH<y>]<Ads><conc>.xlsx"
        )

    try:
        book = pd.ExcelFile(path_or_buffer)
    except Exception as exc:
        raise ValueError(f"cannot open workbook ({exc})") from exc

    sheet = next((s for s in book.sheet_names if _ascii(s).startswith("dados")),
                 book.sheet_names[0])
    df = book.parse(sheet, header=None)

    # ── Split metadata block from the data table ──────────────────────────────
    table_row = None
    for i in range(len(df)):
        if _ascii(df.iat[i, 0]).strip().startswith("tabela de dados"):
            table_row = i
            break
    if table_row is None or table_row + 2 > len(df):
        raise ValueError("no 'TABELA DE DADOS' marker found in the Dados sheet")

    md: dict[str, str] = {}
    for i in range(table_row):
        key = _ascii(df.iat[i, 0]).strip()
        val = df.iat[i, 1]
        if key and pd.notna(val):
            md[key] = str(val).strip()

    def md_get(fragment: str, default=None):
        for k, v in md.items():
            if fragment in k:
                return v
        return default

    warnings: list[str] = []

    # Metadata cross-checks — the filename is authoritative, the metadata is
    # verified against it so silent mismatches surface instead of hiding.
    md_n = md_get("numero de metais")
    if md_n is not None:
        try:
            if int(float(md_n)) != meta["n_metals"]:
                warnings.append(
                    f"metadata says {md_n} metals but the filename encodes "
                    f"{meta['n_metals']} — using the filename"
                )
        except ValueError:
            pass
    md_metals = md_get("metais presentes")
    if md_metals:
        listed = [s.strip() for s in re.split(r"[,;+]", md_metals) if s.strip()]
        ours = [m_["code"] for m_ in meta["metals"]]
        if listed and [s.lower() for s in listed] != [s.lower() for s in ours]:
            warnings.append(
                f"metadata lists metals {listed} but the filename encodes "
                f"{ours} — using the filename"
            )
    md_dose = md_get("concentracao de adsorvente")
    if md_dose is not None:
        try:
            dose_val = float(str(md_dose).replace(",", "."))
            if abs(dose_val - meta["AdsConc"]) > 1e-9:
                warnings.append(
                    f"metadata dose {dose_val} g/L differs from the filename "
                    f"({meta['AdsConc']} g/L) — using the metadata value"
                )
                meta["AdsConc"] = dose_val
        except ValueError:
            pass

    std_mode = _std_mode(md_get("formato de incerteza", "NAO"))
    meta["std_mode"] = std_mode
    meta["study_label"] = md_get("tipo de estudo", "")

    # ── Data table ────────────────────────────────────────────────────────────
    header = [str(v).strip() if pd.notna(v) else "" for v in df.iloc[table_row + 1]]
    data = df.iloc[table_row + 2:].reset_index(drop=True)
    data.columns = range(data.shape[1])

    is_kin = meta["type"] == "kinetics"
    x_names = ("t_min",) if is_kin else ("ponto",)
    c_tag = "Ct" if is_kin else "Ce"

    cols = {name: i for i, name in enumerate(header) if name}
    x_col = next((cols[n] for n in x_names if n in cols), None)
    if x_col is None:
        raise ValueError(
            f"data table lacks the '{x_names[0]}' column expected for a "
            f"{'kinetic' if is_kin else 'isotherm'} series"
        )
    rep_col = cols.get("rep")

    metal_cols = {}
    for m_ in meta["metals"]:
        code = m_["code"]
        c0 = cols.get(f"{code}_C0_mgL")
        cx = cols.get(f"{code}_{c_tag}_mgL")
        if c0 is None or cx is None:
            raise ValueError(
                f"data table lacks the {code}_C0_mgL / {code}_{c_tag}_mgL "
                f"columns for metal {code} named in the file"
            )
        metal_cols[code] = {
            "C0": c0, "C": cx,
            "C0_std": cols.get(f"{code}_C0_std"),
            "C_std":  cols.get(f"{code}_{c_tag}_std"),
        }

    def num(row, col):
        if col is None:
            return None
        v = row.iloc[col]
        if pd.isna(v):
            return None
        try:
            return float(str(v).replace(",", "."))
        except ValueError:
            return None

    def std_of(raw_std, conc):
        if std_mode == "none" or raw_std is None:
            return 0.0
        if std_mode == "pct":
            return abs(raw_std) / 100.0 * conc if conc is not None else 0.0
        return abs(raw_std)

    points = []
    for _, row in data.iterrows():
        x = num(row, x_col)
        if x is None:                       # trailing blank / comment rows
            continue
        rep = num(row, rep_col)
        rep = int(rep) if rep is not None else 1
        for code, cc in metal_cols.items():
            C0 = num(row, cc["C0"])
            C = num(row, cc["C"])
            if C0 is None or C is None:
                warnings.append(
                    f"row {x_names[0]}={x:g}, rep={rep}: missing {code} "
                    f"concentration — point skipped"
                )
                continue
            points.append({
                "metal":  code,
                "x":      x,
                "rep":    rep,
                "C0":     C0,
                "C0_std": std_of(num(row, cc["C0_std"]), C0),
                "C":      C,
                "C_std":  std_of(num(row, cc["C_std"]), C),
            })

    if not points:
        raise ValueError("data table contains no usable rows")

    return {"meta": meta, "points": points, "warnings": warnings}
