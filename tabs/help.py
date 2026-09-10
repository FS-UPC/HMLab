"""
tabs/help.py — Help tab (HMLab).
"""

import streamlit as st

from units import MASS_TABLE_COLUMNS, MASS_TABLE_PATH, load_mass_table


def render() -> None:

    st.subheader("📖 Help & Documentation")

    st.markdown("""
This application processes XRF (X-Ray Fluorescence) concentration data to
characterise heavy-metal adsorption onto solid adsorbents. The instrument CSV
files already contain calibrated concentration values (mg/L = ppm), so no
fluorescence calibration is required to obtain results.

---

### 🗺️ Tab overview

| Tab | Purpose | File prefix |
|-----|---------|-------------|
| **Kinetics** | Fit adsorption kinetics models from time-series data | `AK` / `SK` |
| **Isotherms** | Fit Langmuir / Freundlich isotherm models | `AI` / `SI` |
| **Indicators** | Competitiveness indicators across Mono / Bi / Tri solutions | — |
| **Download Results** | Export the report (TXT) and all data as CSVs | — |
| **Help** | This page: workflow, conventions and the IUPAC table | — |

---

### 🔬 Recommended workflow

#### 1 — Load data
Point the app to your experiment data folder in the sidebar, or upload a ZIP
archive containing your data files — Mode 1 XRF reports (`.csv`) and/or Mode 2
series workbooks (`.xlsx`), which may be mixed freely.  Sub-folders inside the
ZIP are supported — files will be flattened automatically.

#### 2 — Kinetics tab
1. Select (or upload) kinetics files (`AK*` Mode 1 and/or `SK*` Mode 2).
2. For each file the app:
   - Reads the measured concentration **Cₜ** (and its std) directly from the CSV.
   - Computes the adsorption capacity **q(t) = (N − Cₜ) / C_ads**, where N and
     C_ads are extracted from the filename.
   - Averages replicas that share the same (Metal, N, AdsCode, AdsConc, t) group.
3. Pseudo-1st and pseudo-2nd order models are fitted per series.
4. Additional kinetic models can be selected optionally.
5. Linearised diagnostic plots are shown alongside the main q(t) chart.

> **Multi-metal files**: when `n > 1` (e.g. `AK2…`), each metal in the file
> becomes an independent data series, processed exactly like a mono-element
> one.  The solution it came from (`Mono`, `Bi`, `Tri`) is kept alongside it so
> the same metal measured alone and in a mixture never gets merged.

#### 3 — Isotherms tab
1. Select (or upload) isotherm files (`AI*` Mode 1 and/or `SI*` Mode 2).
2. For each file the app:
   - Reads **Cₑ** (equilibrium concentration) and its std from the CSV.
   - Computes **qₑ = (N − Cₑ) / C_ads**.
   - Averages replicates grouped by (Metal, N, AdsCode, AdsConc).
3. Fits **Langmuir** and **Freundlich** models per (Metal × AdsCode × AdsConc) series.
4. Additional isotherm models can be selected optionally.
5. Linearised diagnostic plots are shown.

#### 4 — Indicators tab
Reads the fits produced by the two tabs above and reports the competitiveness
indicators (see below) as tables **and figures**.  Run Kinetics and/or
Isotherms first.

Pick a **reference mixture** and the charts show: the model fits of each metal
alone vs in that mixture; the Level 2 ratio bars (R_q, R_v, R_h for kinetics;
R_q, R_K for isotherms); the Level 3 chart of which co-ion suppresses which
metal; the cross reading of velocity against capacity; the adsorbed fraction on
the three bases; the mixture's curves drawn on all three bases side by side;
and the total adsorbed charge per system.

> For kinetics you can choose the **reference capacity**: the *fitted qₑ* of the
> better model (what the method document prescribes, since it extrapolates to
> equilibrium) or the *value measured at the most stable point* (no
> extrapolation, but it underestimates qₑ if the curve is still rising).

#### 5 — Download Results tab
1. Check the status tiles to confirm which analyses are complete.
2. Click **⬇️ Download results as TXT** for a full plain-text report. Its
   measured-data tables follow the **Base** selected above — headers and values
   alike — and every fitted parameter (qₑ, q_max, k₂, K_L, K_F, MAE) is
   additionally written on **all three bases** side by side; quantities that do
   not depend on the basis (k₁, n, R_L, R², and the indicators) are marked as
   such. A table whose elements cannot all be converted stays on the mass basis
   and says so, so one table never mixes units.
3. Choose the **Base** (Mass / Mol / Charge) and download the **CSV files**.
   Every kinetic and isotherm table is exported on that basis, which appears in
   the file name, in the column headers and in a `Basis` column; switch the
   Base and download again to get the same tables on another basis.
   Competitiveness indicators are dimensionless ratios, so they are exported
   once and are valid on every basis.

Every plotted quantity has a CSV: the data points (`*_q_vs_time_*`,
`*_qe_vs_ce_*`), the fitted model curves (`*_fitted_curves_*`), the fitted
parameters (`*_model_parameters_*`), the linearised diagnostics (`*_linear_*`)
and the optional-model parameters (`*_extra_models_*`). The grouped
Mono/Bi/Tri comparison figures re-plot the same series, so their data is the
corresponding data-point file filtered by its `Metal`, `System` and `Solution`
columns.

The Indicators-tab figures are covered too: the ratio and fraction charts come
from `indicators_*_level1/level2/level3_*` and `indicators_cross_axis`, the
total-charge chart from `indicators_*_charge_balance`, and the three-panel
charts from `kinetics_three_bases` / `isotherms_three_bases`, which carry the
mass, molar and charge capacities side by side in one table. Indicator files
are written for **both** studies and do not require opening the Indicators
tab.

---
""")

    _render_base_section()

    st.markdown("""
---

### 🏁 Competitiveness indicators

Adsorption from a mixture is compared with adsorption of the same metal alone,
following the **implicit competition** approach: each metal is fitted with the
same single-component models inside each matrix, and competition is read
afterwards by comparing the fitted parameters between matrices.

The reference quantity is **q at the most stable point** (longest contact time)
for kinetics, and **q_max** from the Langmuir fit for isotherms.

#### Level 1 — apparent competition (mixture only)

| Indicator | Formula | Answers |
|-----------|---------|---------|
| Mass fraction adsorbed | $f_i^{mass} = q_{e,i}^{mg/g} / \\sum_j q_{e,j}^{mg/g}$ | share of adsorbed **mass** |
| Molar fraction adsorbed | $f_i^{mol} = q_{e,i}^{mmol/g} / \\sum_j q_{e,j}^{mmol/g}$ | share of adsorbed **ions** |
| Charge fraction adsorbed | $f_i^{charge} = q_{e,i}^{meq/g} / \\sum_j q_{e,j}^{meq/g}$ | share of adsorbed **charge** |
| Distribution coefficient | $K_{d,i} = q_{e,i}/C_{e,i}$ | tendency to move to the solid |
| Selectivity factor | $\\alpha_{i/j} = K_{d,i}/K_{d,j}$ | α > 1 → adsorbent prefers *i* |

with $q^{meq/g} = q^{mmol/g} \\times |z|$, where $z$ is the ionic charge of the
assumed aqueous species (also in the table below).  The charge basis matters
when adsorption works by ion exchange or surface-charge neutralisation: one
Al³⁺ neutralises the same charge as three Na⁺.

#### Level 2 — real competitive effect (mixture vs mono)

| Indicator | Formula |
|-----------|---------|
| Capacity ratio | $R_q = q_e^{mix} / q_e^{mono}$ |
| Competitive suppression | $S_q = 1 - R_q$ |
| PFO rate-constant ratio | $R_{k1} = k_1^{mix} / k_1^{mono}$ |
| PFO suppression | $S_{k1} = 1 - R_{k1}$ |
| PSO rate-constant ratio | $R_{k2} = k_2^{mix} / k_2^{mono}$ |
| PSO suppression | $S_{k2} = 1 - R_{k2}$ |

**Reading R**:  R > 1 → synergy · R ≈ 1 → non-interaction · R < 1 → antagonism.

#### Level 3 — pairwise competition

Comparing the same metal across mono → each binary → ternary identifies *which*
co-solute competes with it.  This needs binary matrices; with only mono and
ternary data the full mixture cannot be decomposed into pairs.

> All these indicators are ratios or fractions and so are **dimensionless** —
> R_q, S_q and α are identical whichever unit basis is selected.  The
> fractions differ between bases by construction, which is exactly why all
> three are reported side by side.

---

### 📐 Kinetic models

**Pseudo-1st-order (PFO)**:
$$q(t) = q_e \\left(1 - e^{-k_1 t}\\right)$$

**Pseudo-2nd-order (PSO)**:
$$q(t) = \\frac{k_2\\,q_e^2\\,t}{1 + k_2\\,q_e\\,t}$$

| Parameter | Units | Description |
|-----------|-------|-------------|
| qₑ | mg/g | Equilibrium adsorption capacity |
| k₁ | min⁻¹ | Pseudo-1st-order rate constant |
| k₂ | g·mg⁻¹·min⁻¹ | Pseudo-2nd-order rate constant |

---

### 📐 Isotherm models

**Langmuir**:
$$q_e = \\frac{q_{max} K_L C_e}{1 + K_L C_e}$$

Separation factor:
$$R_L = \\frac{1}{1 + K_L C_0}$$

**Freundlich**:
$$q_e = K_F C_e^{1/n}$$

| Parameter | Units | Description |
|-----------|-------|-------------|
| q_max | mg/g | Maximum adsorption capacity |
| K_L | L/mg | Langmuir constant |
| R_L | — | Separation factor (adsorption favorability) |
| K_F | (mg/g)(L/mg)^(1/n) | Freundlich constant |
| n | — | Adsorption intensity |

**Interpretation:**
- Langmuir:  0 < R_L < 1 → favorable;  R_L = 1 → linear;  R_L > 1 → unfavorable
- Freundlich:  1/n < 1 → favorable;  1/n = 1 → linear;  1/n > 1 → unfavorable

---

### ❓ Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| Files not listed in the folder selector | Wrong folder path in the sidebar |
| Filename cannot be parsed | Does not match the expected pattern (see naming convention below) |
| "Fit did not converge" | Too few data points, or data are noisy / inconsistent |
| q(t) or qₑ is negative | Cₜ or Cₑ > N — check CSV data and filename parameters |
| ND values in CSV | "Not Detected" is treated as 0.0 mg/L |

---

### 🗂 File naming convention

Two input modes are supported, and they can be mixed in one analysis:

- **Mode 1** — one XRF CSV per experimental point (`AK…`/`AI…` files below).
- **Mode 2** — one Excel workbook per complete series (`SK…`/`SI…` files),
  holding all times (kinetics) or all C₀ levels (isotherm) plus replicates
  and a metadata block. See the Mode 2 section further down.

The app extracts metadata directly from filenames. All values use the **`p`**
character as a decimal separator (e.g. `6p6` = 6.6).  Replica number is always
the last token before the extension: `R{n}`.

The number of metals `n` is encoded in the prefix (e.g. `AK1` = 1 metal,
`AK2` = 2 metals). The middle section of the filename contains **n** metal
code+concentration pairs followed by one adsorbent code+concentration pair.

#### Kinetics files — `AK{n}{Metal1}{C1}[{Metal2}{C2}…]{AdsCode}{AdsConc}t{time}R{n}.csv`

| Token | Meaning | Example |
|-------|---------|---------|
| `AK{n}` | Prefix + number of metals | `AK1`, `AK2` |
| `{MetalX}{CX}` | Metal code + initial concentration (mg/L), repeated n times | `Cu30`, `Mn50` |
| `{AdsCode}{AdsConc}` | Adsorbent code + concentration (g/L) | `HIB6p6` |
| `t{time}` | Contact time (minutes, 3 digits) | `t005`, `t090` |
| `R{n}` | Replica number | `R1` |

Examples:
- `AK1Cu30HIB6p6t005R1.csv` → 1 metal: Cu at 30 mg/L, HIB adsorbent at 6.6 g/L, 5 min, replica 1
- `AK2Pb100Cr20MXB2t030R1.csv` → 2 metals: Pb@100 + Cr@20 mg/L, MXB at 2 g/L, 30 min, replica 1

#### Isotherm files — `AI{n}{Metal1}{C1}[{Metal2}{C2}…]{AdsCode}{AdsConc}R{n}.csv`

| Token | Meaning | Example |
|-------|---------|---------|
| `AI{n}` | Prefix + number of metals | `AI1`, `AI2` |
| `{MetalX}{CX}` | Metal code + initial concentration (mg/L), repeated n times | `Cu10` |
| `{AdsCode}{AdsConc}` | Adsorbent code + concentration (g/L) | `HIB6p6` |
| `R{n}` | Replica number | `R1` |

Examples:
- `AI1Cu50HIB6p6R1.csv` → 1 metal: Cu at N=50 mg/L, HIB at 6.6 g/L, replica 1
- `AI2Cu30Mn50HIB6p6R1.csv` → 2 metals: Cu@30 + Mn@50 mg/L, HIB at 6.6 g/L, replica 1

> Different N values with the same (Metal, AdsCode, AdsConc) form one isotherm series.

---

### 📗 Mode 2 — one workbook per series (`SK*`/`SI*` .xlsx)

A Mode 2 file gathers **every point** of one kinetic or isotherm series in a
single Excel workbook.  The filename identifies the *series*; whatever varies
point to point (time or C₀), the replicates and the metadata live **inside**
the file (sheet `Dados`).  Concentrations never appear in the name — only the
metals present.

#### Name — `S{K|I}{x}{Metals}M{xx}[pH{y}]{Ads}{conc}.xlsx`

| Token | Meaning | Example |
|-------|---------|---------|
| `SK` / `SI` | Kinetic series / Isotherm series | `SK`, `SI` |
| `{x}` | Number of metals, then x chemical symbols with **no** concentrations between them | `2PbCd` |
| `M{xx}` | Matrix code (NaCl molarity, 2 digits); NaCl and pH are encoded here | `M10` |
| `pH{y}` | Optional pH | `pH3` |
| `{Ads}{conc}` | Adsorbent code (3 letters) + concentration in g/L | `BTO1` = BTO at 1 g/L |

Examples: `SK1PbM10BTO1.xlsx` (mono kinetics), `SK2PbCdM10BTO1.xlsx` (binary),
`SI3PbCdCuM10BTO1.xlsx` (ternary isotherm), `SK2CuCrM10pH3BTO1.xlsx` (with pH).

#### Workbook layout — sheet `Dados`

A metadata block (label / value pairs), then a `TABELA DE DADOS` marker, then
the data table:

| Study | Columns |
|-------|---------|
| Kinetics (`SK`) | `t_min`, `rep`, then per metal: `<Metal>_C0_mgL`, `<Metal>_C0_std`, `<Metal>_Ct_mgL`, `<Metal>_Ct_std` |
| Isotherm (`SI`) | `ponto`, `rep`, then per metal: `<Metal>_C0_mgL`, `<Metal>_C0_std`, `<Metal>_Ce_mgL`, `<Metal>_Ce_std` |

Rules:
- Concentrations are **always mg/L** (= ppm); q and the molar basis are
  computed by the app from the IUPAC molar masses.
- No V or m needed — the adsorbent dose (g/L) already is m/V.
- Kinetics: one row per (time, replica), C₀ fixed. Isotherm: one row per
  (point, replica), C₀ varies per point.

#### Uncertainty (the `_std` columns)

The metadata field **“Formato de incerteza (std)”** controls how the `_std`
columns are read:

| Code | Meaning |
|------|---------|
| `SIM-ABS` | Uncertainties present, in **absolute** values (mg/L) |
| `SIM-PCT` | Uncertainties present, as **percentages** (converted to mg/L by the app) |
| `NAO` | No uncertainty — the `_std` columns are ignored |

> The ZIP name is free; only the workbook names inside follow the standard.
> Multi-metal series are processed exactly like Mode 1: each metal becomes an
> independent Mono/Bi/Tri-tagged series feeding the same fits, grouped plots
> and competitiveness indicators.

---
""")

    _render_mass_table()


# ── Base — Mass vs Mol vs Charge ───────────────────────────────────────────────

def _render_base_section() -> None:
    st.markdown("### ⚖️ Base — Mass vs Mol vs Charge")
    st.markdown(
        "Every tab except this one has a **Base** selector with three "
        "options: **Mass** (mg/L · mg/g), **Mol** (mmol/L · mmol/g) and "
        "**Charge** (meq/L · meq/g).  Measurements always enter the app in "
        "mg/L (the instrument output) and capacities are computed as mg/g; "
        "the selector is a *lens* that reprojects those values without "
        "recalculating the experiment:"
    )
    st.latex(
        r"C\,[\text{mmol/L}] = \frac{C\,[\text{mg/L}]}{M}"
        r"\qquad "
        r"q\,[\text{mmol/g}] = \frac{q\,[\text{mg/g}]}{M}"
    )
    st.latex(
        r"C\,[\text{meq/L}] = C\,[\text{mmol/L}]\cdot|z|"
        r"\qquad "
        r"q\,[\text{meq/g}] = q\,[\text{mmol/g}]\cdot|z|"
    )
    st.markdown(
        "where **M** is the molar mass in g/mol (numerically equal to "
        "mg/mmol) and **z** is the ionic charge of the assumed aqueous "
        "species — both taken from the *Molar mass* and *Ionic charge (z)* "
        "columns of the IUPAC table at the bottom of this page.\n\n"
        "Because the experiment is fitted once and only reprojected, the "
        "model parameters follow the same rule:"
    )
    st.markdown("""
| Quantity | Mass | Mol | Charge |
|----------|------|-----|--------|
| qₑ, q_max | mg/g | mmol/g = mg/g ÷ M | meq/g = mmol/g × ∣z∣ |
| Cₜ, Cₑ | mg/L | mmol/L = mg/L ÷ M | meq/L = mmol/L × ∣z∣ |
| k₁ (PFO) | min⁻¹ | **unchanged** | **unchanged** — k₁ is basis-invariant |
| k₂ (PSO) | g·mg⁻¹·min⁻¹ | k₂ × M (g·mmol⁻¹·min⁻¹) | k₂ × M ÷ ∣z∣ (g·meq⁻¹·min⁻¹) |
| K_L (Langmuir) | L/mg | K_L × M (L/mmol) | K_L × M ÷ ∣z∣ (L/meq) |
""")
    st.markdown(
        "**Which base to use?**\n\n"
        "- **Mass** answers *how much contaminant mass was removed* — the "
        "right question for engineering and for comparison with "
        "environmental limits in mg/L.\n"
        "- **Mol** answers *how many ions occupied the adsorption sites* — "
        "the fair comparison between elements with very different molar "
        "masses, and the basis recommended for competition analysis.\n"
        "- **Charge** answers *how much ionic charge was adsorbed* — "
        "relevant when the mechanism is ion exchange or surface-charge "
        "neutralisation: one Al³⁺ neutralises the same surface charge as "
        "three Na⁺.  For a set of same-valence metals (e.g. Pb²⁺, Cd²⁺, "
        "Cu²⁺, all z = 2) the charge basis is simply 2× the molar one and "
        "gives the same relative picture; it only tells a different story "
        "with mixed valences (Al³⁺, Cr³⁺…) or oxyanions.\n"
    )
    st.markdown(
        "> If a metal's molar mass — or, on the Charge base, its ionic "
        "charge — is missing from the table, the app shows a warning naming "
        "that element and keeps its series on the mass basis rather than "
        "producing a wrong number."
    )


# ── IUPAC molar-mass lookup table ──────────────────────────────────────────────

def _render_mass_table() -> None:
    st.divider()
    st.markdown("### ⚛️ IUPAC molar-mass lookup table")
    st.markdown(
        "Standard atomic weights from IUPAC / CIAAW, used for every mg ↔ mol "
        "conversion in the app.  The table covers elements **Z = 1 to 92** "
        "(hydrogen to uranium); elements beyond uranium have no IUPAC standard "
        "atomic weight."
    )

    df = load_mass_table()
    if df.empty:
        st.error(
            f"Lookup table not found at `{MASS_TABLE_PATH}`. "
            "Molar conversions are unavailable until it is restored."
        )
        return

    st.caption(
        "The table is keyed by **symbol**, not by atomic number or element "
        "name.  To support a new species — including a compound with a known "
        "molar mass — append a row to `data/iupac_molar_masses.csv` with its "
        "own symbol, molar mass and ionic charge; nothing else needs to change."
    )

    search = st.text_input(
        "Filter by symbol or element name",
        key="help_mass_filter",
        placeholder="e.g. Cu, Pb, lead…",
    ).strip()

    view = df.rename(columns=MASS_TABLE_COLUMNS)
    if search:
        mask = (
            view["Symbol"].str.contains(search, case=False, na=False) |
            view["Element"].str.contains(search, case=False, na=False)
        )
        view = view[mask]
        if view.empty:
            st.info(f"No entry matches “{search}”.")
            return

    st.dataframe(view, width='stretch', hide_index=True)

    no_mass = df[df["molar_mass"].isna()]["symbol"].tolist()
    if no_mass:
        st.caption(
            f"No standard atomic weight is defined for **{', '.join(no_mass)}** "
            f"(elements without a stable isotope of constant terrestrial "
            f"composition).  Selecting mol units for one of these produces a "
            f"warning and the affected series stay on the mass basis."
        )

    st.download_button(
        "⬇️  Download the molar-mass table (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="iupac_molar_masses.csv",
        mime="text/csv",
    )
