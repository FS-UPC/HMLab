# HMLab — Heavy-Metal Adsorption Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A [Streamlit](https://streamlit.io) application for the automated kinetic,
isotherm and competitiveness analysis of mono- and multi-element heavy-metal
batch adsorption experiments.

HMLab takes X-ray fluorescence (XRF) concentration reports and carries them
through the whole analytical chain without any programming: it parses the
experimental metadata from the filenames, computes adsorption capacities by
mass balance, fits kinetic and equilibrium isotherm models by nonlinear least
squares, reports every quantity on three reporting bases (mass, molar and
charge-equivalent), and derives a three-level set of competitiveness
indicators for multi-element datasets.

## Installation and use

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens in your default browser. No account, server or
programming knowledge is required — point HMLab at a data folder or upload a
ZIP archive and work through the tabs.

Requires Python 3.10 or newer.

## Input data

Data are loaded from a folder or an uploaded ZIP archive. Two input modes are
supported and can be mixed freely in one analysis:

| Mode | One file per… | Format | Example |
|------|---------------|--------|---------|
| **1** | experimental point | XRF CSV report | `AK2Pb100Cu100M10BTO1t030R1.csv` |
| **2** | complete series | Excel workbook (metadata + data table) | `SK3PbCdCuM10BTO1.xlsx` |

Prefixes are `AK`/`SK` for kinetics and `AI`/`SI` for isotherms. All
experimental metadata — the metals present, their initial concentrations, the
adsorbent code and dosage, the contact time, the replica index and the
solution type — is encoded in the filename, using `p` as the decimal separator
(`6p6` = 6.6 g/L).

Multi-metal files are processed one metal at a time, keeping the solution type
(Mono / Bi / Tri) attached to each series, so the same metal measured alone and
measured in a mixture is never merged — that bookkeeping is what makes the
competitiveness indicators possible. The complete naming conventions and the
Mode 2 workbook layout are documented in the app's **Help** tab.

### Example datasets

- `tests/` — experimental XRF reports (mono-element Cu and Pb kinetic and
  isotherm series, and a binary Pb+Cu series), Mode 1.
- `hmlabfiles/` — simulated Pb/Cu/Cr and Pb/Cd/Cu benchmark suites in Mode 1
  and Mode 2 (three mono-element, three binary and one ternary series each,
  for both kinetics and isotherms).

## Reporting bases — the "Base" selector

Concentrations always enter in mg/L, and every tab can reproject the results
onto three bases: **Mass** (mg/L · mg/g), **Mol** (mmol/L · mmol/g) and
**Charge** (meq/L · meq/g). Conversions use the IUPAC standard atomic weights
and ionic charges in `data/iupac_molar_masses.csv` (keyed by symbol, and
extendable with custom species — see `data/README.md`).

Every model is fitted **once**, on the mass basis, and the parameters are
reprojected algebraically onto the other bases. Switching the basis therefore
never re-runs the experiment, and the dimensionless indicators are identical on
all three bases.

The choice of basis matters physically: comparing metals by mass answers *how
much contaminant was removed*, whereas comparing them by mole answers *how many
adsorption sites were occupied* — and the two can rank the same metals in
opposite orders.

## Analysis

**Kinetics** — up to seven models: pseudo-first-order, pseudo-second-order,
Elovich, Avrami fractionary, pseudo-*n*th-order, intraparticle diffusion and
its segmented extension. **Isotherms** — up to seven models: Langmuir,
Freundlich, Sips, Tóth, Halsey, Harkins–Jura and Janovics. All are fitted by
bounded nonlinear least squares (`scipy.optimize.curve_fit`), reported with
*R*² and mean absolute error, and accompanied by linearised diagnostic plots.

### Competitiveness indicators

Following the implicit-competition approach — each metal is fitted with the
same single-component models inside each matrix, and competition is read
afterwards by comparing the fitted parameters between matrices:

- **Level 1** — within one mixture: adsorbed fractions (mass / mol / charge),
  distribution coefficients *K*_d, selectivity factors α, rankings.
- **Level 2** — mixture versus mono-element reference: capacity ratio *R*_q and
  suppression *S*_q, rate ratios *R*_k1 / *R*_k2 / *R*_v / *R*_h, half-time
  *t*½, affinity ratio *R*_K, favourability *R*_L.
- **Level 3** — pairwise, in binary matrices: suppression *S*_i(j), competitor
  ranking, pair dominance, binary→ternary cooperativity *C*_i.
- **Cross-axis** — kinetics against isotherm, classifying the competition as
  kinetic, thermodynamic, strong antagonism or synergy.

## Tabs

| Tab | Purpose |
|-----|---------|
| **Kinetics** | Model fitting, linearised diagnostics, grouped Mono/Bi/Tri comparisons |
| **Isotherms** | Model fitting, linearised diagnostics, grouped Mono/Bi/Tri comparisons |
| **Indicators** | The three-level competitiveness framework and its figures |
| **Download Results** | Plain-text report and CSV export of every table and plotted quantity |
| **Help** | Full documentation, naming conventions and the searchable IUPAC table |

## Project layout

```
app.py              entry point (data source, sidebar, tab layout)
utils.py            Mode 1 parsers, XRF CSV reader, kinetic models
mode2.py            Mode 2 workbook parser/reader
units.py            IUPAC lookup table + Mass/Mol/Charge conversion
indicators.py       competitiveness indicators (Levels 1-3, cross-axis)
plots.py            grouped Mono/Bi/Tri comparison plots
indicator_plots.py  competitiveness figures
exports.py          downloadable tables for any reporting basis
tabs/               one module per tab
data/               iupac_molar_masses.csv (Z = 1-92, extendable)
tests/              experimental sample datasets
hmlabfiles/         simulated benchmark datasets
```

## Citing HMLab

If you use HMLab in your research, please cite the accompanying SoftwareX
article:

> F. Suñol, F. G. Velasco, J. E. García, *HMLab: a Streamlit application for
> the automated kinetic, isotherm and competitiveness analysis of mono- and
> multi-element heavy-metal batch adsorption experiments*, SoftwareX (2026).

## Contributing

Bug reports, feature requests and pull requests are welcome via the GitHub
issue tracker. The modular architecture makes it straightforward to add new
kinetic or isotherm models, instrument parsers, or indicators.

## License

Released under the MIT License — see [LICENSE](LICENSE).
