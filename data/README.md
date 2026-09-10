# `iupac_molar_masses.csv`

Lookup table used by HMLab for every mg ↔ mmol ↔ meq conversion
(see `units.py` and the **Help** tab).

## Source

Standard atomic weights from **IUPAC / CIAAW** — <https://www.ciaaw.org>,
*Atomic Weights of the Elements 2021*, including the 2024 revisions
(Gd, Lu, Zr). The table covers elements **Z = 1 to 92** (hydrogen to
uranium); elements beyond uranium have no IUPAC standard atomic weight.

## Columns

| Column | Meaning |
|--------|---------|
| `symbol` | Chemical symbol — the key the app looks up |
| `Z` | Atomic number |
| `name` | Element name |
| `species` | Assumed predominant aqueous species in an adsorption context |
| `charge` | Ionic charge *z* of that species (used for the Charge basis) |
| `molar_mass` | Standard atomic weight, g/mol (= mg/mmol) |
| `uncertainty` | Uncertainty of the standard atomic weight |

`molar_mass` and `charge` are empty where IUPAC defines no value: elements
with no stable isotope of constant terrestrial composition (Tc, Pm, and
those beyond Bi) have no standard atomic weight, and noble gases have no
ionic charge.

## Extending the table

The table is keyed by `symbol`, **not** by atomic number or element name, so
a new species — including a compound with a known molar mass — is added by
appending a row with its own symbol, molar mass and ionic charge. Nothing
else in the code needs to change.
