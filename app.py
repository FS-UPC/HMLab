"""
app.py — main entry point for the HMLab heavy-metal adsorption application.

Run with:
    streamlit run app.py

Module structure:
    app.py              ← this file (data source + tab layout)
    utils.py            ← shared helpers (XRF CSV reader, Mode 1 parsers, models)
    mode2.py            ← Mode 2 workbook parser/reader
    units.py            ← IUPAC molar masses + Mass/Mol/Charge conversion
    indicators.py       ← competitiveness indicators (Levels 1–3, cross-axis)
    plots.py            ← grouped Mono/Bi/Tri comparison plots
    indicator_plots.py  ← competitiveness figures
    exports.py          ← downloadable tables for any reporting basis
    tabs/kinetics.py    ← Kinetics tab
    tabs/isotherms.py   ← Isotherms tab
    tabs/indicators.py  ← Indicators tab
    tabs/download.py    ← Download Results tab
    tabs/help.py        ← Help tab
"""

import os
import re
import zipfile
import tempfile
import shutil

import streamlit as st

# Import tab renderers
from tabs.kinetics      import render as render_kinetics
from tabs.isotherms     import render as render_isotherms
from tabs.indicators    import render as render_indicators
from tabs.download      import render as render_download
from tabs.help          import render as render_help
from utils import folder_picker

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HMLab",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 HMLab - Heavy-Metal Adsorption Analysis")


# ── ZIP helper ─────────────────────────────────────────────────────────────────

# Filename patterns used to classify files by tab (for the summary only).
# AK/AI = Mode 1 (one CSV per point); SK/SI = Mode 2 (one XLSX per series).
_FILE_PATTERNS = {
    "Kinetics":  re.compile(r"^(AK|SK)\d", re.IGNORECASE),
    "Isotherms": re.compile(r"^(AI|SI)\d", re.IGNORECASE),
}
_DATA_EXT = (".csv", ".xlsx")

DEFAULT_DATA_FOLDER = "tests"


def _extract_zip(uploaded_zip) -> tuple[str, dict[str, int]]:
    """
    Extract *uploaded_zip* (a Streamlit UploadedFile) into a fresh temp
    directory, flattening any sub-folder structure so all data files end up in
    the root of that directory.

    Returns:
        tmpdir  – absolute path to the temp directory
        summary – {tab_name: file_count} for the sidebar status badge
    """
    tmpdir = tempfile.mkdtemp(prefix="hmlab_zip_")
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        for member in zf.infolist():
            # Skip directories and macOS/Windows metadata entries
            if member.is_dir():
                continue
            fname = os.path.basename(member.filename)
            if not fname or fname.startswith((".", "__", "~$")):
                continue
            if not fname.lower().endswith(_DATA_EXT):
                continue
            dest = os.path.join(tmpdir, fname)
            # Avoid overwriting if two sub-folders contain a same-named file
            if os.path.exists(dest):
                base, ext = os.path.splitext(fname)
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(tmpdir, f"{base}_{counter}{ext}")
                    counter += 1
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    # Build summary
    all_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(_DATA_EXT)]
    summary: dict[str, int] = {}
    for tab, pat in _FILE_PATTERNS.items():
        n = sum(1 for f in all_files if pat.search(f))
        if n:
            summary[tab] = n
    unclassified = len(all_files) - sum(summary.values())
    if unclassified:
        summary["Other"] = unclassified

    return tmpdir, summary


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Data Source")

    data_source = st.radio(
        "Input mode",
        ["📁 Folder", "🗜️ ZIP file"],
        horizontal=True,
        key="data_source_mode",
    )

    # ── Mode A: folder picker ──────────────────────────────────────────────
    if data_source == "📁 Folder":
        # If we're switching back from ZIP mode, clean up the old temp dir
        _old_tmp = st.session_state.pop("_zip_tmpdir", None)
        if _old_tmp and os.path.isdir(_old_tmp):
            shutil.rmtree(_old_tmp, ignore_errors=True)
        st.session_state.pop("_zip_summary", None)

        EXPERIMENT_DIR = folder_picker(
            "Experiment data folder", DEFAULT_DATA_FOLDER, "exp"
        )

    # ── Mode B: ZIP upload ─────────────────────────────────────────────────
    else:
        uploaded_zip = st.file_uploader(
            "Upload experiment ZIP",
            type="zip",
            key="zip_uploader",
            help=(
                "Pack your Mode 1 XRF reports (.csv) and/or Mode 2 series "
                "workbooks (.xlsx) into a single ZIP. Sub-folders are "
                "supported — files will be flattened automatically."
            ),
        )

        if uploaded_zip is not None:
            # Re-extract only when a new file is uploaded (compare by name+size)
            zip_fingerprint = (uploaded_zip.name, uploaded_zip.size)
            if st.session_state.get("_zip_fingerprint") != zip_fingerprint:
                # Remove previous temp dir if any
                _old_tmp = st.session_state.get("_zip_tmpdir")
                if _old_tmp and os.path.isdir(_old_tmp):
                    shutil.rmtree(_old_tmp, ignore_errors=True)

                with st.spinner("Extracting ZIP…"):
                    tmpdir, summary = _extract_zip(uploaded_zip)

                st.session_state["_zip_tmpdir"]      = tmpdir
                st.session_state["_zip_summary"]     = summary
                st.session_state["_zip_fingerprint"] = zip_fingerprint

            EXPERIMENT_DIR = st.session_state["_zip_tmpdir"]

            # Show a concise file-count badge per tab
            summary = st.session_state.get("_zip_summary", {})
            if summary:
                st.success(f"✅ **{uploaded_zip.name}** loaded")
                total = sum(summary.values())
                st.caption(f"{total} data file(s) found.")
            else:
                st.warning("No recognised files found in ZIP.")

        else:
            # No ZIP uploaded yet — fall back to the default folder so tabs
            # don't crash, and inform the user.
            EXPERIMENT_DIR = DEFAULT_DATA_FOLDER
            st.info("Upload a ZIP file to load your experiment data.")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_kin, tab_iso, tab_ind, tab_dl, tab_help = st.tabs(
    ["Kinetics", "Isotherms", "Indicators", "Download Results", "Help"]
)

with tab_kin:
    render_kinetics(EXPERIMENT_DIR)

with tab_iso:
    render_isotherms(EXPERIMENT_DIR)

with tab_ind:
    render_indicators()

with tab_dl:
    render_download()

with tab_help:
    render_help()
