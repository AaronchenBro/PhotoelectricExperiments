#!/usr/bin/env python3
"""Generate report plots for the InGaN green LED PL/EL experiment.

This script uses:
  1. extracted_results.csv for peak-energy, integrated-intensity, and FWHM trends
  2. the original OHSP spectral CSV for normalized PL/EL spectra

Default input/output paths are configured below. You can also override them from
the command line, for example:

    python plot_report_figures.py \
        --results-csv "Week 6 data/new/extracted_results.csv" \
        --spectra-csv "Week 6 data/new/datas.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def running_in_notebook() -> bool:
    """Return True when executed in a Jupyter/IPython notebook kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


IN_NOTEBOOK = running_in_notebook()

try:
    if not IN_NOTEBOOK:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to make plots. Install it with:\n"
        "    python -m pip install matplotlib\n"
    ) from exc


# ---------------------------------------------------------------------------
# Easy-to-edit settings
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Change these input file names if your files are elsewhere.
DEFAULT_RESULTS_CSV = SCRIPT_DIR / "Week 6 data" / "new" / "extracted_results.csv"
DEFAULT_SPECTRA_CSV = SCRIPT_DIR / "Week 6 data" / "new" / "datas.csv"

# Change these lists if the order of spectra in the original CSV changes.
PL_PUMP_POWERS_MW = [41.4, 14.7, 23.5, 18.5, 8.71]
EL_CURRENTS_MA = [90, 100, 80, 60, 70]

# The OHSP spectral block should have one wavelength column plus 10 spectra.
# Left 5 intensity columns = PL, right 5 intensity columns = EL.
N_SPECTRA = 10
PL_COLUMN_INDICES = list(range(0, 5))  # zero-based among the 10 intensity columns
EL_COLUMN_INDICES = list(range(5, 10))

# If automatic spectrum-block detection fails, set this to the 1-based row
# number of the first spectral data row, for example 68 for the current datas.csv.
MANUAL_SPECTRUM_START_ROW = None

# Photon-energy plotting range for normalized spectra. The 2.0-2.7 eV range
# keeps the green LED emission and excludes a possible 405 nm PL laser peak.
PLOT_ENERGY_MIN_EV = 2.0
PLOT_ENERGY_MAX_EV = 2.7

PHOTON_ENERGY_CONSTANT_EV_NM = 1240.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-quality PL/EL plots from extracted results and spectra."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to extracted_results.csv.",
    )
    parser.add_argument(
        "--spectra-csv",
        type=Path,
        default=DEFAULT_SPECTRA_CSV,
        help="Path to the original OHSP spectral CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG/PDF outputs. Defaults to the results CSV folder.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=MANUAL_SPECTRUM_START_ROW,
        help=(
            "1-based row number of the first spectral data row. "
            "Use this only if automatic detection fails."
        ),
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=PLOT_ENERGY_MIN_EV,
        help="Lower photon-energy limit for normalized spectra, in eV.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=PLOT_ENERGY_MAX_EV,
        help="Upper photon-energy limit for normalized spectra, in eV.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Spectral CSV encoding. If omitted, common encodings are tried automatically.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not write all_plots.pdf.",
    )
    return parser.parse_args()


def read_raw_spectral_csv(csv_path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Read an OHSP CSV with metadata rows as a raw string table."""
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp950", "gb18030", "latin1"]
    last_error: Exception | None = None

    for enc in encodings:
        if enc is None:
            continue
        try:
            return pd.read_csv(
                csv_path,
                header=None,
                dtype=str,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not decode {csv_path}. Last error: {last_error}")


def parse_wavelength_nm(value: object) -> float | None:
    """Parse wavelength cells such as '530nm', '530 nm', or '530'."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:nm)?", text, flags=re.I)
    if not match:
        return None
    wavelength_nm = float(match.group(1))
    if not (100.0 <= wavelength_nm <= 2000.0):
        return None
    return wavelength_nm


def locate_spectrum_columns(row: pd.Series, n_spectra: int = N_SPECTRA) -> tuple[int, list[int]] | None:
    """Identify the wavelength column and the following spectral intensity columns."""
    for wavelength_col in range(len(row)):
        if parse_wavelength_nm(row.iloc[wavelength_col]) is None:
            continue

        intensity_cols: list[int] = []
        for col in range(wavelength_col + 1, len(row)):
            value = pd.to_numeric(pd.Series([row.iloc[col]]), errors="coerce").iloc[0]
            if pd.notna(value):
                intensity_cols.append(col)
            if len(intensity_cols) == n_spectra:
                return wavelength_col, intensity_cols

    return None


def row_has_spectrum_data(row: pd.Series, n_spectra: int = N_SPECTRA) -> bool:
    """Check whether one raw CSV row looks like wavelength plus spectral intensities."""
    return locate_spectrum_columns(row, n_spectra=n_spectra) is not None


def wavelength_from_spectrum_row(row: pd.Series, n_spectra: int = N_SPECTRA) -> float | None:
    """Return the wavelength from a row after spectral columns have been detected."""
    located = locate_spectrum_columns(row, n_spectra=n_spectra)
    if located is None:
        return None
    wavelength_col, _ = located
    return parse_wavelength_nm(row.iloc[wavelength_col])


def detect_spectrum_start(raw: pd.DataFrame, n_spectra: int = N_SPECTRA) -> int:
    """Return zero-based row index of the first spectral-data row."""
    first_col = raw.iloc[:, 0].fillna("").astype(str)
    marker_rows = first_col[first_col.str.contains("Spectrum Data", case=False, regex=False)]
    if not marker_rows.empty:
        marker_index = int(marker_rows.index[0])
        for idx in range(marker_index + 1, len(raw)):
            if row_has_spectrum_data(raw.iloc[idx], n_spectra=n_spectra):
                return idx

    # Fallback: find the first sustained sequence of numeric wavelength rows.
    for idx in range(len(raw)):
        if not row_has_spectrum_data(raw.iloc[idx], n_spectra=n_spectra):
            continue
        lookahead_rows = raw.iloc[idx : min(idx + 8, len(raw))]
        wavelengths = [
            wavelength_from_spectrum_row(row, n_spectra=n_spectra)
            for _, row in lookahead_rows.iterrows()
            if row_has_spectrum_data(row, n_spectra=n_spectra)
        ]
        if len(wavelengths) >= 5 and np.all(np.diff(wavelengths[:5]) > 0):
            return idx

    raise ValueError(
        "Could not detect the spectral-data block. Re-run with --start-row set "
        "to the 1-based row number of the first spectral row."
    )


def extract_spectrum_table(
    raw: pd.DataFrame,
    start_row: int | None,
    n_spectra: int = N_SPECTRA,
) -> pd.DataFrame:
    """Extract wavelength and intensity columns from the OHSP spectral block."""
    if start_row is None:
        start_index = detect_spectrum_start(raw, n_spectra=n_spectra)
    else:
        if start_row < 1:
            raise ValueError("--start-row must be a 1-based row number, e.g. 68.")
        start_index = start_row - 1

    records: list[list[float]] = []
    for _, row in raw.iloc[start_index:].iterrows():
        if not row_has_spectrum_data(row, n_spectra=n_spectra):
            if records:
                break
            continue

        located = locate_spectrum_columns(row, n_spectra=n_spectra)
        if located is None:
            continue

        wavelength_col, intensity_cols = located
        wavelength_nm = parse_wavelength_nm(row.iloc[wavelength_col])
        intensities = pd.to_numeric(row.iloc[intensity_cols], errors="coerce").to_numpy(float)
        records.append([float(wavelength_nm), *intensities])

    if not records:
        raise ValueError("No spectral rows were extracted from the original CSV.")

    columns = ["wavelength_nm"] + [f"spectrum_{idx + 1}" for idx in range(n_spectra)]
    table = pd.DataFrame(records, columns=columns)
    table = table.dropna().sort_values("wavelength_nm").drop_duplicates("wavelength_nm")
    return table.reset_index(drop=True)


def spectra_as_energy_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert wavelength to photon energy and sort spectra by increasing energy."""
    wavelength_nm = table["wavelength_nm"].to_numpy(float)
    intensities = table[[f"spectrum_{idx + 1}" for idx in range(N_SPECTRA)]].to_numpy(float)
    energy_eV = PHOTON_ENERGY_CONSTANT_EV_NM / wavelength_nm

    order = np.argsort(energy_eV)
    return energy_eV[order], intensities[order, :]


def normalize_in_window(intensity: np.ndarray) -> np.ndarray:
    """Normalize one spectrum by its own maximum inside the plotted range."""
    max_value = np.nanmax(intensity)
    if max_value <= 0 or not np.isfinite(max_value):
        return np.full_like(intensity, np.nan, dtype=float)
    return intensity / max_value


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.32, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=11, direction="in", top=True, right=True)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_and_optionally_show(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def make_normalized_spectra_plot(
    energy_eV: np.ndarray,
    intensities: np.ndarray,
    column_indices: list[int],
    labels: list[str],
    title: str,
    output_path: Path,
    energy_min: float,
    energy_max: float,
) -> plt.Figure:
    mask = (energy_eV >= energy_min) & (energy_eV <= energy_max)
    if mask.sum() < 2:
        raise ValueError(f"No spectrum points found in {energy_min:.3f}-{energy_max:.3f} eV.")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for col_idx, label in zip(column_indices, labels):
        y = normalize_in_window(intensities[mask, col_idx])
        ax.plot(energy_eV[mask], y, linewidth=2.0, label=label)

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title)
    ax.set_xlim(energy_min, energy_max)
    ax.legend(frameon=False, fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def make_raw_spectra_plot(
    energy_eV: np.ndarray,
    intensities: np.ndarray,
    column_indices: list[int],
    labels: list[str],
    title: str,
    output_path: Path,
    energy_min: float,
    energy_max: float,
) -> plt.Figure:
    mask = (energy_eV >= energy_min) & (energy_eV <= energy_max)
    if mask.sum() < 2:
        raise ValueError(f"No spectrum points found in {energy_min:.3f}-{energy_max:.3f} eV.")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for col_idx, label in zip(column_indices, labels):
        ax.plot(energy_eV[mask], intensities[mask, col_idx], linewidth=2.0, label=label)

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Raw intensity")
    ax.set_title(title)
    ax.set_xlim(energy_min, energy_max)
    ax.legend(frameon=False, fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def require_columns(results: pd.DataFrame) -> None:
    expected = {
        "experiment type",
        "condition value",
        "condition unit",
        "peak photon energy (eV)",
        "integrated intensity",
        "FWHM (meV)",
    }
    missing = expected.difference(results.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"extracted_results.csv is missing required columns: {missing_text}")


def make_trend_plot(
    results: pd.DataFrame,
    experiment_type: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> plt.Figure:
    subset = results[results["experiment type"].astype(str).str.upper() == experiment_type].copy()
    if subset.empty:
        raise ValueError(f"No {experiment_type} rows found in extracted_results.csv.")

    subset["condition value"] = pd.to_numeric(subset["condition value"], errors="coerce")
    subset[y_column] = pd.to_numeric(subset[y_column], errors="coerce")
    subset = subset.dropna(subset=["condition value", y_column]).sort_values("condition value")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(
        subset["condition value"],
        subset[y_column],
        marker="o",
        markersize=6,
        linewidth=2.0,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def close_or_show(figures: list[plt.Figure]) -> None:
    if IN_NOTEBOOK:
        plt.show()
    for fig in figures:
        plt.close(fig)


def main() -> int:
    args = parse_args()

    results_csv = args.results_csv.expanduser().resolve()
    spectra_csv = args.spectra_csv.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else results_csv.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.energy_min >= args.energy_max:
        raise ValueError("--energy-min must be smaller than --energy-max.")

    results = pd.read_csv(results_csv)
    require_columns(results)

    raw_spectra = read_raw_spectral_csv(spectra_csv, encoding=args.encoding)
    spectrum_table = extract_spectrum_table(raw_spectra, start_row=args.start_row)
    energy_eV, intensities = spectra_as_energy_arrays(spectrum_table)

    pl_labels = [f"{value:g} mW" for value in PL_PUMP_POWERS_MW]
    el_labels = [f"{value:g} mA" for value in EL_CURRENTS_MA]
    pl_raw_pairs = sorted(zip(PL_PUMP_POWERS_MW, PL_COLUMN_INDICES))
    pl_raw_labels = [f"{power:g} mW" for power, _ in pl_raw_pairs]
    pl_raw_column_indices = [col_idx for _, col_idx in pl_raw_pairs]

    plot_specs = [
        (
            "PL_normalized_spectra.png",
            lambda path: make_normalized_spectra_plot(
                energy_eV,
                intensities,
                PL_COLUMN_INDICES,
                pl_labels,
                "PL normalized spectra",
                path,
                args.energy_min,
                args.energy_max,
            ),
        ),
        (
            "PL_raw_spectra.png",
            lambda path: make_raw_spectra_plot(
                energy_eV,
                intensities,
                pl_raw_column_indices,
                pl_raw_labels,
                "PL raw spectra",
                path,
                args.energy_min,
                args.energy_max,
            ),
        ),
        (
            "EL_normalized_spectra.png",
            lambda path: make_normalized_spectra_plot(
                energy_eV,
                intensities,
                EL_COLUMN_INDICES,
                el_labels,
                "EL normalized spectra",
                path,
                args.energy_min,
                args.energy_max,
            ),
        ),
        (
            "PL_peak_energy_vs_power.png",
            lambda path: make_trend_plot(
                results,
                "PL",
                "peak photon energy (eV)",
                "Pump power (mW)",
                "Peak photon energy (eV)",
                "PL peak energy vs pump power",
                path,
            ),
        ),
        (
            "EL_peak_energy_vs_current.png",
            lambda path: make_trend_plot(
                results,
                "EL",
                "peak photon energy (eV)",
                "Injection current (mA)",
                "Peak photon energy (eV)",
                "EL peak energy vs injection current",
                path,
            ),
        ),
        (
            "PL_integrated_intensity_vs_power.png",
            lambda path: make_trend_plot(
                results,
                "PL",
                "integrated intensity",
                "Pump power (mW)",
                "Integrated intensity",
                "PL integrated intensity vs pump power",
                path,
            ),
        ),
        (
            "EL_integrated_intensity_vs_current.png",
            lambda path: make_trend_plot(
                results,
                "EL",
                "integrated intensity",
                "Injection current (mA)",
                "Integrated intensity",
                "EL integrated intensity vs injection current",
                path,
            ),
        ),
        (
            "PL_FWHM_vs_power.png",
            lambda path: make_trend_plot(
                results,
                "PL",
                "FWHM (meV)",
                "Pump power (mW)",
                "FWHM (meV)",
                "PL FWHM vs pump power",
                path,
            ),
        ),
        (
            "EL_FWHM_vs_current.png",
            lambda path: make_trend_plot(
                results,
                "EL",
                "FWHM (meV)",
                "Injection current (mA)",
                "FWHM (meV)",
                "EL FWHM vs injection current",
                path,
            ),
        ),
    ]

    saved_files: list[Path] = []
    figures: list[plt.Figure] = []
    for filename, make_plot in plot_specs:
        output_path = output_dir / filename
        figures.append(make_plot(output_path))
        saved_files.append(output_path)

    if not args.no_pdf:
        pdf_path = output_dir / "all_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        saved_files.append(pdf_path)

    close_or_show(figures)

    print("Saved report plots:")
    for path in saved_files:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
