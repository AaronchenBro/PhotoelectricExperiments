#!/usr/bin/env python3
"""Process OHSP spectra for InGaN green LED PL and EL lab analysis.

Default behavior:
  - Reads:  Week 6 data/new/datas.csv
  - Writes: all result CSV/XLSX/PNG files beside datas.csv

The CSV exported by the OHSP spectrometer contains an upper metadata block and
a lower spectrum block. This script detects the spectrum block automatically.
If detection fails, pass --start-row with the 1-based row number of the first
spectral row, for example:

    python process_spectra.py --start-row 68
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to make plots. Install it with:\n"
        "    python -m pip install matplotlib\n"
    ) from exc


# ---------------------------------------------------------------------------
# Easy-to-edit experiment settings
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR / "Week 6 data" / "new" / "datas.csv"

# If automatic spectrum-block detection fails, set this to the 1-based row
# number of the first spectral data row, e.g. 68. Leave as None for auto-detect.
MANUAL_SPECTRUM_START_ROW = None

# The OHSP file is expected to have 1 wavelength column followed by 10 spectra.
# Columns 1-5 after wavelength are PL, columns 6-10 after wavelength are EL.
N_SPECTRA = 10
PL_COLUMN_INDICES = list(range(0, 5))  # zero-based among the 10 intensity cols
EL_COLUMN_INDICES = list(range(5, 10))

PL_PUMP_POWERS_MW = [41.4, 14.7, 23.5, 18.5, 8.71]
EL_CURRENTS_MA = [90, 100, 80, 60, 70]

# Green InGaN LED emission window. This excludes the possible 405 nm laser peak
# near 3.06 eV from PL peak finding, integration, FWHM, and normalized plots.
DEFAULT_ENERGY_MIN_EV = 2.0
DEFAULT_ENERGY_MAX_EV = 2.7

PHOTON_ENERGY_CONSTANT_EV_NM = 1240.0


@dataclass(frozen=True)
class SpectrumMetric:
    experiment_type: str
    condition_value: float
    condition_unit: str
    peak_wavelength_nm: float
    peak_photon_energy_eV: float
    integrated_intensity: float
    fwhm_eV: float
    fwhm_meV: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process OHSP PL/EL spectra and generate lab-report plots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the OHSP CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted_results and PNG files. Defaults to CSV folder.",
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
        default=DEFAULT_ENERGY_MIN_EV,
        help="Lower photon-energy bound for green LED analysis, in eV.",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=DEFAULT_ENERGY_MAX_EV,
        help="Upper photon-energy bound for green LED analysis, in eV.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="CSV encoding. If omitted, common encodings are tried automatically.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip writing extracted_results.xlsx.",
    )
    return parser.parse_args()


def read_raw_csv(csv_path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Read the irregular OHSP CSV as strings, preserving metadata rows."""
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
    """Parse a wavelength cell such as '530nm', '530 nm', or '530'."""
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


def row_has_spectrum_data(row: pd.Series, n_spectra: int = N_SPECTRA) -> bool:
    wavelength_nm = parse_wavelength_nm(row.iloc[0] if len(row) else None)
    if wavelength_nm is None or len(row) < n_spectra + 1:
        return False

    intensities = pd.to_numeric(row.iloc[1 : n_spectra + 1], errors="coerce")
    return bool(intensities.notna().sum() == n_spectra)


def detect_spectrum_start(raw: pd.DataFrame, n_spectra: int = N_SPECTRA) -> int:
    """Return the zero-based DataFrame row index where spectral data starts."""
    first_col = raw.iloc[:, 0].fillna("").astype(str)
    marker_rows = first_col[first_col.str.contains("Spectrum Data", case=False, regex=False)]
    if not marker_rows.empty:
        marker_index = int(marker_rows.index[0])
        for idx in range(marker_index + 1, len(raw)):
            if row_has_spectrum_data(raw.iloc[idx], n_spectra=n_spectra):
                return idx

    # Fallback: find the first row starting a sustained run of spectrum-like rows.
    for idx in range(len(raw)):
        if not row_has_spectrum_data(raw.iloc[idx], n_spectra=n_spectra):
            continue
        lookahead = raw.iloc[idx : min(idx + 8, len(raw))]
        candidate_waves = [
            parse_wavelength_nm(row.iloc[0])
            for _, row in lookahead.iterrows()
            if row_has_spectrum_data(row, n_spectra=n_spectra)
        ]
        if len(candidate_waves) >= 5 and np.all(np.diff(candidate_waves[:5]) > 0):
            return idx

    raise ValueError(
        "Could not find the spectrum block. Re-run with --start-row set to the "
        "1-based row number of the first spectral data row."
    )


def extract_spectrum_table(
    raw: pd.DataFrame,
    start_row: int | None,
    n_spectra: int = N_SPECTRA,
) -> pd.DataFrame:
    """Extract wavelength and intensity columns from the raw OHSP table."""
    if start_row is not None:
        if start_row < 1:
            raise ValueError("--start-row must be a 1-based row number, e.g. 68.")
        start_index = start_row - 1
    else:
        start_index = detect_spectrum_start(raw, n_spectra=n_spectra)

    records: list[list[float]] = []
    for _, row in raw.iloc[start_index:].iterrows():
        if not row_has_spectrum_data(row, n_spectra=n_spectra):
            if records:
                break
            continue

        wavelength_nm = parse_wavelength_nm(row.iloc[0])
        intensities = pd.to_numeric(row.iloc[1 : n_spectra + 1], errors="coerce").to_numpy(float)
        records.append([float(wavelength_nm), *intensities])

    if not records:
        raise ValueError("No numeric spectrum rows were extracted from the CSV.")

    columns = ["wavelength_nm"] + [f"spectrum_{idx + 1}" for idx in range(n_spectra)]
    spectrum = pd.DataFrame(records, columns=columns)
    spectrum = spectrum.dropna().sort_values("wavelength_nm").drop_duplicates("wavelength_nm")

    if spectrum.shape[1] != n_spectra + 1:
        raise ValueError(f"Expected 1 wavelength column plus {n_spectra} spectra.")
    return spectrum.reset_index(drop=True)


def prepare_energy_arrays(
    spectrum: pd.DataFrame,
    n_spectra: int = N_SPECTRA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wavelength, photon energy, and intensities sorted by increasing energy."""
    wavelength_nm = spectrum["wavelength_nm"].to_numpy(float)
    intensity_matrix = spectrum[[f"spectrum_{idx + 1}" for idx in range(n_spectra)]].to_numpy(float)
    energy_eV = PHOTON_ENERGY_CONSTANT_EV_NM / wavelength_nm

    order = np.argsort(energy_eV)
    return wavelength_nm[order], energy_eV[order], intensity_matrix[order, :]


def interpolate_crossing(x0: float, y0: float, x1: float, y1: float, y_target: float) -> float:
    """Linear interpolation for the x value where y crosses y_target."""
    if np.isclose(y1, y0):
        return 0.5 * (x0 + x1)
    fraction = (y_target - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0)


def fwhm_from_spectrum(energy_eV: np.ndarray, intensity: np.ndarray) -> float:
    """Compute FWHM in eV using interpolated half-maximum crossings."""
    if len(energy_eV) < 3 or np.nanmax(intensity) <= 0:
        return np.nan

    peak_index = int(np.nanargmax(intensity))
    half_max = 0.5 * float(intensity[peak_index])

    left_crossing = np.nan
    for idx in range(peak_index - 1, -1, -1):
        y0 = float(intensity[idx])
        y1 = float(intensity[idx + 1])
        if (y0 <= half_max <= y1) or (y1 <= half_max <= y0):
            left_crossing = interpolate_crossing(
                float(energy_eV[idx]), y0, float(energy_eV[idx + 1]), y1, half_max
            )
            break

    right_crossing = np.nan
    for idx in range(peak_index, len(energy_eV) - 1):
        y0 = float(intensity[idx])
        y1 = float(intensity[idx + 1])
        if (y0 >= half_max >= y1) or (y1 >= half_max >= y0):
            right_crossing = interpolate_crossing(
                float(energy_eV[idx]), y0, float(energy_eV[idx + 1]), y1, half_max
            )
            break

    if np.isnan(left_crossing) or np.isnan(right_crossing):
        return np.nan
    return abs(right_crossing - left_crossing)


def analyze_one_spectrum(
    energy_eV: np.ndarray,
    intensity: np.ndarray,
    experiment_type: str,
    condition_value: float,
    condition_unit: str,
    energy_min: float,
    energy_max: float,
) -> SpectrumMetric:
    """Extract peak position, integrated intensity, and FWHM in the green window."""
    mask = (energy_eV >= energy_min) & (energy_eV <= energy_max)
    if mask.sum() < 3:
        raise ValueError(
            f"Too few data points in analysis range {energy_min:.3f}-{energy_max:.3f} eV."
        )

    e_window = energy_eV[mask]
    i_window = intensity[mask]
    peak_idx = int(np.nanargmax(i_window))
    peak_energy_eV = float(e_window[peak_idx])
    peak_wavelength_nm = PHOTON_ENERGY_CONSTANT_EV_NM / peak_energy_eV
    integrated_intensity = float(np.trapezoid(i_window, e_window))
    fwhm_eV = float(fwhm_from_spectrum(e_window, i_window))

    return SpectrumMetric(
        experiment_type=experiment_type,
        condition_value=float(condition_value),
        condition_unit=condition_unit,
        peak_wavelength_nm=float(peak_wavelength_nm),
        peak_photon_energy_eV=peak_energy_eV,
        integrated_intensity=integrated_intensity,
        fwhm_eV=fwhm_eV,
        fwhm_meV=fwhm_eV * 1000.0 if np.isfinite(fwhm_eV) else np.nan,
    )


def normalize(intensity: np.ndarray) -> np.ndarray:
    maximum = np.nanmax(intensity)
    if maximum <= 0 or not np.isfinite(maximum):
        return np.full_like(intensity, np.nan, dtype=float)
    return intensity / maximum


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_normalized_spectra_plot(
    energy_eV: np.ndarray,
    intensity_matrix: np.ndarray,
    column_indices: list[int],
    labels: list[str],
    title: str,
    output_path: Path,
    energy_min: float,
    energy_max: float,
) -> None:
    mask = (energy_eV >= energy_min) & (energy_eV <= energy_max)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for col_idx, label in zip(column_indices, labels):
        ax.plot(
            energy_eV[mask],
            normalize(intensity_matrix[mask, col_idx]),
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_trend_plot(
    results: pd.DataFrame,
    experiment_type: str,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    subset = results[results["experiment type"] == experiment_type].copy()
    subset = subset.sort_values(x_column)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(subset[x_column], subset[y_column], marker="o", linewidth=1.8, markersize=5.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_results_dataframe(metrics: list[SpectrumMetric]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment type": item.experiment_type,
                "condition value": item.condition_value,
                "condition unit": item.condition_unit,
                "peak wavelength (nm)": item.peak_wavelength_nm,
                "peak photon energy (eV)": item.peak_photon_energy_eV,
                "integrated intensity": item.integrated_intensity,
                "FWHM (eV)": item.fwhm_eV,
                "FWHM (meV)": item.fwhm_meV,
            }
            for item in metrics
        ]
    )


def main() -> int:
    args = parse_args()
    csv_path = args.input.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else csv_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.energy_min >= args.energy_max:
        raise ValueError("--energy-min must be smaller than --energy-max.")

    raw = read_raw_csv(csv_path, encoding=args.encoding)
    spectrum = extract_spectrum_table(raw, start_row=args.start_row, n_spectra=N_SPECTRA)
    wavelength_nm, energy_eV, intensity_matrix = prepare_energy_arrays(spectrum, n_spectra=N_SPECTRA)

    metrics: list[SpectrumMetric] = []
    for col_idx, pump_mw in zip(PL_COLUMN_INDICES, PL_PUMP_POWERS_MW):
        metrics.append(
            analyze_one_spectrum(
                energy_eV,
                intensity_matrix[:, col_idx],
                experiment_type="PL",
                condition_value=pump_mw,
                condition_unit="mW",
                energy_min=args.energy_min,
                energy_max=args.energy_max,
            )
        )

    for col_idx, current_ma in zip(EL_COLUMN_INDICES, EL_CURRENTS_MA):
        metrics.append(
            analyze_one_spectrum(
                energy_eV,
                intensity_matrix[:, col_idx],
                experiment_type="EL",
                condition_value=current_ma,
                condition_unit="mA",
                energy_min=args.energy_min,
                energy_max=args.energy_max,
            )
        )

    results = build_results_dataframe(metrics)

    csv_output = output_dir / "extracted_results.csv"
    results.to_csv(csv_output, index=False)

    if not args.no_excel:
        xlsx_output = output_dir / "extracted_results.xlsx"
        try:
            results.to_excel(xlsx_output, index=False)
        except ModuleNotFoundError:
            print("openpyxl is not installed; skipped extracted_results.xlsx", file=sys.stderr)

    pl_labels = [f"{value:g} mW" for value in PL_PUMP_POWERS_MW]
    el_labels = [f"{value:g} mA" for value in EL_CURRENTS_MA]

    save_normalized_spectra_plot(
        energy_eV,
        intensity_matrix,
        PL_COLUMN_INDICES,
        pl_labels,
        "PL normalized spectra",
        output_dir / "PL_normalized_spectra.png",
        args.energy_min,
        args.energy_max,
    )
    save_normalized_spectra_plot(
        energy_eV,
        intensity_matrix,
        EL_COLUMN_INDICES,
        el_labels,
        "EL normalized spectra",
        output_dir / "EL_normalized_spectra.png",
        args.energy_min,
        args.energy_max,
    )

    save_trend_plot(
        results,
        "PL",
        "condition value",
        "peak photon energy (eV)",
        "Pump power (mW)",
        "Peak photon energy (eV)",
        "PL peak energy vs pump power",
        output_dir / "PL_peak_energy_vs_power.png",
    )
    save_trend_plot(
        results,
        "EL",
        "condition value",
        "peak photon energy (eV)",
        "Injection current (mA)",
        "Peak photon energy (eV)",
        "EL peak energy vs injection current",
        output_dir / "EL_peak_energy_vs_current.png",
    )
    save_trend_plot(
        results,
        "PL",
        "condition value",
        "integrated intensity",
        "Pump power (mW)",
        "Integrated spectral intensity (a.u. eV)",
        "PL integrated intensity vs pump power",
        output_dir / "PL_integrated_intensity_vs_power.png",
    )
    save_trend_plot(
        results,
        "EL",
        "condition value",
        "integrated intensity",
        "Injection current (mA)",
        "Integrated spectral intensity (a.u. eV)",
        "EL integrated intensity vs injection current",
        output_dir / "EL_integrated_intensity_vs_current.png",
    )
    save_trend_plot(
        results,
        "PL",
        "condition value",
        "FWHM (meV)",
        "Pump power (mW)",
        "FWHM (meV)",
        "PL FWHM vs pump power",
        output_dir / "PL_FWHM_vs_power.png",
    )
    save_trend_plot(
        results,
        "EL",
        "condition value",
        "FWHM (meV)",
        "Injection current (mA)",
        "FWHM (meV)",
        "EL FWHM vs injection current",
        output_dir / "EL_FWHM_vs_current.png",
    )

    print(f"Read spectrum rows: {len(spectrum)}")
    print(f"Photon energy analysis range: {args.energy_min:.3f}-{args.energy_max:.3f} eV")
    print(f"Saved outputs to: {output_dir}")
    print()
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
