#!/usr/bin/env python3
"""Process OHSP summary rows for the temperature-dependent InGaN LED experiment.

Important report note:
    ourTemp.csv contains only OHSP summary data, not full spectral curves.
    Therefore this script analyzes temperature dependence of peak wavelength,
    peak photon energy, FWHM, illuminance, irradiance, and peak signal, but it
    does not generate temperature-dependent raw or normalized spectra.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
# Easy-to-edit settings
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Change this input file name if the summary CSV is elsewhere.
DEFAULT_INPUT_CSV = SCRIPT_DIR / "Week 6 data" / "ourTemp.csv"

# ourTemp.csv is a Chinese OHSP export; gbk is required for readable headers.
CSV_ENCODING = "gbk"

# Labels in original row order.
TEMPERATURE_LABELS = ["RM", "73°C", "41°C", "57°C", "86°C"]

# Numerical temperatures used for trend plots. RM is treated as 25 °C.
NUMERICAL_TEMPERATURES_C = [25, 73, 41, 57, 86]

PHOTON_ENERGY_CONSTANT_EV_NM = 1240.0

# Required OHSP summary columns.
COL_ILLUMINANCE = "照度E(lx)"
COL_IRRADIANCE = "辐照度Ee(mW/cm2)"
COL_PEAK_WAVELENGTH = "峰值波长(nm)"
COL_CENTER_WAVELENGTH = "中心波长(nm)"
COL_CENTROID_WAVELENGTH = "质心波长(nm)"
COL_FWHM_NM = "半宽度(nm)"
COL_PEAK_SIGNAL = "峰值信号"
COL_INTEGRATION_TIME = "积分时间(ms)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process temperature-dependent OHSP summary rows from ourTemp.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to ourTemp.csv or another OHSP summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for temperature_summary_results.csv and plots. Defaults to input folder.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame) -> None:
    required = [
        COL_ILLUMINANCE,
        COL_IRRADIANCE,
        COL_PEAK_WAVELENGTH,
        COL_CENTER_WAVELENGTH,
        COL_CENTROID_WAVELENGTH,
        COL_FWHM_NM,
        COL_PEAK_SIGNAL,
        COL_INTEGRATION_TIME,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_text}")


def read_summary_csv(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding=CSV_ENCODING)
    require_columns(df)

    if len(df) != len(TEMPERATURE_LABELS):
        raise ValueError(
            f"Expected {len(TEMPERATURE_LABELS)} summary rows, but found {len(df)} rows."
        )
    return df


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        bad_rows = (values[values.isna()].index + 1).tolist()
        raise ValueError(f"Column {column!r} has non-numeric value(s) in row(s): {bad_rows}")
    return values


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    peak_wavelength_nm = numeric_column(df, COL_PEAK_WAVELENGTH)
    fwhm_nm = numeric_column(df, COL_FWHM_NM)

    peak_energy_eV = PHOTON_ENERGY_CONSTANT_EV_NM / peak_wavelength_nm
    fwhm_eV = PHOTON_ENERGY_CONSTANT_EV_NM * fwhm_nm / (peak_wavelength_nm**2)
    fwhm_meV = 1000.0 * fwhm_eV

    return pd.DataFrame(
        {
            "label": TEMPERATURE_LABELS,
            "numerical_temperature_C": NUMERICAL_TEMPERATURES_C,
            "illuminance_lx": numeric_column(df, COL_ILLUMINANCE),
            "irradiance_mW_cm2": numeric_column(df, COL_IRRADIANCE),
            "peak_wavelength_nm": peak_wavelength_nm,
            "peak_energy_eV": peak_energy_eV,
            "center_wavelength_nm": numeric_column(df, COL_CENTER_WAVELENGTH),
            "centroid_wavelength_nm": numeric_column(df, COL_CENTROID_WAVELENGTH),
            "FWHM_nm": fwhm_nm,
            "FWHM_meV": fwhm_meV,
            "peak_signal": numeric_column(df, COL_PEAK_SIGNAL),
            "integration_time_ms": numeric_column(df, COL_INTEGRATION_TIME),
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.32, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=11, direction="in", top=True, right=True)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_trend_plot(
    sorted_df: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(
        sorted_df["numerical_temperature_C"],
        sorted_df[y_column],
        marker="o",
        markersize=6,
        linewidth=2.0,
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_csv = args.input.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else input_csv.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_summary_csv(input_csv)
    summary_df = build_summary_table(df)
    sorted_df = summary_df.sort_values("numerical_temperature_C").reset_index(drop=True)

    summary_path = output_dir / "temperature_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_specs = [
        (
            "Temp_illuminance_vs_temperature.png",
            "illuminance_lx",
            "Illuminance (lx)",
            "Illuminance vs temperature",
        ),
        (
            "Temp_irradiance_vs_temperature.png",
            "irradiance_mW_cm2",
            "Irradiance (mW/cm²)",
            "Irradiance vs temperature",
        ),
        (
            "Temp_peak_energy_vs_temperature.png",
            "peak_energy_eV",
            "Peak photon energy (eV)",
            "Peak photon energy vs temperature",
        ),
        (
            "Temp_peak_wavelength_vs_temperature.png",
            "peak_wavelength_nm",
            "Peak wavelength (nm)",
            "Peak wavelength vs temperature",
        ),
        (
            "Temp_FWHM_vs_temperature.png",
            "FWHM_meV",
            "FWHM (meV)",
            "FWHM vs temperature",
        ),
        (
            "Temp_peak_signal_vs_temperature.png",
            "peak_signal",
            "Peak signal",
            "Peak signal vs temperature",
        ),
    ]

    saved_files = [summary_path]
    for filename, y_column, y_label, title in plot_specs:
        output_path = output_dir / filename
        save_trend_plot(sorted_df, y_column, y_label, title, output_path)
        saved_files.append(output_path)

    print("Original row-order table:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print("Sorted-by-temperature table:")
    print(sorted_df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print(
        "Report note: ourTemp.csv contains only summary data, so this script "
        "does not generate temperature-dependent raw or normalized spectra."
    )
    print()
    print("Saved files:")
    for path in saved_files:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
