#!/usr/bin/env python3
"""Extract TFT parameters from transfer-curve CSV files.

This script calculates three quantities directly from transfer-curve
coordinate data:

1. Threshold voltage (Vth)
2. Subthreshold swing (SS)
3. On/off current ratio (Ion/Ioff)

The calculation follows the usual TFT workflow:

- Linear region: fit ID-VGS near the turn-on segment and extrapolate to ID = 0
- Saturation region: fit sqrt(ID)-VGS near the turn-on segment and extrapolate
  to sqrt(ID) = 0
- SS: fit log10(ID)-VGS in the steepest subthreshold segment
- Ion/Ioff: max(ID) / min(ID) from the same transfer curve

The input CSV files are expected to live under ./data and contain these
columns:

- 栅极电压(V)
- 漏极电压(V)
- 漏极电流(A)

Example:
    python3 extract_tft_params.py
    python3 extract_tft_params.py --match no_light --output no_light_parameters.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT = ROOT_DIR / "transfer_parameters.csv"

VGS_COLUMN = "栅极电压(V)"
VDS_COLUMN = "漏极电压(V)"
ID_COLUMN = "漏极电流(A)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Vth, SS, and Ion/Ioff from transfer-curve CSV data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory that contains transfer CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file used to save the calculated parameters.",
    )
    parser.add_argument(
        "--match",
        default="",
        help="Only analyze files whose names contain this substring.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of neighboring points used in each fitting window.",
    )
    return parser.parse_args()


def discover_transfer_files(data_dir: Path, match: str) -> list[Path]:
    files = []
    for path in sorted(data_dir.glob("*.csv")):
        name = path.name.lower()
        if "linear" not in name and "sat" not in name:
            continue
        if match and match.lower() not in name:
            continue
        files.append(path)
    return files


def load_transfer_curve(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {VGS_COLUMN, VDS_COLUMN, ID_COLUMN} - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path.name} is missing required columns: {missing_text}")

        for row in reader:
            vgs = float(row[VGS_COLUMN])
            vds = float(row[VDS_COLUMN])
            id_abs = abs(float(row[ID_COLUMN]))
            rows.append((vgs, id_abs, vds))

    if len(rows) < 2:
        raise ValueError(f"{path.name} does not contain enough data points.")

    rows.sort(key=lambda item: item[0])
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    x_mean = mean(xs)
    y_mean = mean(ys)

    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("All x values in the fit window are identical.")

    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    fitted = [slope * x + intercept for x in xs]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, fitted))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return slope, intercept, r_squared


def infer_turn_on_direction(curve: list[tuple[float, float, float]]) -> int:
    ids = [item[1] for item in curve]
    ion_index = max(range(len(ids)), key=ids.__getitem__)
    ioff_index = min(range(len(ids)), key=ids.__getitem__)

    return 1 if curve[ion_index][0] >= curve[ioff_index][0] else -1


def select_best_window(
    curve: list[tuple[float, float, float]],
    transform: Callable[[float], float],
    window_size: int,
    required_sign: int,
) -> dict[str, float]:
    best: dict[str, float] | None = None

    for start in range(0, len(curve) - window_size + 1):
        segment = curve[start : start + window_size]
        xs = [point[0] for point in segment]

        try:
            ys = [transform(point[1]) for point in segment]
            slope, intercept, r_squared = fit_line(xs, ys)
        except ValueError:
            continue

        if slope == 0:
            continue

        sign = 1 if slope > 0 else -1
        if sign != required_sign:
            continue

        score = abs(slope) * max(r_squared, 0.0)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "vgs_start": xs[0],
                "vgs_end": xs[-1],
            }

    if best is None:
        raise ValueError("No valid fit window was found for this curve.")

    return best


def classify_regime(path: Path, vds_values: list[float]) -> str:
    name = path.name.lower()
    if "linear" in name:
        return "linear"
    if "sat" in name:
        return "saturation"
    return "saturation" if abs(mean(vds_values)) >= 1.0 else "linear"


def calc_vth(
    curve: list[tuple[float, float, float]], regime: str, direction: int, window_size: int
) -> dict[str, float]:
    if regime == "saturation":
        transform = math.sqrt
        method = "sqrt(ID)-VGS extrapolation"
    else:
        transform = lambda current: current
        method = "ID-VGS extrapolation"

    fit = select_best_window(
        curve=curve,
        transform=transform,
        window_size=window_size,
        required_sign=direction,
    )
    vth = -fit["intercept"] / fit["slope"]

    return {
        "vth_v": vth,
        "method": method,
        "fit_start_vgs": fit["vgs_start"],
        "fit_end_vgs": fit["vgs_end"],
        "fit_r_squared": fit["r_squared"],
    }


def calc_ss(
    curve: list[tuple[float, float, float]], direction: int, window_size: int
) -> dict[str, float]:
    fit = select_best_window(
        curve=curve,
        transform=math.log10,
        window_size=window_size,
        required_sign=direction,
    )
    ss_mv_per_dec = 1000.0 / abs(fit["slope"])

    return {
        "ss_mv_dec": ss_mv_per_dec,
        "fit_start_vgs": fit["vgs_start"],
        "fit_end_vgs": fit["vgs_end"],
        "fit_r_squared": fit["r_squared"],
    }


def calc_on_off(curve: list[tuple[float, float, float]]) -> dict[str, float]:
    ids = [point[1] for point in curve]
    ion = max(ids)
    ioff = min(ids)

    if ioff <= 0:
        raise ValueError("Ioff must be positive to calculate Ion/Ioff.")

    return {
        "ion_a": ion,
        "ioff_a": ioff,
        "on_off_ratio": ion / ioff,
    }


def analyze_file(path: Path, window_size: int) -> dict[str, float | str]:
    curve = load_transfer_curve(path)
    direction = infer_turn_on_direction(curve)
    vds_values = [point[2] for point in curve]
    regime = classify_regime(path, vds_values)

    vth = calc_vth(curve, regime, direction, window_size)
    ss = calc_ss(curve, direction, window_size)
    on_off = calc_on_off(curve)

    return {
        "file": path.name,
        "regime": regime,
        "vds_mean_v": mean(vds_values),
        "turn_on_direction": "increasing VGS" if direction > 0 else "decreasing VGS",
        "vth_method": vth["method"],
        "vth_v": vth["vth_v"],
        "vth_fit_start_vgs": vth["fit_start_vgs"],
        "vth_fit_end_vgs": vth["fit_end_vgs"],
        "vth_fit_r_squared": vth["fit_r_squared"],
        "ss_mv_dec": ss["ss_mv_dec"],
        "ss_fit_start_vgs": ss["fit_start_vgs"],
        "ss_fit_end_vgs": ss["fit_end_vgs"],
        "ss_fit_r_squared": ss["fit_r_squared"],
        "ion_a": on_off["ion_a"],
        "ioff_a": on_off["ioff_a"],
        "ion_ioff": on_off["on_off_ratio"],
    }


def write_results(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("No result rows were provided.")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000 or abs(value) < 1e-3:
        return f"{value:.4e}"
    return f"{value:.4f}"


def print_summary(rows: list[dict[str, float | str]]) -> None:
    print("\nCalculated TFT parameters from transfer curves:\n")
    header = (
        f"{'file':<24} {'regime':<11} {'Vth (V)':>12} "
        f"{'SS (mV/dec)':>14} {'Ion/Ioff':>14}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{str(row['file']):<24} "
            f"{str(row['regime']):<11} "
            f"{format_float(float(row['vth_v'])):>12} "
            f"{format_float(float(row['ss_mv_dec'])):>14} "
            f"{format_float(float(row['ion_ioff'])):>14}"
        )


def main() -> None:
    args = parse_args()

    if args.window < 3:
        raise SystemExit("--window must be at least 3.")

    files = discover_transfer_files(args.data_dir, args.match)
    if not files:
        raise SystemExit("No transfer-curve CSV files matched the current filters.")

    results = [analyze_file(path, args.window) for path in files]
    write_results(args.output, results)
    print_summary(results)
    print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()
