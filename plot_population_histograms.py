#!/usr/bin/env python3
"""Plot the catalog-level histogram files produced by sky_map.c.

The sky_map catalog mode writes four histogram files in the skymaps directory:

    area50_histogram.dat
    area90_histogram.dat
    distance90_interval_histogram.dat
    snr_histogram.dat

Each file contains header metadata followed by columns

    bin_low bin_high bin_center count fraction density

This script uses the per-bin fractions for the bar heights, and marks the
sample median from the file header in each panel.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

# Keep Matplotlib from trying to write under the user's home directory on
# systems where that cache location is not writable from this run context.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sky_map_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


HISTOGRAMS = (
    {
        "filename": "area50_histogram.dat",
        "title": "50% credible area",
        "xlabel": r"Sky area (deg$^2$)",
        "color": "#d97835",
    },
    {
        "filename": "area90_histogram.dat",
        "title": "90% credible area",
        "xlabel": r"Sky area (deg$^2$)",
        "color": "#b23a48",
    },
    {
        "filename": "distance90_interval_histogram.dat",
        "title": "90% luminosity-distance interval",
        "xlabel": r"90% interval width in $D_L$ (Mpc)",
        "color": "#2f7f7b",
    },
    {
        "filename": "snr_histogram.dat",
        "title": "Detected network SNR",
        "xlabel": "Network SNR",
        "color": "#3f5aa8",
    },
)


def parse_header(path: Path) -> dict[str, object]:
    """Read key metadata from comment lines in a sky_map histogram file."""
    meta: dict[str, object] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("#"):
                break

            fields = raw_line[1:].strip().split()
            if not fields:
                continue

            key = fields[0]
            values = fields[1:]

            if key == "quantity" and values:
                meta["quantity"] = values[0]
            elif key == "units" and values:
                meta["units"] = " ".join(values)
            elif key == "samples" and len(values) >= 3 and values[1] == "finite_samples":
                meta["samples"] = int(values[0])
                meta["finite_samples"] = int(values[2])
            elif key == "samples" and len(values) >= 2:
                meta["samples"] = int(values[0])
                meta["finite_samples"] = int(values[1])
            elif key == "bins" and values:
                meta["bins"] = int(values[0])
            elif key == "histogram_range" and len(values) >= 2:
                meta["histogram_range"] = (float(values[0]), float(values[1]))
            elif key == "sample_min_max_mean" and len(values) >= 3:
                meta["sample_min"] = float(values[0])
                meta["sample_max"] = float(values[1])
                meta["sample_mean"] = float(values[2])
            elif key == "sample_median" and values:
                meta["sample_median"] = float(values[0])
            elif key == "underflow_overflow" and len(values) >= 2:
                meta["underflow"] = int(values[0])
                meta["overflow"] = int(values[1])

    return meta


def read_histogram(path: Path) -> tuple[dict[str, object], np.ndarray]:
    """Load a histogram file and return metadata plus a 2D data array."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing histogram file: {path}. If this is distance90_interval_histogram.dat, "
            "rerun sky_map so the new luminosity-distance posterior intervals are written."
        )

    meta = parse_header(path)
    data = np.loadtxt(path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 6:
        raise ValueError(f"Histogram file {path} has {data.shape[1]} columns; expected at least 6")

    return meta, data


def compact_number(x: float) -> str:
    """Format diagnostic values compactly for panel annotations."""
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax >= 1000.0:
        return f"{x:.0f}"
    if ax >= 100.0:
        return f"{x:.1f}"
    if ax >= 10.0:
        return f"{x:.2f}"
    return f"{x:.3f}"


def median_from_histogram(data: np.ndarray) -> float | None:
    """Estimate the median from histogram counts when no exact header median exists."""
    counts = data[:, 3]
    total = np.sum(counts)
    if total <= 0.0:
        return None

    target = 0.5 * total
    cumulative = 0.0

    for row in data:
        low, high, _center, count = row[:4]
        if count <= 0.0:
            continue
        if cumulative + count >= target:
            frac = (target - cumulative) / count
            frac = min(max(frac, 0.0), 1.0)
            return low + frac * (high - low)
        cumulative += count

    return float(data[-1, 2])


def apply_plot_style() -> None:
    """Set a clean scientific plotting style without external dependencies."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfaf6",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#2f2f2f",
            "axes.labelcolor": "#242424",
            "axes.titlecolor": "#181818",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#dfd9cd",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.facecolor": "#fbfaf6",
            "savefig.bbox": "tight",
        }
    )


def plot_histogram_panel(ax: plt.Axes, path: Path, spec: dict[str, str], show_density: bool) -> None:
    """Draw one histogram panel."""
    meta, data = read_histogram(path)

    lows = data[:, 0]
    highs = data[:, 1]
    centers = data[:, 2]
    heights = data[:, 5] if show_density else data[:, 4]
    widths = highs - lows

    ax.bar(
        centers,
        heights,
        width=widths,
        align="center",
        color=spec["color"],
        edgecolor="#1f1f1f",
        linewidth=0.45,
        alpha=0.88,
    )

    median = meta.get("sample_median")
    if not isinstance(median, float) or not np.isfinite(median):
        median = median_from_histogram(data)
    if isinstance(median, float) and np.isfinite(median):
        ax.axvline(median, color="#151515", linewidth=1.25, linestyle="--", alpha=0.9)

    ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("Probability density" if show_density else "Fraction of detected sources")
    ax.set_xlim(lows[0], highs[-1])
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    samples = meta.get("finite_samples", meta.get("samples", None))
    text_lines = []
    if samples is not None:
        text_lines.append(f"N = {samples}")
    if isinstance(median, float) and np.isfinite(median):
        text_lines.append(f"median = {compact_number(median)}")
    if text_lines:
        ax.text(
            0.97,
            0.95,
            "\n".join(text_lines),
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "#fbfaf6",
                "edgecolor": "#c8bda8",
                "alpha": 0.92,
            },
        )


def make_plot(hist_dir: Path, output: Path, formats: list[str], dpi: int, show_density: bool) -> list[Path]:
    """Create the four-panel population histogram plot."""
    apply_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)

    for ax, spec in zip(axes.flat, HISTOGRAMS):
        plot_histogram_panel(ax, hist_dir / spec["filename"], spec, show_density)

    fig.suptitle("Detected BNS Catalog Summary", fontsize=18, fontweight="bold", y=1.02)

    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        outpath = output.with_suffix(f".{fmt}")
        fig.savefig(outpath, dpi=dpi)
        written.append(outpath)

    plt.close(fig)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the four population histogram files produced by sky_map catalog mode."
    )
    parser.add_argument(
        "--hist-dir",
        default="skymaps",
        type=Path,
        help="Directory containing the *_histogram.dat files. Default: skymaps",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output path stem. Default: <hist-dir>/population_histograms",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output formats to write. Default: png pdf",
    )
    parser.add_argument(
        "--dpi",
        default=220,
        type=int,
        help="Raster DPI for PNG output. Default: 220",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot probability density instead of per-bin fraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hist_dir = args.hist_dir
    output = args.output if args.output is not None else hist_dir / "population_histograms"

    written = make_plot(hist_dir, output, args.formats, args.dpi, args.density)

    for path in written:
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
