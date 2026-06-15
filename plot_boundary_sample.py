#!/usr/bin/env python
"""Combine and plot 90% sky-localization boundary maps for a source sample.

Catalog mode in sky_map.c writes one boundary-pixel map per source:

    skymaps/sky90_boundary_1.dat
    skymaps/sky90_boundary_2.dat
    ...

Each file is a RING-ordered HEALPix map with value 1 on the 90% credible-region
boundary and 0 elsewhere. This script selects N of those maps and renders the
union of their boundaries: pixels touched by one or more selected contours are
set to 1, and all other pixels are set to 0.

By default, sources with 90% credible areas larger than 50 square degrees are
excluded before selecting the sample. The area is read from summary_#.dat.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

# Keep Matplotlib/Healpy from trying to write cache files in an unwritable home
# directory when the script is launched from a restricted environment.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sky_map_matplotlib"))

try:
    import healpy as hp
except ImportError as exc:  # pragma: no cover - exercised only on missing deps.
    raise SystemExit(
        "plot_boundary_sample.py requires healpy. In this workspace, try running "
        "`python plot_boundary_sample.py ...` rather than `python3 ...`, since the "
        "project plotting scripts use the Python environment with healpy installed."
    ) from exc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap


BOUNDARY_RE = re.compile(r"^sky90_boundary_(\d+)\.dat$")


def read_area90(summary_path: Path) -> float:
    """Read the 90% credible sky area from a summary_#.dat file."""
    with summary_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Summary row in {summary_path} has fewer than 4 columns")

            return float(fields[3])

    raise ValueError(f"No source-summary row found in {summary_path}")


def discover_boundary_files(map_dir: Path) -> list[tuple[int, Path]]:
    """Return available boundary maps sorted by source number."""
    files: list[tuple[int, Path]] = []

    for path in map_dir.glob("sky90_boundary_*.dat"):
        match = BOUNDARY_RE.match(path.name)
        if match is None:
            continue
        files.append((int(match.group(1)), path))

    files.sort(key=lambda item: item[0])
    return files


def filter_by_area90(
    files: list[tuple[int, Path]],
    summary_dir: Path,
    max_area90: float,
) -> list[tuple[int, Path]]:
    """Keep only sources with area90 at or below the requested threshold."""
    if max_area90 <= 0.0:
        print("Area90 filter disabled because --max-area90 <= 0", file=sys.stderr)
        return files

    kept: list[tuple[int, Path]] = []
    skipped_large = 0
    skipped_missing = 0

    for source_id, path in files:
        summary_path = summary_dir / f"summary_{source_id}.dat"
        try:
            area90 = read_area90(summary_path)
        except (OSError, ValueError) as exc:
            print(
                f"Warning: skipping source {source_id}; could not read area90 from {summary_path}: {exc}",
                file=sys.stderr,
            )
            skipped_missing += 1
            continue

        if np.isfinite(area90) and area90 <= max_area90:
            kept.append((source_id, path))
        else:
            skipped_large += 1

    print(
        f"Area90 filter: kept {len(kept)}/{len(files)} sources with area90 <= {max_area90:g} sq deg; "
        f"excluded {skipped_large} large-area sources and {skipped_missing} sources without usable summaries.",
        file=sys.stderr,
    )

    if not kept:
        raise ValueError(
            f"No boundary files remain after applying area90 <= {max_area90:g} sq deg"
        )

    return kept


def select_boundary_files(
    files: list[tuple[int, Path]],
    requested_n: int,
    random_selection: bool,
    seed: int,
    start_id: int | None,
) -> list[tuple[int, Path]]:
    """Select the boundary maps to combine."""
    if requested_n < 1:
        raise ValueError("N must be at least 1")

    candidates = files
    if start_id is not None:
        candidates = [(source_id, path) for source_id, path in files if source_id >= start_id]

    if not candidates:
        raise ValueError("No boundary files match the requested selection")

    if requested_n > len(candidates):
        print(
            f"Warning: requested N={requested_n}, but only {len(candidates)} boundary files are available. "
            f"Using all {len(candidates)} files.",
            file=sys.stderr,
        )
        requested_n = len(candidates)

    if random_selection:
        rng = random.Random(seed)
        selected = rng.sample(candidates, requested_n)
        selected.sort(key=lambda item: item[0])
        return selected

    return candidates[:requested_n]


def infer_nside(npix: int) -> int:
    """Infer NSIDE from a HEALPix map length and validate it."""
    nside = int(round(np.sqrt(npix / 12.0)))
    if 12 * nside * nside != npix:
        raise ValueError(f"Map length {npix} is not a valid HEALPix npix=12*nside^2")
    return nside


def load_boundary_union(selected: list[tuple[int, Path]]) -> tuple[np.ndarray, int, list[int]]:
    """Read selected boundary maps and return their binary pixel-wise union."""
    boundary_union: np.ndarray | None = None
    nside: int | None = None
    source_ids: list[int] = []

    for index, (source_id, path) in enumerate(selected, start=1):
        data = np.loadtxt(path, dtype=np.float64)
        if data.ndim != 1:
            raise ValueError(f"{path} is not a one-column HEALPix map")

        this_nside = infer_nside(len(data))
        if boundary_union is None:
            nside = this_nside
            boundary_union = np.zeros_like(data)
        elif len(data) != len(boundary_union):
            raise ValueError(
                f"{path} has {len(data)} pixels, expected {len(boundary_union)} pixels from the first map"
            )

        boundary_union = np.maximum(boundary_union, (data > 0.0).astype(np.float64))
        source_ids.append(source_id)

        if index % 25 == 0 or index == len(selected):
            print(f"Loaded {index}/{len(selected)} boundary maps", file=sys.stderr)

    if boundary_union is None or nside is None:
        raise ValueError("No maps were loaded")

    return boundary_union, nside, source_ids


def make_boundary_cmap(max_count: float):
    """Return an invertible colormap suitable for healpy's colorbar code."""
    cmap = LinearSegmentedColormap.from_list("boundary_union", ["white", "#111111"])
    cmap.set_bad("white")
    cmap.set_under("white")
    return cmap


def add_sky_labels() -> None:
    """Add lightweight RA/Dec labels matching the existing sky-map style."""
    for hour in range(2, 24, 2):
        text = hp.projtext(
            hour * 180.0 / 12.0 + 5.0,
            4.0,
            f"{hour}h",
            lonlat=True,
            coord="G",
            fontsize="medium",
            fontweight=100,
            zorder=10,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

    for dec in range(-75, 0, 15):
        text = hp.projtext(
            360.0,
            dec + 7.0 * dec / 75.0,
            f"{dec}" + r"$^\circ$",
            lonlat=True,
            coord="G",
            fontsize="medium",
            fontweight=100,
            zorder=10,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])

    for dec in range(0, 90, 15):
        text = hp.projtext(
            360.0,
            dec - 3.0 * dec / 75.0,
            f"{dec}" + r"$^\circ$",
            lonlat=True,
            coord="G",
            fontsize="medium",
            fontweight=100,
            zorder=10,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white")])


def render_boundary_map(
    boundary_union: np.ndarray,
    source_ids: list[int],
    output_stem: Path,
    formats: list[str],
    dpi: int,
    graticule: bool,
    max_area90: float,
) -> list[Path]:
    """Render the binary boundary-union map."""
    max_count = float(np.max(boundary_union))
    cmap = make_boundary_cmap(max_count)
    title = f"90% sky-localization boundary union for {len(source_ids)} detected sources"
    plot_map = boundary_union.copy()
    plot_map[plot_map <= 0.0] = hp.UNSEEN

    hp.mollview(
        plot_map,
        title=title,
        rot=[-180, 0, 0],
        flip="astro",
        min=1.0 if max_count > 1.0 else 0.0,
        max=max_count if max_count > 1.0 else 1.0,
        cbar=False,
        cmap=cmap,
        badcolor="white",
        bgcolor="white",
        notext=(not graticule),
        xsize=1800,
    )

    if graticule:
        hp.graticule(dpar=15, dmer=30, color="0.72")
        add_sky_labels()

    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        ax.set_facecolor("white")

    label_lines = [
        f"N = {len(source_ids)}",
        f"source IDs {source_ids[0]}-{source_ids[-1]}",
    ]
    if max_area90 > 0.0:
        label_lines.append(rf"area90 $\leq$ {max_area90:g} deg$^2$")
    label_lines.append(f"nonzero pixels = {np.count_nonzero(boundary_union > 0.0)}")
    label = "\n".join(label_lines)
    fig.text(
        0.015,
        0.02,
        label,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(path, facecolor="white", bbox_inches="tight", dpi=dpi)
        written.append(path)

    plt.close(fig)
    return written


def write_union_files(output_stem: Path, boundary_union: np.ndarray, source_ids: list[int]) -> tuple[Path, Path]:
    """Save the binary boundary-union map and selected source IDs."""
    map_path = output_stem.with_suffix(".dat")
    ids_path = output_stem.with_name(output_stem.name + "_source_ids.txt")

    np.savetxt(map_path, boundary_union, fmt="%.0f")
    with ids_path.open("w", encoding="utf-8") as handle:
        handle.write("# source_ids included in the binary boundary-union map\n")
        for source_id in source_ids:
            handle.write(f"{source_id}\n")

    return map_path, ids_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine N sky90_boundary_*.dat maps and render their 90% contour union."
    )
    parser.add_argument("N", type=int, help="Number of boundary maps to select and combine.")
    parser.add_argument(
        "--map-dir",
        default="skymaps",
        type=Path,
        help="Directory containing sky90_boundary_*.dat files. Default: skymaps",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        type=Path,
        help="Directory containing summary_*.dat files. Default: same as --map-dir",
    )
    parser.add_argument(
        "--max-area90",
        default=50.0,
        type=float,
        help="Exclude sources with 90% credible area larger than this many square degrees. Use <=0 to disable. Default: 50",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output path stem. Default: <map-dir>/sky90_boundary_sample_<N_used>",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Select a random subset rather than the first N source IDs.",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Random seed used with --random. Default: 1234",
    )
    parser.add_argument(
        "--start-id",
        default=None,
        type=int,
        help="For non-random selection, start from the first source ID >= this value.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output plot formats. Default: png pdf",
    )
    parser.add_argument(
        "--dpi",
        default=220,
        type=int,
        help="Raster DPI for PNG output. Default: 220",
    )
    parser.add_argument(
        "--no-graticule",
        action="store_true",
        help="Suppress RA/Dec graticule lines and labels.",
    )
    parser.add_argument(
        "--no-save-union",
        action="store_true",
        help="Do not write the binary boundary-union map and selected source-ID list.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = discover_boundary_files(args.map_dir)

    if not files:
        print(f"No sky90_boundary_*.dat files found in {args.map_dir}", file=sys.stderr)
        return 1

    summary_dir = args.summary_dir if args.summary_dir is not None else args.map_dir
    try:
        files = filter_by_area90(files, summary_dir, args.max_area90)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    selected = select_boundary_files(files, args.N, args.random, args.seed, args.start_id)
    boundary_union, nside, source_ids = load_boundary_union(selected)

    output_stem = args.output
    if output_stem is None:
        output_stem = args.map_dir / f"sky90_boundary_sample_{len(source_ids)}"

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Combined {len(source_ids)} boundary maps at NSIDE={nside}; "
        f"boundary pixels = {np.count_nonzero(boundary_union > 0.0)}"
    )

    if not args.no_save_union:
        map_path, ids_path = write_union_files(output_stem, boundary_union, source_ids)
        print(f"Wrote {map_path}")
        print(f"Wrote {ids_path}")

    for path in render_boundary_map(
        boundary_union,
        source_ids,
        output_stem,
        args.formats,
        args.dpi,
        graticule=(not args.no_graticule),
        max_area90=args.max_area90,
    ):
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
