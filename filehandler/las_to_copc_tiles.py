"""Split large LAS files into COPC.laz tiles, matching the existing
Linker_Rhein scheme: COPC/Prio_<N>/segment_<i>.copc.laz (no prefix, no
zero padding).

Unlike the older version, cutting does NOT follow a rigid square grid
(filters.splitter) but the real track axis (as in the colleague script
segmentiere_200m.py): every point is projected onto the nearest position on
the axis via KDTree and binned into equal-length pieces by distance along
that axis. This gives far more even tile sizes, even where the track curves
(a single global rotate + square grid cannot do that).

Two ways to obtain the axis:

1. --axis-file <path>: text file with one axis vertex per line, in one of two
   auto-detected formats:
     a) whitespace-separated "station x y" or just "x y".
     b) semicolon-separated "x;y;z;code;name" (combined-control-point format,
        e.g. passpunkte_kombiniert.txt) - z, code and name are ignored.
   In both cases: '.' or ',' as the decimal separator, '#' lines are ignored.
   When the station column is missing (format b, or "x y" in format a), it is
   computed as the cumulative distance along the (already ordered) points.
   Passing a file skips the expensive automatic axis estimation, which keeps
   runtime low.

2. No file given: the axis is estimated from the point cloud itself (PCA
   principal direction + windowed means along that direction, so curves are
   captured too). This costs one extra, decimated read of the input file.

Requires: pdal CLI on PATH (tested with 2.10.1), plus laspy, numpy and scipy
(for axis smoothing).

    python las_to_copc_tiles.py \
        --input km_29-30.las km_30-31.las \
        --out /path/to/COPC \
        --start-prio 1 \
        --edge-length 100 \
        --axis-file track_axis.txt
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Rough plausibility bounding boxes (with margin) for common German CRS, in
# their respective projection (metres). Only a sanity check against a wrong or
# forgotten --a-srs, NOT an exact boundary.
GERMANY_BBOX_PER_EPSG = {
    "EPSG:5683": (3_280_000, 3_920_000, 5_235_000, 6_110_000),  # DB_REF / GK Zone 3
    "EPSG:5684": (4_280_000, 4_920_000, 5_235_000, 6_110_000),  # DB_REF / GK Zone 4
    "EPSG:31467": (3_280_000, 3_920_000, 5_235_000, 6_110_000),  # GK Zone 3 (alt)
    "EPSG:31468": (4_280_000, 4_920_000, 5_235_000, 6_110_000),  # GK Zone 4 (alt)
    "EPSG:25832": (150_000, 950_000, 5_235_000, 6_110_000),  # UTM 32N
    "EPSG:25833": (150_000, 950_000, 5_235_000, 6_110_000),  # UTM 33N
}


def check_location_plausible(
    xmin: float, xmax: float, ymin: float, ymax: float, a_srs: str, context: str
) -> None:
    """Rough sanity check: do the coordinates sit roughly where one would
    expect them for the given --a-srs in Germany? Warning only, never fatal -
    it can fire wrongly on unusual but correct data (abroad, test data)."""
    box = GERMANY_BBOX_PER_EPSG.get(a_srs.strip().upper())
    if box is None:
        return  # unknown or no CRS -> cannot be checked
    bx_min, bx_max, by_min, by_max = box
    if xmax < bx_min or xmin > bx_max or ymax < by_min or ymin > by_max:
        print(
            f"WARNING ({context}): coordinates X={xmin:.0f}..{xmax:.0f}, Y={ymin:.0f}..{ymax:.0f} "
            f"fall outside the range expected for {a_srs} in Germany "
            f"(X={bx_min}..{bx_max}, Y={by_min}..{by_max}). "
            f"--a-srs may be wrong, or the data may use a different CRS than "
            f"assumed - please check.",
            file=sys.stderr,
        )


# ------------------------- Axis: load or estimate -------------------------


def load_axis_from_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a position file in one of two formats (detected per line):

    1. Whitespace-separated: 'station x y' or just 'x y'.
    2. Semicolon-separated (combined-control-point format):
       'x;y;z;code;name' - z, code and name are ignored, only x and y are
       used. Station is then always computed from the cumulative distance in
       row order.

    In both cases: '.' or ',' as the decimal separator, '#' lines are ignored,
    and a leading BOM is stripped automatically.
    Returns (xy, station), sorted by station."""
    rows: list[tuple[float | None, float, float]] = []
    with open(path, encoding="utf-8-sig") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ";" in line:
                # combined-control-point format: x;y;z;code;name
                fields = line.split(";")
                if len(fields) < 2:
                    continue
                try:
                    x = float(fields[0].strip().replace(",", "."))
                    y = float(fields[1].strip().replace(",", "."))
                except ValueError:
                    continue
                rows.append((None, x, y))
            else:
                values = [float(t.replace(",", ".")) for t in line.split()]
                if len(values) >= 3:
                    rows.append((values[0], values[1], values[2]))
                elif len(values) == 2:
                    rows.append((None, values[0], values[1]))
    if len(rows) < 2:
        raise ValueError(f"axis file {path} has too few valid rows (at least 2 needed).")

    xy = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
    if rows[0][0] is None:
        # no station column -> cumulative distance in row order
        d = np.hypot(*np.diff(xy, axis=0).T)
        station = np.concatenate([[0.0], np.cumsum(d)])
    else:
        station = np.array([r[0] for r in rows], dtype=np.float64)
        order = np.argsort(station)
        station = station[order]
        xy = xy[order]
    return xy, station


def estimate_axis_from_cloud(
    path: Path,
    chunk_size: int = 2_000_000,
    sample_step: int = 20,
    window_length: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a curved track axis straight from the point cloud: coarse
    direction via PCA, then windowed XY means along that direction so curves
    are captured (rather than just a straight line)."""
    import laspy

    xs, ys = [], []
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            xs.append(np.asarray(chunk.x[::sample_step]))
            ys.append(np.asarray(chunk.y[::sample_step]))
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy
    cov = np.cov(dx, dy)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    yaw = float(np.arctan2(principal_axis[1], principal_axis[0]))

    # project along the PCA principal direction, then group into windows
    t = dx * np.cos(yaw) + dy * np.sin(yaw)
    bin_idx = np.floor((t - t.min()) / window_length).astype(np.int64)
    order = np.argsort(bin_idx)
    bin_sorted = bin_idx[order]
    x_sorted, y_sorted = x[order], y[order]
    bounds_idx = np.searchsorted(bin_sorted, np.unique(bin_sorted))
    bounds_idx = np.append(bounds_idx, len(bin_sorted))

    vertices = []
    for i in range(len(bounds_idx) - 1):
        a, b = bounds_idx[i], bounds_idx[i + 1]
        vertices.append((x_sorted[a:b].mean(), y_sorted[a:b].mean()))
    xy = np.array(vertices, dtype=np.float64)

    d = np.hypot(*np.diff(xy, axis=0).T)
    station = np.concatenate([[0.0], np.cumsum(d)])
    return xy, station


def densify_axis(xy: np.ndarray, station: np.ndarray, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Lightly smooth (when scipy is present and there are enough vertices) and
    densify the axis onto a fine point raster for the KDTree projection."""
    try:
        from scipy.interpolate import UnivariateSpline

        if len(station) >= 8:
            sx = UnivariateSpline(station, xy[:, 0], s=len(station) * 0.25)
            sy = UnivariateSpline(station, xy[:, 1], s=len(station) * 0.25)
            test_s = np.linspace(station[0], station[-1], min(2000, len(station) * 10))
            xy_test = np.column_stack([sx(test_s), sy(test_s)])
            # plausibility check: the smoothed curve must not run wild
            if np.all(
                np.hypot(
                    *(
                        xy_test
                        - np.column_stack(
                            [
                                np.interp(test_s, station, xy[:, 0]),
                                np.interp(test_s, station, xy[:, 1]),
                            ]
                        )
                    ).T
                )
                < step * 50
            ):
                xy = xy_test
                station = test_s
    except ImportError:
        pass

    dxy_parts, ds_parts = [], []
    for i in range(len(station) - 1):
        seg_len = station[i + 1] - station[i]
        if seg_len <= 0:
            continue
        n = max(2, int(seg_len / step))
        t = np.linspace(0, 1, n, endpoint=False)
        dxy_parts.append(xy[i] + t[:, None] * (xy[i + 1] - xy[i]))
        ds_parts.append(station[i] + t * seg_len)
    dxy_parts.append(xy[-1:])
    ds_parts.append(station[-1:])
    return np.vstack(dxy_parts), np.concatenate(ds_parts)


# ------------------------- PDAL helpers -------------------------


def run_pipeline(pipeline: dict, context: str) -> None:
    proc = subprocess.run(
        ["pdal", "pipeline", "--stdin"],
        input=json.dumps(pipeline),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"pdal pipeline failed: {context}")


# ------------------------- Bin writer (intermediate .las per axis section) -------------------------


class BinWriter:
    def __init__(self, path: Path, src_header):
        import laspy

        hdr = laspy.LasHeader(version=src_header.version, point_format=src_header.point_format)
        hdr.scales = src_header.scales.copy()
        hdr.offsets = src_header.offsets.copy()
        self.path = path
        self.w = laspy.open(str(path), mode="w", header=hdr)
        self.pf = src_header.point_format
        self.n = 0
        self.mins = np.array([np.inf] * 3)
        self.maxs = np.array([-np.inf] * 3)

    def write(self, raw_sub, xs, ys, zs) -> None:
        from laspy.point.record import PackedPointRecord

        self.w.write_points(PackedPointRecord(raw_sub, self.pf))
        self.n += len(raw_sub)
        self.mins = np.minimum(self.mins, [xs.min(), ys.min(), zs.min()])
        self.maxs = np.maximum(self.maxs, [xs.max(), ys.max(), zs.max()])

    def close(self) -> None:
        self.w.close()
        # Fix the header bounds (byte 179, same position in LAS 1.2-1.4).
        # Purely cosmetic - pdal recomputes the COPC bounds in stage 2 anyway.
        with open(self.path, "r+b") as f:
            f.seek(179)
            f.write(
                struct.pack(
                    "<6d",
                    self.maxs[0],
                    self.mins[0],
                    self.maxs[1],
                    self.mins[1],
                    self.maxs[2],
                    self.mins[2],
                )
            )


# ------------------------- Main processing per input file -------------------------


def process_file(
    input_path: Path,
    prio_dir: Path,
    edge_length: float,
    a_srs: str,
    tmp_root: Path,
    workers: int,
    axis_xy: np.ndarray | None,
    axis_station: np.ndarray | None,
    chunk_size: int,
    sample_step: int,
    window_length: float,
) -> None:
    import laspy
    from scipy.spatial import cKDTree

    prio_dir.mkdir(parents=True, exist_ok=True)

    if axis_xy is None:
        print(f"{input_path.name}: no --axis-file given, estimating axis from the point cloud...")
        axis_xy, axis_station = estimate_axis_from_cloud(
            input_path, chunk_size=chunk_size, sample_step=sample_step, window_length=window_length
        )
        print(f"  -> {len(axis_xy)} estimated axis vertices")
    else:
        print(f"{input_path.name}: using axis from file ({len(axis_xy)} vertices)")

    dens_xy, dens_station = densify_axis(axis_xy, axis_station)
    tree = cKDTree(dens_xy)

    tmp_dir = tmp_root / f"{prio_dir.name}_achsen_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[int, BinWriter] = {}
    with laspy.open(str(input_path)) as r:
        check_location_plausible(
            r.header.mins[0],
            r.header.maxs[0],
            r.header.mins[1],
            r.header.maxs[1],
            a_srs,
            input_path.name,
        )
        total = r.header.point_count
        done = 0
        print(f"{input_path.name}: {total} points, binning along the axis (edge length {edge_length} m)")
        while done < total:
            n = min(chunk_size, total - done)
            ch = r.read_points(n)
            _, idx = tree.query(np.column_stack([ch.x, ch.y]), workers=-1)
            station = dens_station[idx]
            bidx = np.floor(station / edge_length).astype(np.int64)
            arr = ch.array
            for b in np.unique(bidx):
                m = bidx == b
                if int(b) not in writers:
                    dst = tmp_dir / f"bin_{int(b)}.las"
                    writers[int(b)] = BinWriter(dst, r.header)
                writers[int(b)].write(arr[m], ch.x[m], ch.y[m], ch.z[m])
            done += n
            if done % (20 * chunk_size) == 0 or done >= total:
                print(f"  ... {100.0 * done / total:5.1f} %  ({done} / {total})", flush=True)

    for w in writers.values():
        w.close()
    print(f"  -> {len(writers)} axis sections (intermediate .las)")

    # Stage 2: convert each section to .copc.laz individually, renamed to the
    # existing scheme segment_<i>.copc.laz (starting at 1, ordered by position
    # along the axis). Independent PDAL subprocesses -> parallelisable.
    sorted_bins = sorted(writers.keys())

    def convert(i_bin: tuple[int, int]) -> None:
        i, b = i_bin
        src = writers[b].path
        dst = prio_dir / f"segment_{i}.copc.laz"
        writer_stage: dict = {"type": "writers.copc", "filename": str(dst), "forward": "all"}
        if a_srs:
            writer_stage["a_srs"] = a_srs
        pipeline = {"pipeline": [{"type": "readers.las", "filename": str(src)}, writer_stage]}
        run_pipeline(pipeline, f"converting {src} -> {dst}")

    tasks = list(enumerate(sorted_bins, start=1))
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(convert, task): task for task in tasks}
        for finished, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except BaseException as exc:
                errors.append(exc)
            if finished % 10 == 0 or finished == len(tasks):
                print(f"  conversion: {finished}/{len(tasks)}", flush=True)

    if errors:
        raise RuntimeError(
            f"{len(errors)} of {len(tasks)} conversions failed for {input_path} - first error: {errors[0]}"
        )

    produced = sorted(prio_dir.glob("segment_*.copc.laz"))
    print(f"  -> {len(produced)} COPC tiles -> {prio_dir}")
    if produced:
        print(f"  example filename: {produced[0].name}")

    for w in writers.values():
        w.path.unlink()
    tmp_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True, help="LAS input files, one per Prio folder, in order"
    )
    parser.add_argument("--out", type=Path, required=True, help="COPC root directory (e.g. .../QGIS/COPC)")
    parser.add_argument(
        "--start-prio", type=int, default=1, help="Prio number of the first input file (default 1)"
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=100.0,
        help="segment length along the axis in metres (default 100)",
    )
    parser.add_argument(
        "--a-srs",
        default="EPSG:5683",
        help="assign a CRS without reprojecting (default EPSG:5683). Empty string = no assignment.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/tmp"),
        help="location for intermediate .las sections (default /tmp)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel PDAL processes for stage 2. Default: CPU core count.",
    )
    parser.add_argument(
        "--axis-file",
        type=Path,
        default=None,
        help="optional text file holding the track axis, one vertex per line. "
        "Two formats are auto-detected: whitespace-separated 'station x y' or "
        "'x y'; or semicolon-separated 'x;y;z;code;name' as in "
        "passpunkte_kombiniert.txt (z, code, name are ignored). "
        "'.' or ',' as the decimal separator, '#' = comment. Passing a file "
        "skips the automatic axis estimation from the point cloud (saves time). "
        "Applies to all --input files together.",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=20.0,
        help="only without --axis-file: window width (m) for automatic axis "
        "estimation (default 20). Smaller follows curves more closely but is "
        "more sensitive to point noise.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=20,
        help="only without --axis-file: decimation for axis estimation (every nth point, default 20).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="points per processing block during the main read (default 2 million)",
    )
    args = parser.parse_args()
    workers = args.workers or os.cpu_count() or 4

    axis_xy = axis_station = None
    if args.axis_file:
        axis_xy, axis_station = load_axis_from_file(args.axis_file)
        print(f"axis file loaded: {len(axis_xy)} vertices from {args.axis_file}")
        check_location_plausible(
            axis_xy[:, 0].min(),
            axis_xy[:, 0].max(),
            axis_xy[:, 1].min(),
            axis_xy[:, 1].max(),
            args.a_srs,
            f"axis file {args.axis_file.name}",
        )

    for i, input_path in enumerate(args.input_path):
        prio_no = args.start_prio + i
        prio_dir = args.out_root / f"Prio_{prio_no}"
        start = time.monotonic()
        print(
            f"=== [{i + 1}/{len(args.input_path)}] {input_path.name} -> Prio_{prio_no} "
            f"(started {time.strftime('%H:%M:%S')}, {workers} workers) ==="
        )
        process_file(
            input_path,
            prio_dir,
            args.edge_length,
            args.a_srs,
            args.tmp_dir,
            workers,
            axis_xy,
            axis_station,
            args.chunk_size,
            args.sample_step,
            args.window_length,
        )
        duration = time.monotonic() - start
        print(f"=== Prio_{prio_no} finished nach {duration / 60:.1f} min ===\n")


if __name__ == "__main__":
    main()
