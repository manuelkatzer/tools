"""Work out which LAS files belong to a ground-truth file (survey targets,
.shp/.gpkg) and tile only those into COPC.

Core idea: a LAS file reveals its bounding box in the first 375 bytes (public
header block, offset 179). So there is no need to download gigabytes just to
find out where a file is located - a few hundred bytes per file are enough,
whether reading locally, over a network share, or via an HTTP range request.

Three subcommands:

  index        Reads only the headers of all LAS/LAZ files, compares them
               against the GT file and writes
                 - footprints.csv  (WKT rectangles, loadable in QGIS as
                   "Delimited Text" -> drop it next to the GT and look)
                 - matches.csv     (the matching files, input for stage 3)
               It also tries to determine the GT <-> LAS CRS relationship
               automatically, which is the most common source of error.

  axis         Builds a track-axis file from trajectory.csv (or from the GT
               file) in the format las_zu_copc_tiles.py expects.

  process      Walks through matches.csv: make the file available (read local
               and network paths directly, or copy into --tmp-dir first with
               --copy-local, or download over HTTP), split it into COPC tiles
               with las_zu_copc_tiles.py, then delete any temporary LAS.

Dependencies:
  index/axis    numpy only. Optional: geopandas or fiona (otherwise the
                built-in minimal readers for .shp and .gpkg are used), and
                pyproj for real CRS transformations.
  process       additionally everything las_zu_copc_tiles.py needs
                (laspy, scipy, pdal CLI).

Examples:

    # 1. Which files match at all?  (pass the network path directly, only
    #    ~375 bytes per file are read)
    python las_gt_pipeline.py index \\
        --gt target_kontrolle.gpkg \\
        --las-dir "\\\\192.168.50.10\\Befahrungen\\24067_GSH_Hagen_Hamm" \\
        --index-out ./index

    # 2. Build the axis from the trajectory
    python las_gt_pipeline.py axis \\
        --from-csv .../2103/trajectory.csv --out achse_2103.txt

    # 3. Process only the first 10 matches (start small!)
    python las_gt_pipeline.py process \\
        --matches ./index/matches.csv \\
        --out /path/zu/QGIS/COPC \\
        --axis-file achse_2103.txt \\
        --edge-length 25 --limit 10 --copy-local
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

LAS_HEADER_BYTES = 375  # enough for LAS 1.0 - 1.4
VLR_MAX_BYTES = 65_536  # upper read limit for CRS detection


# =========================== Sources (local / HTTP) ===========================


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def read_head_bytes(source: str, n: int) -> bytes:
    """Read the first n bytes of a file - directly for local/UNC paths, as a
    range request over HTTP. Servers without range support send the whole file,
    but we still read only n bytes and then stop."""
    if is_url(source):
        import urllib.request

        req = urllib.request.Request(source, headers={"Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read(n)
    with open(source, "rb") as fh:
        return fh.read(n)


def collect_sources(
    directory: Path | None, url_list: Path | None, pattern: str
) -> list[str]:
    sources: list[str] = []
    if directory:
        for path in sorted(directory.rglob(pattern)):
            if path.is_file():
                sources.append(str(path))
        # also pick up the second extension when using the default pattern
        if pattern == "*.las":
            for path in sorted(directory.rglob("*.laz")):
                if path.is_file():
                    sources.append(str(path))
    if url_list:
        with open(url_list, encoding="utf-8-sig") as fh:
            for row in fh:
                row = row.strip()
                if row and not row.startswith("#"):
                    sources.append(row)
    return sources


# =========================== Read LAS header ===========================


def parse_las_header(raw: bytes) -> dict:
    """Parse the public header block. Works for LAS 1.0-1.4 and equally for
    LAZ, because the header sits uncompressed at the front there too."""
    if len(raw) < 227 or raw[:4] != b"LASF":
        raise ValueError("not a valid LAS/LAZ file (missing 'LASF' signature)")
    version = (raw[24], raw[25])
    header_size = struct.unpack_from("<H", raw, 94)[0]
    point_offset = struct.unpack_from("<I", raw, 96)[0]
    vlr_count = struct.unpack_from("<I", raw, 100)[0]
    point_format = raw[104] & 0b0011_1111  # top bits = LAZ compression flag
    point_length = struct.unpack_from("<H", raw, 105)[0]
    count = struct.unpack_from("<I", raw, 107)[0]
    scales = struct.unpack_from("<3d", raw, 131)
    offsets = struct.unpack_from("<3d", raw, 155)
    maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", raw, 179)
    if version >= (1, 4) and len(raw) >= 255:
        count_14 = struct.unpack_from("<Q", raw, 247)[0]
        if count_14:
            count = count_14
    return {
        "version": f"{version[0]}.{version[1]}",
        "header_size": header_size,
        "point_offset": point_offset,
        "vlr_count": vlr_count,
        "point_format": point_format,
        "point_length": point_length,
        "points": int(count),
        "scales": scales,
        "offsets": offsets,
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy,
        "minz": minz,
        "maxz": maxz,
    }


def epsg_from_vlrs(raw: bytes, header_size: int, vlr_count: int) -> str | None:
    """Look for the CRS in the VLRs: WKT first (record_id 2112), otherwise the
    GeoTIFF keys (record_id 34735, key 3072 = ProjectedCSTypeGeoKey)."""
    pos = header_size
    for _ in range(vlr_count):
        if pos + 54 > len(raw):
            return None
        user_id = raw[pos + 2 : pos + 18].rstrip(b"\x00").decode("latin-1", "ignore")
        record_id = struct.unpack_from("<H", raw, pos + 18)[0]
        length = struct.unpack_from("<H", raw, pos + 20)[0]
        data = raw[pos + 54 : pos + 54 + length]
        pos += 54 + length
        if "LASF_Projection" not in user_id:
            continue
        if record_id == 2112 and data:  # WKT
            wkt = data.rstrip(b"\x00").decode("latin-1", "ignore")
            matches = re.findall(
                r'(?:AUTHORITY|ID)\s*\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]', wkt
            )
            if matches:
                return f"EPSG:{matches[-1]}"
            return "WKT without EPSG code"
        if record_id == 34735 and len(data) >= 8:  # GeoKeyDirectory
            _, _, _, n_keys = struct.unpack_from("<4H", data, 0)
            for i in range(n_keys):
                off = 8 + i * 8
                if off + 8 > len(data):
                    break
                key_id, tiff_tag, _count, value = struct.unpack_from("<4H", data, off)
                if key_id == 3072 and tiff_tag == 0 and value not in (0, 32767):
                    return f"EPSG:{value}"
    return None


def read_las_info(source: str, mit_crs: bool = True) -> dict:
    raw = read_head_bytes(source, LAS_HEADER_BYTES)
    info = parse_las_header(raw)
    info["source"] = source
    info["name"] = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    info["crs"] = None
    if mit_crs and info["vlr_count"]:
        try:
            n = min(info["point_offset"], VLR_MAX_BYTES)
            if n > len(raw):  # VLRs extend beyond the first read
                raw = read_head_bytes(source, n)
            info["crs"] = epsg_from_vlrs(raw, info["header_size"], info["vlr_count"])
        except Exception:  # noqa: BLE001 - CRS is extra info only
            pass
    return info


# =========================== Filename -> kilometre ===========================

KM_PATTERN = re.compile(r"[_\-](\d{1,4}[.,]\d+)[_\-](\d{1,4}[.,]\d+)[_\-]")


def km_from_name(name: str) -> tuple[float | None, float | None]:
    """Detect kilometre markers such as '..._180,0_180,1_...' in the filename."""
    m = KM_PATTERN.search(name)
    if not m:
        return None, None
    return float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))


# =========================== Load GT file ===========================


def _load_gt_geopandas(path: Path, columns: tuple[str, str] | None):
    import geopandas as gpd  # noqa: PLC0415

    g = gpd.read_file(path)
    crs = g.crs.to_string() if g.crs is not None else None
    if columns:
        xy = np.column_stack(
            [g[columns[0]].to_numpy(float), g[columns[1]].to_numpy(float)]
        )
        return xy, None  # CRS of the attributes is unknown
    geo = g.geometry.representative_point()
    return np.column_stack([geo.x.to_numpy(), geo.y.to_numpy()]), crs


def _load_gt_fiona(path: Path, columns: tuple[str, str] | None):
    import fiona  # noqa: PLC0415

    points = []
    with fiona.open(path) as src:
        crs = src.crs_wkt or (str(src.crs) if src.crs else None)
        for f in src:
            if columns:
                p = f["properties"]
                points.append((float(p[columns[0]]), float(p[columns[1]])))
            else:
                koord = f["geometry"]["coordinates"]
                while isinstance(koord[0], (list, tuple)):
                    koord = koord[0]
                points.append((float(koord[0]), float(koord[1])))
    return np.array(points, dtype=np.float64), (None if columns else crs)


def _load_gt_shp(path: Path):
    """Minimal reader for point shapefiles (type 1/11/21), no third-party deps."""
    data = path.read_bytes()
    if struct.unpack_from(">i", data, 0)[0] != 9994:
        raise ValueError(f"{path} is not a shapefile")
    points = []
    pos = 100
    while pos + 8 <= len(data):
        _nr, length_words = struct.unpack_from(">2i", data, pos)
        inhalt = pos + 8
        geom_type = struct.unpack_from("<i", data, inhalt)[0]
        if geom_type in (1, 11, 21):
            x, y = struct.unpack_from("<2d", data, inhalt + 4)
            points.append((x, y))
        pos = inhalt + length_words * 2
    crs = None
    prj = path.with_suffix(".prj")
    if prj.is_file():
        wkt = prj.read_text(encoding="utf-8-sig", errors="ignore")
        m = re.findall(r'(?:AUTHORITY|ID)\s*\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]', wkt)
        crs = f"EPSG:{m[-1]}" if m else wkt[:120]
    return np.array(points, dtype=np.float64), crs


def _gpkg_blob_to_xy(blob: bytes) -> tuple[float, float] | None:
    """Parse a GeoPackage geometry blob (point) without third-party deps."""
    if len(blob) < 8 or blob[:2] != b"GP":
        return None
    flags = blob[3]
    envelope_type = (flags >> 1) & 0b111
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_type)
    if envelope_bytes is None or (flags >> 4) & 1:  # unknown or empty geometry
        return None
    pos = 8 + envelope_bytes
    if pos + 21 > len(blob):
        return None
    little = blob[pos] == 1
    fmt = "<" if little else ">"
    geom_type = struct.unpack_from(fmt + "I", blob, pos + 1)[0] % 1000
    if geom_type != 1:  # points only
        return None
    x, y = struct.unpack_from(fmt + "2d", blob, pos + 5)
    return x, y


def _load_gt_gpkg(path: Path, layer: str | None, columns: tuple[str, str] | None):
    """Minimal reader for point GeoPackages via sqlite3."""
    import sqlite3  # noqa: PLC0415

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT table_name, srs_id FROM gpkg_contents WHERE data_type='features'"
        ).fetchall()
        if not rows:
            raise ValueError("no vector layers found in the GeoPackage")
        table, srs_id = rows[0]
        if layer:
            for t, s in rows:
                if t == layer:
                    table, srs_id = t, s
                    break
        elif len(rows) > 1:
            print(
                f"  note: {len(rows)} layers in the GPKG, using '{table}' "
                f"(others: {', '.join(t for t, _ in rows[1:])})"
            )
        if columns:
            rows = con.execute(
                f'SELECT "{columns[0]}","{columns[1]}" FROM "{table}"'
            ).fetchall()
            xy = np.array(
                [(float(a), float(b)) for a, b in rows if a is not None],
                dtype=np.float64,
            )
            return xy, None
        geom_column = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
            (table,),
        ).fetchone()[0]
        points = []
        for (blob,) in con.execute(f'SELECT "{geom_column}" FROM "{table}"'):
            if blob:
                xy = _gpkg_blob_to_xy(blob)
                if xy:
                    points.append(xy)
        return np.array(points, dtype=np.float64), (
            f"EPSG:{srs_id}" if srs_id else None
        )
    finally:
        con.close()


def load_gt(
    path: Path, layer: str | None, columns: tuple[str, str] | None
) -> tuple[np.ndarray, str | None]:
    errors = []
    for name, func in (
        ("geopandas", lambda: _load_gt_geopandas(path, columns)),
        ("fiona", lambda: _load_gt_fiona(path, columns)),
    ):
        try:
            xy, crs = func()
            print(f"GT read with {name}: {len(xy)} points, CRS per file: {crs}")
            return xy, crs
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".gpkg":
            xy, crs = _load_gt_gpkg(path, layer, columns)
        elif suffix == ".shp":
            xy, crs = _load_gt_shp(path)
        else:
            raise ValueError(f"format {suffix} not readable without geopandas/fiona")
        print(f"GT read with built-in reader: {len(xy)} points, CRS per file: {crs}")
        return xy, crs
    except Exception as exc:  # noqa: BLE001
        errors.append(f"built-in: {exc}")
    raise SystemExit("could not read the GT file:\n  " + "\n  ".join(errors))


def remove_outliers(xy: np.ndarray, max_distance: float) -> np.ndarray:
    """Drop points far from the median of the cloud - typically dummy/zero rows
    that would otherwise inflate the bounding box."""
    if max_distance <= 0 or len(xy) < 3:
        return xy
    center = np.median(xy, axis=0)
    d = np.hypot(*(xy - center).T)
    kept = d <= max_distance
    if not kept.all():
        print(
            f"  {int((~kept).sum())} GT point(s) further than {max_distance:.0f} m from the "
            f"median -> ignored (largest distance was {d.max():.0f} m)"
        )
    return xy[kept]


# =========================== CRS candidates ===========================

CRS_CANDIDATES = [
    "EPSG:25832",
    "EPSG:25833",
    "EPSG:5683",
    "EPSG:5684",
    "EPSG:31467",
    "EPSG:31468",
]


def build_candidates(
    gt_xy: np.ndarray, gt_crs: str | None, las_crs: str | None
) -> list[tuple[str, np.ndarray]]:
    """Build variants of the GT coordinates, one of which should match the LAS
    CRS. Deliberately includes the 'missing zone prefix' cases that turn up
    constantly in Gauss-Krueger exports (3,392,674 vs. 392,674)."""
    candidates: list[tuple[str, np.ndarray]] = [("GT unchanged", gt_xy)]
    if gt_xy[:, 0].max() < 1_000_000:
        for delta, name in (
            (3_000_000, "X + 3,000,000 (GK zone 3 added)"),
            (4_000_000, "X + 4,000,000 (GK zone 4 added)"),
        ):
            candidates.append((name, gt_xy + np.array([delta, 0.0])))
    else:
        candidates.append(
            ("X - 3,000,000 (GK zone removed)", gt_xy - np.array([3_000_000.0, 0.0]))
        )
        candidates.append(
            ("X - 4,000,000 (GK zone removed)", gt_xy - np.array([4_000_000.0, 0.0]))
        )
    if not las_crs or not las_crs.upper().startswith("EPSG:"):
        return candidates
    try:
        from pyproj import Transformer  # noqa: PLC0415
    except ImportError:
        print("  note: pyproj not installed - real reprojections will not be tested.")
        return candidates
    sources = []
    if gt_crs and gt_crs.upper().startswith("EPSG:"):
        sources.append(gt_crs.upper())
    sources += [c for c in CRS_CANDIDATES if c not in sources]
    for source in sources:
        if source.upper() == las_crs.upper():
            continue
        try:
            tf = Transformer.from_crs(source, las_crs, always_xy=True)
            x, y = tf.transform(gt_xy[:, 0], gt_xy[:, 1])
            new_xy = np.column_stack([x, y])
            if np.isfinite(new_xy).all():
                candidates.append((f"GT as {source} -> {las_crs}", new_xy))
        except Exception:  # noqa: BLE001
            continue
    return candidates


def count_in_boxes(xy: np.ndarray, infos: list[dict], buffer: float) -> np.ndarray:
    """How many GT points fall inside at least one LAS bounding box?"""
    matches = np.zeros(len(xy), dtype=bool)
    for info in infos:
        matches |= (
            (xy[:, 0] >= info["minx"] - buffer)
            & (xy[:, 0] <= info["maxx"] + buffer)
            & (xy[:, 1] >= info["miny"] - buffer)
            & (xy[:, 1] <= info["maxy"] + buffer)
        )
    return matches


# =========================== Subcommand: index ===========================


def cmd_index(args: argparse.Namespace) -> None:
    sources = collect_sources(args.las_dir, args.url_list, args.pattern)
    if not sources:
        raise SystemExit("no LAS/LAZ files found - check --las-dir / --url-list.")
    print(
        f"{len(sources)} LAS/LAZ sources found, reading headers (only ~{LAS_HEADER_BYTES} bytes each)..."
    )

    infos: list[dict] = []
    failed: list[tuple[str, str]] = []
    start = time.monotonic()
    for i, source in enumerate(sources, start=1):
        try:
            infos.append(read_las_info(source, mit_crs=not args.no_crs_detection))
        except Exception as exc:  # noqa: BLE001
            failed.append((source, str(exc)))
        if i % 25 == 0 or i == len(sources):
            print(
                f"  ... {i}/{len(sources)} ({time.monotonic() - start:.0f} s)",
                flush=True,
            )
    if failed:
        print(
            f"  {len(failed)} file(s) unreadable, e.g. {failed[0][0]}: {failed[0][1]}"
        )
    if not infos:
        raise SystemExit("not a single header could be read.")

    crs_values = {i["crs"] for i in infos if i["crs"]}
    las_crs = None
    if crs_values:
        las_crs = sorted(crs_values)[0]
        print(f"CRS per LAS VLRs: {', '.join(sorted(crs_values))}")
    else:
        print("CRS per LAS VLRs: not set (these files carry no CRS!)")
    if args.las_crs:
        las_crs = args.las_crs
        print(f"LAS CRS forced to {las_crs} via --las-crs")

    gt_xy, gt_crs = load_gt(args.gt, args.gt_layer, args.gt_columns)
    gt_xy = remove_outliers(gt_xy, args.gt_outlier_distance)
    print(
        f"GT extent:  X {gt_xy[:, 0].min():.1f}..{gt_xy[:, 0].max():.1f}, "
        f"Y {gt_xy[:, 1].min():.1f}..{gt_xy[:, 1].max():.1f}"
    )
    total = (
        min(i["minx"] for i in infos),
        max(i["maxx"] for i in infos),
        min(i["miny"] for i in infos),
        max(i["maxy"] for i in infos),
    )
    print(
        f"LAS extent: X {total[0]:.1f}..{total[1]:.1f}, Y {total[2]:.1f}..{total[3]:.1f}"
    )

    print("\nCRS variants (hits = GT points falling inside at least one LAS box):")
    scored = []
    for name, xy in build_candidates(gt_xy, gt_crs, las_crs):
        n = int(count_in_boxes(xy, infos, args.buffer).sum())
        scored.append((n, name, xy))
        print(f"  {n:6d} / {len(gt_xy)}   {name}")
    scored.sort(key=lambda t: -t[0])
    best_n, best_name, best_xy = scored[0]
    if best_n == 0:
        print(
            "\nNO variant matches. Possible causes: GT and LAS really cover "
            "different sections, the CRS is not among those tested (set --las-crs, "
            "install pyproj), or --buffer is too small."
        )
    else:
        print(f"\nBest variant: {best_name} ({best_n} of {len(gt_xy)} GT points)")

    # evaluate per file
    rows = []
    for info in infos:
        inside = (
            (best_xy[:, 0] >= info["minx"] - args.buffer)
            & (best_xy[:, 0] <= info["maxx"] + args.buffer)
            & (best_xy[:, 1] >= info["miny"] - args.buffer)
            & (best_xy[:, 1] <= info["maxy"] + args.buffer)
        )
        km_from, km_to = km_from_name(info["name"])
        width = info["maxx"] - info["minx"]
        height = info["maxy"] - info["miny"]
        rows.append(
            {
                "source": info["source"],
                "name": info["name"],
                "gt_points": int(inside.sum()),
                "points": info["points"],
                "km_from": "" if km_from is None else f"{km_from:.3f}",
                "km_to": "" if km_to is None else f"{km_to:.3f}",
                "length_m": f"{max(width, height):.1f}",
                "points_per_m": f"{info['points'] / max(width, height, 1e-9):.0f}",
                "las_version": info["version"],
                "point_format": info["point_format"],
                "crs": info["crs"] or "",
                "minx": f"{info['minx']:.3f}",
                "maxx": f"{info['maxx']:.3f}",
                "miny": f"{info['miny']:.3f}",
                "maxy": f"{info['maxy']:.3f}",
                "minz": f"{info['minz']:.3f}",
                "maxz": f"{info['maxz']:.3f}",
                "wkt": (
                    f"POLYGON(({info['minx']:.3f} {info['miny']:.3f},"
                    f"{info['maxx']:.3f} {info['miny']:.3f},"
                    f"{info['maxx']:.3f} {info['maxy']:.3f},"
                    f"{info['minx']:.3f} {info['maxy']:.3f},"
                    f"{info['minx']:.3f} {info['miny']:.3f}))"
                ),
            }
        )

    def sort_key(z: dict):
        return (float(z["km_from"]) if z["km_from"] else math.inf, z["name"])

    rows.sort(key=sort_key)
    args.index_out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    fp_path = args.index_out / "footprints.csv"
    with open(fp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    matches = [z for z in rows if z["gt_points"] >= args.min_gt_points]
    tr_path = args.index_out / "matches.csv"
    with open(tr_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matches)

    total_gt_points = sum(z["gt_points"] for z in matches)
    print(
        f"\n{len(matches)} of {len(rows)} files contain >= {args.min_gt_points} GT point(s) "
        f"({total_gt_points} assignments in total; a point can match several overlapping boxes)."
    )
    if matches:
        km_values = [float(z["km_from"]) for z in matches if z["km_from"]]
        if km_values:
            print(
                f"kilometre range of the matches: {min(km_values):.3f} - {max(km_values):.3f}"
            )
        densest = sorted(matches, key=lambda z: -z["gt_points"])[:10]
        print("Densest files (start here):")
        for z in densest:
            print(
                f"  {z['gt_points']:4d} GT points  {z['points']:>12,} points  {z['name']}"
            )
    print(f"\nWritten:\n  {fp_path}\n  {tr_path}")
    print(
        "\nCheck in QGIS: Layer > Data Source Manager > Delimited Text, "
        f"file {fp_path.name}, geometry 'Well Known Text (WKT)', field 'wkt', "
        f"CRS = that of the LAS files{f' ({las_crs})' if las_crs else ''}."
    )


# =========================== Subcommand: axis ===========================


def _find_column(header_row: list[str], namen: list[str]) -> int | None:
    lowered = [k.strip().lower().lstrip("#").strip() for k in header_row]
    for name in namen:
        if name in lowered:
            return lowered.index(name)
    return None


def axis_from_csv(path: Path, columns: tuple[int, int] | None) -> np.ndarray:
    """Read x/y from a trajectory CSV. Columns are detected by common names
    (x/easting/rechtswert and y/northing/hochwert); otherwise pass 0-based
    indices via --columns. Order follows the row order, or the time column
    when one is present."""
    with open(path, encoding="utf-8-sig", errors="ignore") as fh:
        sample_text = fh.read(8192)
        fh.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample_text, delimiters=",;\t ").delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.reader(fh, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"{path} ist leer")

    header_row = rows[0]
    ix = iy = it = None
    has_header = any(not _is_number(f) for f in header_row)
    if columns:
        ix, iy = columns
        data = rows[1:] if has_header else rows
    elif has_header:
        ix = _find_column(
            header_row, ["x", "easting", "east", "e", "rechtswert", "x[m]", "x_m"]
        )
        iy = _find_column(
            header_row, ["y", "northing", "north", "n", "hochwert", "y[m]", "y_m"]
        )
        it = _find_column(
            header_row, ["time", "timestamp", "gps_time", "gpstime", "t", "zeit", "sec"]
        )
        if ix is None or iy is None:
            raise SystemExit(
                f"could not detect x/y columns in {path.name}. Header: {header_row}\n"
                f"Please pass --columns <index_x>,<index_y> (0-based)."
            )
        print(
            f"  detected columns: x='{header_row[ix]}', y='{header_row[iy]}'"
            + (f", time='{header_row[it]}'" if it is not None else "")
        )
        data = rows[1:]
    else:
        raise SystemExit(
            f"{path.name} has no header row - please pass --columns <index_x>,<index_y>."
        )

    points = []
    for z in data:
        if len(z) <= max(ix, iy):
            continue
        try:
            x = (
                float(z[ix].replace(",", "."))
                if "," in z[ix] and "." not in z[ix]
                else float(z[ix])
            )
            y = (
                float(z[iy].replace(",", "."))
                if "," in z[iy] and "." not in z[iy]
                else float(z[iy])
            )
        except ValueError:
            continue
        t = None
        if it is not None and len(z) > it:
            try:
                t = float(z[it])
            except ValueError:
                t = None
        points.append((t, x, y))
    if len(points) < 2:
        raise SystemExit(f"too few usable rows in {path}")
    if all(p[0] is not None for p in points):
        points.sort(key=lambda p: p[0])
    return np.array([(p[1], p[2]) for p in points], dtype=np.float64)


def _is_number(s: str) -> bool:
    try:
        float(s.strip().replace(",", "."))
        return True
    except ValueError:
        return False


def sort_along_track(xy: np.ndarray) -> np.ndarray:
    """Put unsorted points (e.g. survey targets) into a plausible order along
    the track: start at the point furthest from the centroid, then always take
    the nearest neighbour not yet used."""
    n = len(xy)
    center = xy.mean(axis=0)
    start = int(np.argmax(np.hypot(*(xy - center).T)))
    free = np.ones(n, dtype=bool)
    order = [start]
    free[start] = False
    current = start
    for _ in range(n - 1):
        d = np.hypot(*(xy - xy[current]).T)
        d[~free] = np.inf
        next_i = int(np.argmin(d))
        order.append(next_i)
        free[next_i] = False
        current = next_i
    return xy[order]


def thin_out(xy: np.ndarray, step: float) -> np.ndarray:
    """Keep only points at least `step` metres from the last kept point -
    guards against standstills and outliers in the trajectory."""
    if step <= 0:
        return xy
    kept = [xy[0]]
    for p in xy[1:]:
        if math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) >= step:
            kept.append(p)
    return np.array(kept, dtype=np.float64)


def cmd_axis(args: argparse.Namespace) -> None:
    if args.from_csv:
        xy = axis_from_csv(args.from_csv, args.columns)
        print(f"{len(xy)} trajectory points read")
    elif args.from_gt:
        xy, crs = load_gt(args.from_gt, args.gt_layer, args.gt_columns)
        xy = remove_outliers(xy, args.gt_outlier_distance)
        xy = sort_along_track(xy)
        print(
            f"{len(xy)} GT points sorted along the track (please check the result in QGIS!)"
        )
    else:
        raise SystemExit("please pass --from-csv or --from-gt.")

    # The axis MUST be in the same CRS as the LAS files, otherwise every point
    # lands in the same bin. The index run tells you which variant fits.
    if args.from_crs and args.to_crs:
        from pyproj import Transformer  # noqa: PLC0415

        tf = Transformer.from_crs(args.from_crs, args.to_crs, always_xy=True)
        x, y = tf.transform(xy[:, 0], xy[:, 1])
        xy = np.column_stack([x, y])
        print(f"  reprojected: {args.from_crs} -> {args.to_crs}")
    if args.x_offset or args.y_offset:
        xy = xy + np.array([args.x_offset, args.y_offset], dtype=np.float64)
        print(f"  shifted by X{args.x_offset:+.1f} / Y{args.y_offset:+.1f}")

    before = len(xy)
    xy = thin_out(xy, args.step)
    d = np.hypot(*np.diff(xy, axis=0).T)
    print(
        f"  thinned: {before} -> {len(xy)} vertices, total length {d.sum():.1f} m, "
        f"largest vertex spacing {d.max():.1f} m"
    )
    if d.max() > 50 * max(args.step, 1.0):
        print(
            "  WARNING: very large gap in the axis - possibly sorted incorrectly, "
            "or several disconnected sections in one file."
        )

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# axis for las_zu_copc_tiles.py (format x;y)\n")
        for x, y in xy:
            fh.write(f"{x:.4f};{y:.4f}\n")
    print(f"Written: {args.out}")


# =========================== Subcommand: process ===========================


def load_chunk_module(path_hint: Path | None):
    candidates = [path_hint] if path_hint else []
    candidates += [
        Path(__file__).resolve().parent / "las_zu_copc_tiles.py",
        Path.cwd() / "las_zu_copc_tiles.py",
    ]
    for k in candidates:
        if k and Path(k).is_file():
            spec = importlib.util.spec_from_file_location("las_zu_copc_tiles", k)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"chunking script loaded: {k}")
            return module
    raise SystemExit("las_zu_copc_tiles.py not found - please pass --chunk-script.")


def fetch_file(
    source: str, target_dir: Path, copy_local: bool = False
) -> tuple[Path, bool]:
    """Make the file available locally. Returns (path, is_temp?).
    Local/UNC paths are normally read in place (no needless copying). With
    copy_local=True the file is pulled onto the local disk first - worth it for
    network shares over VPN, because the tiling script passes over the file
    several times and every pass would otherwise cross the wire again."""
    if not is_url(source):
        if not copy_local:
            return Path(source), False
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(source.replace("\\", "/")).name
        size = os.path.getsize(source)
        print(f"  copying {size / 1e9:.2f} GB -> {target}")
        start = time.monotonic()
        shutil.copyfile(source, target)
        duration = max(time.monotonic() - start, 1e-9)
        print(f"  done in {duration:.0f} s ({size / 1e6 / duration:.0f} MB/s)")
        return target, True
    import urllib.request  # noqa: PLC0415

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.rsplit("/", 1)[-1]
    print(f"  downloading -> {target}")
    start = time.monotonic()
    with urllib.request.urlopen(source, timeout=120) as resp, open(target, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        loaded = 0
        while True:
            block = resp.read(8 << 20)
            if not block:
                break
            fh.write(block)
            loaded += len(block)
            if total and loaded % (256 << 20) < (8 << 20):
                print(f"    {100.0 * loaded / total:5.1f} %", flush=True)
    print(f"  {loaded / 1e9:.2f} GB in {time.monotonic() - start:.0f} s")
    return target, True


def cmd_process(args: argparse.Namespace) -> None:
    with open(args.matches, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.matches} contains no rows.")
    if args.km_from is not None or args.km_to is not None:
        before_n = len(rows)
        rows = [
            z
            for z in rows
            if z.get("km_from")
            and (args.km_from is None or float(z["km_from"]) >= args.km_from)
            and (args.km_to is None or float(z["km_from"]) <= args.km_to)
        ]
        print(f"kilometre filter: {before_n} -> {len(rows)} files")
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} file(s) to process")

    lzc = load_chunk_module(args.chunk_script)
    axis_xy = axis_station = None
    if args.axis_file:
        axis_xy, axis_station = lzc.lade_achse_aus_datei(args.axis_file)
        print(f"axis loaded: {len(axis_xy)} vertices")
    else:
        print(
            "WARNING: no --axis-file - the axis will be estimated per file from the point "
            "cloud, which costs one extra read pass per file."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out).free
    print(f"free space under {args.out}: {free / 1e9:.1f} GB")

    worker = args.worker or os.cpu_count() or 4
    prio = args.start_prio
    for i, row in enumerate(rows, start=1):
        source = row["source"]
        prio_dir = args.out / f"Prio_{prio}"
        if (
            prio_dir.exists()
            and any(prio_dir.glob("segment_*.copc.laz"))
            and not args.overwrite
        ):
            print(
                f"=== [{i}/{len(rows)}] {prio_dir.name} already exists -> skipped "
                f"(--overwrite forces recomputation) ==="
            )
            prio += 1
            continue
        print(
            f"\n=== [{i}/{len(rows)}] {row['name']} -> {prio_dir.name} "
            f"({row.get('gt_points', '?')} GT points, {int(row['points']):,} points) ==="
        )
        if args.dry_run:
            prio += 1
            continue
        path, is_temp = fetch_file(source, args.tmp_dir, args.copy_local)
        try:
            lzc.verarbeite_datei(
                path,
                prio_dir,
                args.edge_length,
                args.a_srs,
                args.tmp_dir,
                worker,
                axis_xy,
                axis_station,
                args.chunk_size,
                args.sample_step,
                args.window_length,
            )
        finally:
            if is_temp and path.exists():
                path.unlink()
                print(f"  temporary LAS deleted: {path.name}")
        prio += 1
    print("\nDone.")


# =========================== CLI ===========================


def _column_pair(text: str) -> tuple[int, int]:
    a, b = text.split(",")
    return int(a), int(b)


def _name_pair(text: str) -> tuple[str, str]:
    a, b = text.split(",")
    return a.strip(), b.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_index = sub.add_parser(
        "index", help="read LAS headers and match them against the GT file"
    )
    p_index.add_argument(
        "--gt", type=Path, required=True, help="GT file (.gpkg or .shp)"
    )
    p_index.add_argument(
        "--gt-layer",
        default=None,
        help="layer name inside the GeoPackage (default: first layer)",
    )
    p_index.add_argument(
        "--gt-columns",
        type=_name_pair,
        default=None,
        help="use two attribute columns instead of the geometry, e.g. 'x,y'. "
        "Useful when the attributes are in a different CRS than the geometry.",
    )
    p_index.add_argument(
        "--las-dir",
        type=Path,
        default=None,
        help="directory, searched recursively",
    )
    p_index.add_argument(
        "--url-list",
        type=Path,
        default=None,
        help="text file with one LAS URL per line",
    )
    p_index.add_argument(
        "--pattern",
        default="*.las",
        help="file pattern (default *.las; *.laz is searched too)",
    )
    p_index.add_argument(
        "--las-crs", default=None, help="force the CRS of the LAS files, e.g. EPSG:5683"
    )
    p_index.add_argument(
        "--buffer",
        type=float,
        default=25.0,
        help="search buffer around each LAS box in metres (default 25)",
    )
    p_index.add_argument(
        "--min-gt-points",
        type=int,
        default=1,
        help="how many GT points make a file a match (default 1)",
    )
    p_index.add_argument(
        "--gt-outlier-distance",
        type=float,
        default=5000.0,
        help="ignore GT points further than this from the median (0 = off, default 5000)",
    )
    p_index.add_argument(
        "--no-crs-detection", action="store_true", help="skip reading VLRs (faster)"
    )
    p_index.add_argument(
        "--index-out",
        type=Path,
        default=Path("./index"),
        help="output directory (default ./index)",
    )
    p_index.set_defaults(func=cmd_index)

    # --- achse ---
    p_axis = sub.add_parser("axis", help="build the axis file for las_zu_copc_tiles.py")
    p_axis.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="trajectory CSV (e.g. trajectory.csv)",
    )
    p_axis.add_argument(
        "--from-gt",
        type=Path,
        default=None,
        help="alternative: use the GT file as the axis source",
    )
    p_axis.add_argument("--gt-layer", default=None)
    p_axis.add_argument("--gt-columns", type=_name_pair, default=None)
    p_axis.add_argument("--gt-outlier-distance", type=float, default=5000.0)
    p_axis.add_argument(
        "--columns",
        type=_column_pair,
        default=None,
        help="0-based column indices 'x,y' for the CSV",
    )
    p_axis.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="minimum vertex spacing in metres (default 1)",
    )
    p_axis.add_argument(
        "--from-crs",
        default=None,
        help="reproject the axis, source CRS (e.g. EPSG:25832)",
    )
    p_axis.add_argument(
        "--to-crs",
        default=None,
        help="reproject the axis, target CRS = CRS of the LAS files",
    )
    p_axis.add_argument(
        "--x-offset",
        type=float,
        default=0.0,
        help="constant offset on X, e.g. 3000000 when the GK zone prefix is missing",
    )
    p_axis.add_argument(
        "--y-offset", type=float, default=0.0, help="constant offset on Y"
    )
    p_axis.add_argument("--out", type=Path, required=True, help="output file")
    p_axis.set_defaults(func=cmd_axis)

    # --- verarbeiten ---
    p_proc = sub.add_parser(
        "process", help="fetch matches, tile them, delete temporary LAS"
    )
    p_proc.add_argument(
        "--matches", type=Path, required=True, help="matches.csv from the index run"
    )
    p_proc.add_argument("--out", type=Path, required=True, help="COPC root directory")
    p_proc.add_argument("--start-prio", type=int, default=1)
    p_proc.add_argument(
        "--edge-length",
        type=float,
        default=25.0,
        help="segment length along the axis in metres (default 25)",
    )
    p_proc.add_argument("--a-srs", default="EPSG:5683")
    p_proc.add_argument("--axis-file", type=Path, default=None)
    p_proc.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help=f"directory for temporary files (default {tempfile.gettempdir()})",
    )
    p_proc.add_argument(
        "--copy-local",
        action="store_true",
        help="copy network-share files into --tmp-dir first and delete them "
        "afterwards. Usually much faster over VPN, because the tiling script "
        "passes over each file several times.",
    )
    p_proc.add_argument("--worker", type=int, default=None)
    p_proc.add_argument("--chunk-size", type=int, default=2_000_000)
    p_proc.add_argument("--sample-step", type=int, default=20)
    p_proc.add_argument("--window-length", type=float, default=20.0)
    p_proc.add_argument(
        "--chunk-script", type=Path, default=None, help="path to las_zu_copc_tiles.py"
    )
    p_proc.add_argument(
        "--limit", type=int, default=None, help="process only the first N matches"
    )
    p_proc.add_argument("--km-from", type=float, default=None)
    p_proc.add_argument("--km-to", type=float, default=None)
    p_proc.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute existing Prio_* folders",
    )
    p_proc.add_argument(
        "--dry-run", action="store_true", help="show what would happen, change nothing"
    )
    p_proc.set_defaults(func=cmd_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
