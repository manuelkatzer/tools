"""Find running rails and track axes in a rail MLS point cloud.

Measures the rail pair spacing from the data instead of assuming 1435 mm.
Detected peaks may sit on the field-side EDGE of the rail head rather than
on the crown centre, which biases the pair spacing by one head width
(1435 + 72 = 1507 mm). The track axis, being the midpoint, is unaffected.

  python track_check.py file.las --cell 0.5 --section 10
  python track_check.py file.las --profile -2.17    # raw lateral profile

Requires: laspy[lazrs], numpy, scipy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import laspy
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

NOMINAL_GAUGE = 1435.0  # mm, standard gauge
HEAD_WIDTH = 72.0  # mm, 60E1 rail head


def load_band(path, cell, band):
    """Two passes: local ground grid, then keep points inside the height band."""
    with laspy.open(str(path)) as f:
        off = np.array(f.header.mins, dtype=np.float64)
        ground, total = {}, 0
        for chunk in f.chunk_iterator(4_000_000):
            x = np.asarray(chunk.x) - off[0]
            y = np.asarray(chunk.y) - off[1]
            z = np.asarray(chunk.z) - off[2]
            key = (x / cell).astype(np.int64) * 100000 + (y / cell).astype(np.int64)
            o = np.argsort(key, kind="stable")
            ks, zs = key[o], z[o]
            cuts = np.flatnonzero(np.diff(ks)) + 1
            for tk, tz in zip(np.split(ks, cuts), np.split(zs, cuts), strict=True):
                k = int(tk[0])
                mz = float(tz.min())
                if k not in ground or mz < ground[k]:
                    ground[k] = mz
            total += len(x)

    kept_xy, kept_h = [], []
    with laspy.open(str(path)) as f:
        for chunk in f.chunk_iterator(4_000_000):
            x = np.asarray(chunk.x) - off[0]
            y = np.asarray(chunk.y) - off[1]
            z = np.asarray(chunk.z) - off[2]
            key = (x / cell).astype(np.int64) * 100000 + (y / cell).astype(np.int64)
            gz = np.array([ground.get(int(k), np.nan) for k in key])
            h = z - gz
            m = np.isfinite(h) & (h > band[0]) & (h < band[1])
            if m.any():
                kept_xy.append(np.c_[x[m], y[m]])
                kept_h.append(h[m])
    if not kept_xy:
        raise SystemExit("No points inside the height band - try a wider --band.")
    return np.vstack(kept_xy), np.concatenate(kept_h), total, len(ground)


def track_direction(xy):
    """Principal direction of the band points, iteratively reweighted.

    The band is dominated by rails (long, thin, parallel), so its principal
    axis estimates the track far better than the tile footprint does.
    """
    p = xy - xy.mean(axis=0)
    for _ in range(3):
        _, _, vt = np.linalg.svd(p, full_matrices=False)
        along = vt[0]
        across = np.array([-along[1], along[0]])
        u = p @ across
        med = np.median(u)
        spread = np.percentile(np.abs(u - med), 75)
        keep = np.abs(u - med) < 4.0 * (spread + 0.5)
        if keep.sum() > 100:
            p = p[keep]
    return along, across


def section_peaks(u_vals, bin_m=0.005):
    """Narrow lateral density peaks after removing the broad baseline.

    A rail head is a narrow spike; ballast shoulder and terrain slope form a
    broad plateau. A rolling median removes the plateau.
    """
    bins = np.arange(u_vals.min(), u_vals.max() + bin_m, bin_m)
    hist, edges = np.histogram(u_vals, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    if hist.size < 80:
        return np.array([]), np.array([])
    base = median_filter(hist.astype(float), size=201, mode="nearest")
    resid = np.convolve(np.clip(hist - base, 0, None), np.ones(3) / 3, mode="same")
    if resid.max() < 5:
        return np.array([]), np.array([])
    pk, _ = find_peaks(resid, prominence=max(resid.max() * 0.30, 8), width=(2, 24), distance=14)
    return mids[pk], resid[pk]


def show_profile(u, h, u0):
    """Print a fine lateral profile so crown and edge can be told apart."""
    m = np.abs(u - u0) < 0.12
    print(f"RAW PROFILE around u={u0:+.3f}  ({m.sum():,} points, 2 mm bins)\n")
    print(f"{'u (m)':>9} {'height':>9} {'n':>7}  density")
    bins = np.arange(u0 - 0.12, u0 + 0.121, 0.002)
    idx = np.digitize(u[m], bins) - 1
    hb = h[m]
    scale = max(1, m.sum() / 600)
    for i in range(len(bins) - 1):
        sel = idx == i
        if not sel.any():
            continue
        top = np.percentile(hb[sel], 95)
        bar = "#" * min(int(sel.sum() / scale), 46)
        print(f"{(bins[i] + bins[i + 1]) / 2:+9.3f} {top:9.3f} {sel.sum():7,}  {bar}")
    print(f"\nCrown  = broad plateau of near-constant height (~{HEAD_WIDTH:.0f} mm wide).")
    print("Edge   = narrow density spike with height dropping off outside it.")
    print(f"Two edges of one rail should sit ~{HEAD_WIDTH:.0f} mm apart.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--cell", type=float, default=0.5, help="ground grid cell size in m (default 0.5)")
    ap.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=(0.10, 0.24),
        metavar=("LOW", "HIGH"),
        help="height band above local ground in m",
    )
    ap.add_argument("--section", type=float, default=10.0, help="along-track section length in m")
    ap.add_argument(
        "--profile", type=float, default=None, help="print a fine raw profile around this lateral offset"
    )
    a = ap.parse_args()

    xy, h, total, cells = load_band(a.file, a.cell, tuple(a.band))
    along, across = track_direction(xy)
    rel = xy - xy.mean(axis=0)
    s = rel @ along
    u = rel @ across

    print("=" * 70)
    print("BAND EXTRACTION")
    print("=" * 70)
    print(f"  points in file:   {total:,}")
    print(f"  ground cells:     {cells:,}  ({a.cell} m)")
    print(
        f"  in band {a.band[0]:.2f}-{a.band[1]:.2f} m above ground: "
        f"{len(xy):,}  ({100 * len(xy) / total:.1f} %)"
    )
    print(f"  track direction:  {np.degrees(np.arctan2(along[1], along[0])) % 180:.2f} deg")
    print(f"  extent along:     {s.max() - s.min():.1f} m")
    print(f"  extent across:    {u.max() - u.min():.1f} m")
    if len(xy) / total > 0.15:
        print("  ! >15 % of points in band - consider a narrower --band")
    print()

    if a.profile is not None:
        show_profile(u, h, a.profile)
        return

    print("=" * 70)
    print("CROSS SECTIONS")
    print("=" * 70)
    found, spacings = [], []
    print(f"{'s (m)':>9} {'points':>10}  peak positions (m)")
    for s0 in np.arange(s.min(), s.max(), a.section):
        sel = (s >= s0) & (s < s0 + a.section)
        if sel.sum() < 500:
            continue
        pos, _ = section_peaks(u[sel])
        if pos.size < 2:
            continue
        pos = np.sort(pos)
        found.append((s0, pos))
        d = np.diff(pos) * 1000
        spacings.extend(d[(d > 1200) & (d < 1700)].tolist())
        print(f"{s0:+9.1f} {sel.sum():10,}  {np.round(pos, 3).tolist()}")

    print()
    print("=" * 70)
    print("MEASURED PAIR SPACING")
    print("=" * 70)
    if not spacings:
        print("  No spacings in the 1200-1700 mm range.")
        print("  Try --band 0.08 0.28, or --cell 0.3 on steep terrain.")
        return
    sp = np.array(spacings)
    hist, edges = np.histogram(sp, bins=np.arange(1200, 1710, 5))
    modal = (edges[:-1] + edges[1:])[np.argmax(hist)] / 2
    print(
        f"  {sp.size} spacings | median {np.median(sp):.0f} mm | modal {modal:.0f} mm | sd {sp.std():.0f} mm"
    )
    print(f"  offset per peak vs rail centreline: {(modal - NOMINAL_GAUGE) / 2:+.0f} mm")
    if abs(modal - NOMINAL_GAUGE) < 25:
        print("  -> peaks on CROWN CENTRE. Gauge directly usable.")
    elif abs(modal - (NOMINAL_GAUGE + HEAD_WIDTH)) < 30:
        print(f"  -> peaks on FIELD-SIDE EDGE (+{HEAD_WIDTH / 2:.0f} mm per rail).")
        print("     Track axis (the midpoint) is unbiased; gauge needs the offset.")
    else:
        print("  -> unexpected. Inspect one rail with --profile.")

    print()
    print("=" * 70)
    print("TRACK AXES")
    print("=" * 70)
    axes = []
    for s0, pos in found:
        for i in range(len(pos) - 1):
            if abs((pos[i + 1] - pos[i]) * 1000 - modal) < 40:
                axes.append((s0, (pos[i] + pos[i + 1]) / 2))
    if not axes:
        print("  none")
        return
    srt = np.sort(np.array([x[1] for x in axes]))
    groups = [[float(srt[0])]]
    for v in srt[1:]:
        if float(v) - groups[-1][-1] < 1.2:
            groups[-1].append(float(v))
        else:
            groups.append([float(v)])
    centres = [float(np.median(g)) for g in groups]
    print(f"  {len(axes)} axis hits across {len(centres)} track(s)")
    for c, g in zip(centres, groups, strict=True):
        print(f"    u = {c:+7.3f} m   {len(g)} sections, sd {np.std(g) * 1000:.0f} mm")
    if len(centres) > 1:
        print(f"  track spacings: {np.round(np.diff(centres), 3).tolist()} m")


if __name__ == "__main__":
    main()
