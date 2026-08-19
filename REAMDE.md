# las-gt-pipeline

Find out which LAS/LAZ files contain ground-truth survey targets, then tile only
those into COPC.

A LAS file reveals its bounding box in the first 375 bytes of the public header
block, so identifying the relevant files costs a few hundred bytes each rather
than gigabytes. This matters when the point clouds live on a network share
reached over VPN.

## Setup

```bash
uv sync --extra crs                    # index + axis
uv sync --extra crs --extra process    # + tiling (laspy, scipy)
sudo pacman -S pdal                    # PDAL CLI, not a Python package
```

## Mounting the data share

```bash
sudo mkdir -p /mnt/befahrungen
sudo mount -t cifs //192.168.50.10/Befahrungen /mnt/befahrungen \
    -o credentials=$HOME/.smbcred,uid=$(id -u),gid=$(id -g),vers=3.0,ro
```

## Workflow

### 1. index — which files match the ground truth?

```bash
uv run las_gt_pipeline.py index \
    --gt ~/GT/Target_Kontrolle.gpkg \
    --las-dir /mnt/befahrungen/24067_GSH_Hagen_Hamm/01_Punktwolken_Zwangspunkte/2103 \
    --las-crs EPSG:5683 \
    --gt-outlier-distance 0 \
    --index-out ./index_2103
```

Writes `footprints.csv` (all files, as WKT rectangles loadable in QGIS via
Delimited Text) and `matches.csv` (the hits, input for stage 3).

The CRS variant table is the important output. If the best variant scores zero,
stop — the coordinate systems do not line up yet and tiling would be wasted work.
`--las-crs` is needed whenever the LAS files carry no CRS in their VLRs; without
it, no real reprojection is attempted at all.

`--gt-outlier-distance 0` disables the median filter. Leave it on (default 5000)
only if the GT file contains dummy or zero rows; turn it off when the file
legitimately spans several routes.

### 2. axis — build the track axis

```bash
uv run las_gt_pipeline.py axis \
    --from-csv /mnt/.../2103/trajectory.csv \
    --columns 13,14 \
    --out axis_2103.txt
```

Column names are auto-detected for common spellings (`x`, `easting`, `northing`,
...). Names like `projectedX[m]` are not recognised, so pass 0-based indices with
`--columns`.

The axis must be in the same CRS as the LAS files, otherwise every point lands in
one segment and nothing errors out. Verify with `head -3 axis_2103.txt` that the
coordinates match the LAS extent printed by the index run. Add
`--from-crs EPSG:25832 --to-crs EPSG:5683` if reprojection is needed.

### 3. process — tile the hits into COPC

```bash
uv run las_gt_pipeline.py process \
    --matches index_2103/matches.csv \
    --out ~/copc_2103 \
    --axis-file axis_2103.txt \
    --edge-length 25 --limit 2 --dry-run
```

Drop `--dry-run` and add `--copy-local --tmp-dir ~/tmp` to run it for real.
`--copy-local` pulls each file to local disk first, which is much faster over VPN
because the tiling script passes over every file several times.

Keep `--tmp-dir` and `--out` on the local filesystem, never on the share.

## Notes

- `las_zu_copc_tiles.py` is loaded dynamically at runtime by `process`. The
  pipeline calls `load_axis_from_file()` and `process_file()` on it; if you
  rename either, rename it in both files together.
- `las_zu_copc_tiles.py` also works standalone:
  `uv run las_zu_copc_tiles.py --input a.las --out ./COPC --axis-file axis.txt`
- LAS filenames containing commas (`180,1_180,2`) are quoted in the output CSVs.
  Parse them with a real CSV reader, not `cut -d,`.
