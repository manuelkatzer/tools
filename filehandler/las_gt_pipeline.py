"""Findet heraus, welche LAS-Dateien zu einer Ground-Truth-Datei (Zielmarken,
.shp/.gpkg) gehoeren, und verarbeitet nur diese zu COPC-Kacheln.

Kerngedanke: eine LAS-Datei verraet ihre Bounding-Box in den ersten 375 Bytes
(Public Header Block, Offset 179). Man muss also KEINE Gigabyte herunterladen,
um zu wissen, wo eine Datei liegt - ein paar hundert Bytes je Datei reichen,
lokal, ueber Netzlaufwerk oder per HTTP-Range-Request.

Drei Unterbefehle:

  index        Liest nur die Header aller LAS/LAZ-Dateien, vergleicht sie mit
               der GT-Datei und schreibt
                 - footprints.csv  (WKT-Rechtecke, in QGIS als "Delimited Text"
                   ladbar -> direkt neben die GT legen und anschauen)
                 - treffer.csv     (die passenden Dateien, Eingabe fuer Stufe 3)
               Dabei wird auch versucht, das CRS-Verhaeltnis GT <-> LAS
               automatisch zu bestimmen (haeufigste Fehlerquelle).

  achse        Baut aus trajectory.csv (oder aus der GT-Datei) eine
               Achsen-Datei im Format, das las_zu_copc_tiles.py erwartet.

  verarbeiten  Geht treffer.csv durch: Datei bereitstellen (lokal direkt lesen,
               per HTTP in --tmp-verzeichnis laden), mit las_zu_copc_tiles.py
               in COPC-Kacheln zerlegen, temporaere LAS wieder loeschen.

Abhaengigkeiten:
  index/achse   nur numpy. Optional: geopandas oder fiona (sonst greifen
                eingebaute Minimal-Leser fuer .shp und .gpkg), pyproj
                (fuer echte CRS-Transformationen).
  verarbeiten   zusaetzlich alles, was las_zu_copc_tiles.py braucht
                (laspy, scipy, pdal-CLI).

Beispiele:

    # 1. Welche Dateien passen ueberhaupt?
    python las_gt_pipeline.py index \\
        --gt target_kontrolle.gpkg \\
        --las-verzeichnis "//server/Befahrungen/24067_GSH_Hagen_Hamm" \\
        --ausgabe-index ./index

    # 2. Achse aus der Trajektorie bauen
    python las_gt_pipeline.py achse \\
        --aus-csv .../2103/trajectory.csv --ziel achse_2103.txt

    # 3. Nur die ersten 10 Treffer verarbeiten (klein anfangen!)
    python las_gt_pipeline.py verarbeiten \\
        --treffer ./index/treffer.csv \\
        --ausgabe /pfad/zu/QGIS/COPC \\
        --achsen-datei achse_2103.txt \\
        --kantenlaenge 25 --limit 10
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
import time
from pathlib import Path

import numpy as np

LAS_HEADER_BYTES = 375  # reicht fuer LAS 1.0 - 1.4
VLR_MAX_BYTES = 65_536  # so viel lesen wir hoechstens fuer die CRS-Erkennung


# =========================== Quellen (lokal / HTTP) ===========================


def ist_url(quelle: str) -> bool:
    return quelle.startswith("http://") or quelle.startswith("https://")


def lies_kopf_bytes(quelle: str, n: int) -> bytes:
    """Liest die ersten n Bytes einer Datei - lokal/UNC direkt, per HTTP als
    Range-Request. Server ohne Range-Unterstuetzung liefern zwar die ganze
    Datei, wir lesen aber trotzdem nur n Bytes und brechen dann ab."""
    if ist_url(quelle):
        import urllib.request

        req = urllib.request.Request(quelle, headers={"Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read(n)
    with open(quelle, "rb") as fh:
        return fh.read(n)


def sammle_quellen(
    verzeichnis: Path | None, url_liste: Path | None, muster: str
) -> list[str]:
    quellen: list[str] = []
    if verzeichnis:
        for pfad in sorted(verzeichnis.rglob(muster)):
            if pfad.is_file():
                quellen.append(str(pfad))
        # zweite Endung mitnehmen, falls Standardmuster
        if muster == "*.las":
            for pfad in sorted(verzeichnis.rglob("*.laz")):
                if pfad.is_file():
                    quellen.append(str(pfad))
    if url_liste:
        with open(url_liste, encoding="utf-8-sig") as fh:
            for zeile in fh:
                zeile = zeile.strip()
                if zeile and not zeile.startswith("#"):
                    quellen.append(zeile)
    return quellen


# =========================== LAS-Header lesen ===========================


def parse_las_header(roh: bytes) -> dict:
    """Zerlegt den Public Header Block. Funktioniert fuer LAS 1.0-1.4 und
    genauso fuer LAZ, weil der Header dort unkomprimiert vorne steht."""
    if len(roh) < 227 or roh[:4] != b"LASF":
        raise ValueError("keine gueltige LAS/LAZ-Datei (Signatur 'LASF' fehlt)")
    version = (roh[24], roh[25])
    header_size = struct.unpack_from("<H", roh, 94)[0]
    offset_punkte = struct.unpack_from("<I", roh, 96)[0]
    anzahl_vlr = struct.unpack_from("<I", roh, 100)[0]
    punktformat = roh[104] & 0b0011_1111  # oberste Bits = LAZ-Kompressionsflag
    punktlaenge = struct.unpack_from("<H", roh, 105)[0]
    anzahl = struct.unpack_from("<I", roh, 107)[0]
    scales = struct.unpack_from("<3d", roh, 131)
    offsets = struct.unpack_from("<3d", roh, 155)
    maxx, minx, maxy, miny, maxz, minz = struct.unpack_from("<6d", roh, 179)
    if version >= (1, 4) and len(roh) >= 255:
        anzahl_14 = struct.unpack_from("<Q", roh, 247)[0]
        if anzahl_14:
            anzahl = anzahl_14
    return {
        "version": f"{version[0]}.{version[1]}",
        "header_size": header_size,
        "offset_punkte": offset_punkte,
        "anzahl_vlr": anzahl_vlr,
        "punktformat": punktformat,
        "punktlaenge": punktlaenge,
        "punkte": int(anzahl),
        "scales": scales,
        "offsets": offsets,
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy,
        "minz": minz,
        "maxz": maxz,
    }


def epsg_aus_vlrs(roh: bytes, header_size: int, anzahl_vlr: int) -> str | None:
    """Sucht in den VLRs nach dem CRS: erst WKT (record_id 2112), sonst die
    GeoTIFF-Keys (record_id 34735, Key 3072 = ProjectedCSTypeGeoKey)."""
    pos = header_size
    for _ in range(anzahl_vlr):
        if pos + 54 > len(roh):
            return None
        user_id = roh[pos + 2 : pos + 18].rstrip(b"\x00").decode("latin-1", "ignore")
        record_id = struct.unpack_from("<H", roh, pos + 18)[0]
        laenge = struct.unpack_from("<H", roh, pos + 20)[0]
        daten = roh[pos + 54 : pos + 54 + laenge]
        pos += 54 + laenge
        if "LASF_Projection" not in user_id:
            continue
        if record_id == 2112 and daten:  # WKT
            wkt = daten.rstrip(b"\x00").decode("latin-1", "ignore")
            treffer = re.findall(
                r'(?:AUTHORITY|ID)\s*\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]', wkt
            )
            if treffer:
                return f"EPSG:{treffer[-1]}"
            return "WKT ohne EPSG-Code"
        if record_id == 34735 and len(daten) >= 8:  # GeoKeyDirectory
            _, _, _, n_keys = struct.unpack_from("<4H", daten, 0)
            for i in range(n_keys):
                off = 8 + i * 8
                if off + 8 > len(daten):
                    break
                key_id, tiff_tag, _count, wert = struct.unpack_from("<4H", daten, off)
                if key_id == 3072 and tiff_tag == 0 and wert not in (0, 32767):
                    return f"EPSG:{wert}"
    return None


def lies_las_info(quelle: str, mit_crs: bool = True) -> dict:
    roh = lies_kopf_bytes(quelle, LAS_HEADER_BYTES)
    info = parse_las_header(roh)
    info["quelle"] = quelle
    info["name"] = quelle.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    info["crs"] = None
    if mit_crs and info["anzahl_vlr"]:
        try:
            n = min(info["offset_punkte"], VLR_MAX_BYTES)
            if n > len(roh):  # VLRs reichen ueber den ersten Lesevorgang hinaus
                roh = lies_kopf_bytes(quelle, n)
            info["crs"] = epsg_aus_vlrs(roh, info["header_size"], info["anzahl_vlr"])
        except Exception:  # noqa: BLE001 - CRS ist nur Zusatzinfo
            pass
    return info


# =========================== Dateiname -> Kilometer ===========================

KM_MUSTER = re.compile(r"[_\-](\d{1,4}[.,]\d+)[_\-](\d{1,4}[.,]\d+)[_\-]")


def km_aus_name(name: str) -> tuple[float | None, float | None]:
    """Erkennt Kilometerangaben wie '..._180,0_180,1_...' im Dateinamen."""
    m = KM_MUSTER.search(name)
    if not m:
        return None, None
    return float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))


# =========================== GT-Datei laden ===========================


def _lade_gt_geopandas(pfad: Path, spalten: tuple[str, str] | None):
    import geopandas as gpd  # noqa: PLC0415

    g = gpd.read_file(pfad)
    crs = g.crs.to_string() if g.crs is not None else None
    if spalten:
        xy = np.column_stack(
            [g[spalten[0]].to_numpy(float), g[spalten[1]].to_numpy(float)]
        )
        return xy, None  # CRS der Attribute ist unbekannt
    geo = g.geometry.representative_point()
    return np.column_stack([geo.x.to_numpy(), geo.y.to_numpy()]), crs


def _lade_gt_fiona(pfad: Path, spalten: tuple[str, str] | None):
    import fiona  # noqa: PLC0415

    punkte = []
    with fiona.open(pfad) as src:
        crs = src.crs_wkt or (str(src.crs) if src.crs else None)
        for f in src:
            if spalten:
                p = f["properties"]
                punkte.append((float(p[spalten[0]]), float(p[spalten[1]])))
            else:
                koord = f["geometry"]["coordinates"]
                while isinstance(koord[0], (list, tuple)):
                    koord = koord[0]
                punkte.append((float(koord[0]), float(koord[1])))
    return np.array(punkte, dtype=np.float64), (None if spalten else crs)


def _lade_gt_shp(pfad: Path):
    """Minimal-Leser fuer Punkt-Shapefiles (Typ 1/11/21), ohne Fremdpakete."""
    daten = pfad.read_bytes()
    if struct.unpack_from(">i", daten, 0)[0] != 9994:
        raise ValueError(f"{pfad} ist kein Shapefile")
    punkte = []
    pos = 100
    while pos + 8 <= len(daten):
        _nr, laenge_worte = struct.unpack_from(">2i", daten, pos)
        inhalt = pos + 8
        typ = struct.unpack_from("<i", daten, inhalt)[0]
        if typ in (1, 11, 21):
            x, y = struct.unpack_from("<2d", daten, inhalt + 4)
            punkte.append((x, y))
        pos = inhalt + laenge_worte * 2
    crs = None
    prj = pfad.with_suffix(".prj")
    if prj.is_file():
        wkt = prj.read_text(encoding="utf-8-sig", errors="ignore")
        m = re.findall(r'(?:AUTHORITY|ID)\s*\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]', wkt)
        crs = f"EPSG:{m[-1]}" if m else wkt[:120]
    return np.array(punkte, dtype=np.float64), crs


def _gpkg_blob_zu_xy(blob: bytes) -> tuple[float, float] | None:
    """Zerlegt einen GeoPackage-Geometrie-Blob (Punkt) ohne Fremdpakete."""
    if len(blob) < 8 or blob[:2] != b"GP":
        return None
    flags = blob[3]
    huellen_typ = (flags >> 1) & 0b111
    huellen_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(huellen_typ)
    if huellen_bytes is None or (flags >> 4) & 1:  # unbekannt oder leere Geometrie
        return None
    pos = 8 + huellen_bytes
    if pos + 21 > len(blob):
        return None
    little = blob[pos] == 1
    fmt = "<" if little else ">"
    typ = struct.unpack_from(fmt + "I", blob, pos + 1)[0] % 1000
    if typ != 1:  # nur Punkte
        return None
    x, y = struct.unpack_from(fmt + "2d", blob, pos + 5)
    return x, y


def _lade_gt_gpkg(pfad: Path, layer: str | None, spalten: tuple[str, str] | None):
    """Minimal-Leser fuer Punkt-GeoPackages ueber sqlite3."""
    import sqlite3  # noqa: PLC0415

    con = sqlite3.connect(f"file:{pfad}?mode=ro", uri=True)
    try:
        zeilen = con.execute(
            "SELECT table_name, srs_id FROM gpkg_contents WHERE data_type='features'"
        ).fetchall()
        if not zeilen:
            raise ValueError("keine Vektor-Layer im GeoPackage gefunden")
        tabelle, srs_id = zeilen[0]
        if layer:
            for t, s in zeilen:
                if t == layer:
                    tabelle, srs_id = t, s
                    break
        elif len(zeilen) > 1:
            print(
                f"  Hinweis: {len(zeilen)} Layer im GPKG, verwende '{tabelle}' "
                f"(andere: {', '.join(t for t, _ in zeilen[1:])})"
            )
        if spalten:
            rows = con.execute(
                f'SELECT "{spalten[0]}","{spalten[1]}" FROM "{tabelle}"'
            ).fetchall()
            xy = np.array(
                [(float(a), float(b)) for a, b in rows if a is not None],
                dtype=np.float64,
            )
            return xy, None
        geom_spalte = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
            (tabelle,),
        ).fetchone()[0]
        punkte = []
        for (blob,) in con.execute(f'SELECT "{geom_spalte}" FROM "{tabelle}"'):
            if blob:
                xy = _gpkg_blob_zu_xy(blob)
                if xy:
                    punkte.append(xy)
        return np.array(punkte, dtype=np.float64), (
            f"EPSG:{srs_id}" if srs_id else None
        )
    finally:
        con.close()


def lade_gt(
    pfad: Path, layer: str | None, spalten: tuple[str, str] | None
) -> tuple[np.ndarray, str | None]:
    fehler = []
    for name, funktion in (
        ("geopandas", lambda: _lade_gt_geopandas(pfad, spalten)),
        ("fiona", lambda: _lade_gt_fiona(pfad, spalten)),
    ):
        try:
            xy, crs = funktion()
            print(f"GT gelesen mit {name}: {len(xy)} Punkte, CRS laut Datei: {crs}")
            return xy, crs
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001
            fehler.append(f"{name}: {exc}")
    suffix = pfad.suffix.lower()
    try:
        if suffix == ".gpkg":
            xy, crs = _lade_gt_gpkg(pfad, layer, spalten)
        elif suffix == ".shp":
            xy, crs = _lade_gt_shp(pfad)
        else:
            raise ValueError(f"Format {suffix} ohne geopandas/fiona nicht lesbar")
        print(
            f"GT gelesen mit eingebautem Leser: {len(xy)} Punkte, CRS laut Datei: {crs}"
        )
        return xy, crs
    except Exception as exc:  # noqa: BLE001
        fehler.append(f"eingebaut: {exc}")
    raise SystemExit("GT-Datei konnte nicht gelesen werden:\n  " + "\n  ".join(fehler))


def entferne_ausreisser(xy: np.ndarray, max_abstand: float) -> np.ndarray:
    """Wirft Punkte weg, die weit vom Median der Wolke entfernt liegen -
    typischerweise Dummy-/Nullzeilen, die die Bounding-Box aufblaehen."""
    if max_abstand <= 0 or len(xy) < 3:
        return xy
    mitte = np.median(xy, axis=0)
    d = np.hypot(*(xy - mitte).T)
    behalten = d <= max_abstand
    if not behalten.all():
        print(
            f"  {int((~behalten).sum())} GT-Punkt(e) weiter als {max_abstand:.0f} m vom Median "
            f"entfernt -> ignoriert (max. Abstand war {d.max():.0f} m)"
        )
    return xy[behalten]


# =========================== CRS-Kandidaten ===========================

CRS_KANDIDATEN = [
    "EPSG:25832",
    "EPSG:25833",
    "EPSG:5683",
    "EPSG:5684",
    "EPSG:31467",
    "EPSG:31468",
]


def erzeuge_kandidaten(
    gt_xy: np.ndarray, gt_crs: str | None, las_crs: str | None
) -> list[tuple[str, np.ndarray]]:
    """Baut Varianten der GT-Koordinaten, von denen eine zum LAS-CRS passen
    sollte. Enthaelt bewusst auch die 'Zonenkennziffer fehlt'-Faelle, die bei
    Gauss-Krueger-Exporten staendig vorkommen (3.392.674 vs. 392.674)."""
    kandidaten: list[tuple[str, np.ndarray]] = [("GT unveraendert", gt_xy)]
    if gt_xy[:, 0].max() < 1_000_000:
        for delta, name in (
            (3_000_000, "X + 3.000.000 (GK-Zone 3 ergaenzt)"),
            (4_000_000, "X + 4.000.000 (GK-Zone 4 ergaenzt)"),
        ):
            kandidaten.append((name, gt_xy + np.array([delta, 0.0])))
    else:
        kandidaten.append(
            ("X - 3.000.000 (GK-Zone entfernt)", gt_xy - np.array([3_000_000.0, 0.0]))
        )
        kandidaten.append(
            ("X - 4.000.000 (GK-Zone entfernt)", gt_xy - np.array([4_000_000.0, 0.0]))
        )
    if not las_crs or not las_crs.upper().startswith("EPSG:"):
        return kandidaten
    try:
        from pyproj import Transformer  # noqa: PLC0415
    except ImportError:
        print(
            "  Hinweis: pyproj nicht installiert - echte Umprojektionen werden nicht getestet."
        )
        return kandidaten
    quellen = []
    if gt_crs and gt_crs.upper().startswith("EPSG:"):
        quellen.append(gt_crs.upper())
    quellen += [c for c in CRS_KANDIDATEN if c not in quellen]
    for quelle in quellen:
        if quelle.upper() == las_crs.upper():
            continue
        try:
            tf = Transformer.from_crs(quelle, las_crs, always_xy=True)
            x, y = tf.transform(gt_xy[:, 0], gt_xy[:, 1])
            neu = np.column_stack([x, y])
            if np.isfinite(neu).all():
                kandidaten.append((f"GT als {quelle} -> {las_crs}", neu))
        except Exception:  # noqa: BLE001
            continue
    return kandidaten


def zaehle_in_boxen(xy: np.ndarray, infos: list[dict], puffer: float) -> np.ndarray:
    """Wie viele GT-Punkte liegen in mindestens einer LAS-Bounding-Box?"""
    treffer = np.zeros(len(xy), dtype=bool)
    for info in infos:
        treffer |= (
            (xy[:, 0] >= info["minx"] - puffer)
            & (xy[:, 0] <= info["maxx"] + puffer)
            & (xy[:, 1] >= info["miny"] - puffer)
            & (xy[:, 1] <= info["maxy"] + puffer)
        )
    return treffer


# =========================== Unterbefehl: index ===========================


def cmd_index(args: argparse.Namespace) -> None:
    quellen = sammle_quellen(args.las_verzeichnis, args.url_liste, args.muster)
    if not quellen:
        raise SystemExit(
            "Keine LAS/LAZ-Dateien gefunden - --las-verzeichnis / --url-liste pruefen."
        )
    print(
        f"{len(quellen)} LAS/LAZ-Quellen gefunden, lese Header (nur ~{LAS_HEADER_BYTES} Bytes je Datei)..."
    )

    infos: list[dict] = []
    fehlerhaft: list[tuple[str, str]] = []
    start = time.monotonic()
    for i, quelle in enumerate(quellen, start=1):
        try:
            infos.append(lies_las_info(quelle, mit_crs=not args.ohne_crs_erkennung))
        except Exception as exc:  # noqa: BLE001
            fehlerhaft.append((quelle, str(exc)))
        if i % 25 == 0 or i == len(quellen):
            print(
                f"  ... {i}/{len(quellen)} ({time.monotonic() - start:.0f} s)",
                flush=True,
            )
    if fehlerhaft:
        print(
            f"  {len(fehlerhaft)} Datei(en) nicht lesbar, z.B. {fehlerhaft[0][0]}: {fehlerhaft[0][1]}"
        )
    if not infos:
        raise SystemExit("Kein einziger Header lesbar.")

    crs_werte = {i["crs"] for i in infos if i["crs"]}
    las_crs = None
    if crs_werte:
        las_crs = sorted(crs_werte)[0]
        print(f"CRS laut LAS-VLRs: {', '.join(sorted(crs_werte))}")
    else:
        print("CRS laut LAS-VLRs: nicht gesetzt (die Dateien tragen kein CRS!)")
    if args.las_crs:
        las_crs = args.las_crs
        print(f"CRS der LAS-Dateien laut --las-crs auf {las_crs} gesetzt")

    gt_xy, gt_crs = lade_gt(args.gt, args.gt_layer, args.gt_spalten)
    gt_xy = entferne_ausreisser(gt_xy, args.gt_ausreisser_abstand)
    print(
        f"GT-Ausdehnung: X {gt_xy[:, 0].min():.1f}..{gt_xy[:, 0].max():.1f}, "
        f"Y {gt_xy[:, 1].min():.1f}..{gt_xy[:, 1].max():.1f}"
    )
    gesamt = (
        min(i["minx"] for i in infos),
        max(i["maxx"] for i in infos),
        min(i["miny"] for i in infos),
        max(i["maxy"] for i in infos),
    )
    print(
        f"LAS-Ausdehnung: X {gesamt[0]:.1f}..{gesamt[1]:.1f}, Y {gesamt[2]:.1f}..{gesamt[3]:.1f}"
    )

    print(
        "\nCRS-Varianten (Treffer = GT-Punkte, die in mindestens einer LAS-Box liegen):"
    )
    bewertung = []
    for name, xy in erzeuge_kandidaten(gt_xy, gt_crs, las_crs):
        n = int(zaehle_in_boxen(xy, infos, args.puffer).sum())
        bewertung.append((n, name, xy))
        print(f"  {n:6d} / {len(gt_xy)}   {name}")
    bewertung.sort(key=lambda t: -t[0])
    n_best, name_best, xy_best = bewertung[0]
    if n_best == 0:
        print(
            "\nKEINE Variante trifft. Moegliche Ursachen: GT und LAS decken wirklich "
            "verschiedene Abschnitte ab, das CRS ist ein anderes als die getesteten "
            "(--las-crs setzen, pyproj installieren), oder --puffer erhoehen."
        )
    else:
        print(f"\nBeste Variante: {name_best} ({n_best} von {len(gt_xy)} GT-Punkten)")

    # je Datei auswerten
    zeilen = []
    for info in infos:
        drin = (
            (xy_best[:, 0] >= info["minx"] - args.puffer)
            & (xy_best[:, 0] <= info["maxx"] + args.puffer)
            & (xy_best[:, 1] >= info["miny"] - args.puffer)
            & (xy_best[:, 1] <= info["maxy"] + args.puffer)
        )
        km_von, km_bis = km_aus_name(info["name"])
        breite = info["maxx"] - info["minx"]
        hoehe = info["maxy"] - info["miny"]
        zeilen.append(
            {
                "quelle": info["quelle"],
                "name": info["name"],
                "gt_punkte": int(drin.sum()),
                "punkte": info["punkte"],
                "km_von": "" if km_von is None else f"{km_von:.3f}",
                "km_bis": "" if km_bis is None else f"{km_bis:.3f}",
                "laenge_m": f"{max(breite, hoehe):.1f}",
                "punkte_pro_m": f"{info['punkte'] / max(breite, hoehe, 1e-9):.0f}",
                "las_version": info["version"],
                "punktformat": info["punktformat"],
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

    def sortier_schluessel(z: dict):
        return (float(z["km_von"]) if z["km_von"] else math.inf, z["name"])

    zeilen.sort(key=sortier_schluessel)
    args.ausgabe_index.mkdir(parents=True, exist_ok=True)
    felder = list(zeilen[0].keys())
    fp_pfad = args.ausgabe_index / "footprints.csv"
    with open(fp_pfad, "w", newline="", encoding="utf-8") as fh:
        schreiber = csv.DictWriter(fh, fieldnames=felder)
        schreiber.writeheader()
        schreiber.writerows(zeilen)

    treffer = [z for z in zeilen if z["gt_punkte"] >= args.min_gt_punkte]
    tr_pfad = args.ausgabe_index / "treffer.csv"
    with open(tr_pfad, "w", newline="", encoding="utf-8") as fh:
        schreiber = csv.DictWriter(fh, fieldnames=felder)
        schreiber.writeheader()
        schreiber.writerows(treffer)

    summe_punkte = sum(z["gt_punkte"] for z in treffer)
    print(
        f"\n{len(treffer)} von {len(zeilen)} Dateien enthalten >= {args.min_gt_punkte} GT-Punkt(e) "
        f"(zusammen {summe_punkte} Zuordnungen; Mehrfachtreffer moeglich, wenn sich Boxen ueberlappen)."
    )
    if treffer:
        km_werte = [float(z["km_von"]) for z in treffer if z["km_von"]]
        if km_werte:
            print(
                f"Kilometerbereich der Treffer: {min(km_werte):.3f} - {max(km_werte):.3f}"
            )
        beste = sorted(treffer, key=lambda z: -z["gt_punkte"])[:10]
        print("Dichteste Dateien (dort zuerst anfangen):")
        for z in beste:
            print(
                f"  {z['gt_punkte']:4d} GT-Punkte  {z['punkte']:>12,} Punkte  {z['name']}"
            )
    print(f"\nGeschrieben:\n  {fp_pfad}\n  {tr_pfad}")
    print(
        "\nIn QGIS pruefen: Layer > Datenquellen-Verwaltung > Getrennter Text, "
        f"Datei {fp_pfad.name}, Geometrie 'Well Known Text (WKT)', Feld 'wkt', "
        f"CRS = das der LAS-Dateien{f' ({las_crs})' if las_crs else ''}."
    )


# =========================== Unterbefehl: achse ===========================


def _spalte_finden(kopf: list[str], namen: list[str]) -> int | None:
    klein = [k.strip().lower().lstrip("#").strip() for k in kopf]
    for name in namen:
        if name in klein:
            return klein.index(name)
    return None


def achse_aus_csv(pfad: Path, spalten: tuple[int, int] | None) -> np.ndarray:
    """Liest x/y aus einer Trajektorien-CSV. Spalten werden ueber gaengige
    Namen erkannt (x/easting/rechtswert bzw. y/northing/hochwert), sonst per
    --spalten als 0-basierte Indizes angeben. Reihenfolge = Zeilenreihenfolge
    bzw. nach Zeitspalte sortiert."""
    with open(pfad, encoding="utf-8-sig", errors="ignore") as fh:
        probe = fh.read(8192)
        fh.seek(0)
        try:
            trenner = csv.Sniffer().sniff(probe, delimiters=",;\t ").delimiter
        except csv.Error:
            trenner = ","
        leser = csv.reader(fh, delimiter=trenner)
        zeilen = list(leser)
    if not zeilen:
        raise SystemExit(f"{pfad} ist leer")

    kopf = zeilen[0]
    ix = iy = it = None
    hat_kopf = any(not _ist_zahl(f) for f in kopf)
    if spalten:
        ix, iy = spalten
        daten = zeilen[1:] if hat_kopf else zeilen
    elif hat_kopf:
        ix = _spalte_finden(
            kopf, ["x", "easting", "east", "e", "rechtswert", "x[m]", "x_m"]
        )
        iy = _spalte_finden(
            kopf, ["y", "northing", "north", "n", "hochwert", "y[m]", "y_m"]
        )
        it = _spalte_finden(
            kopf, ["time", "timestamp", "gps_time", "gpstime", "t", "zeit", "sec"]
        )
        if ix is None or iy is None:
            raise SystemExit(
                f"x/y-Spalten in {pfad.name} nicht erkannt. Kopfzeile: {kopf}\n"
                f"Bitte --spalten <index_x>,<index_y> angeben (0-basiert)."
            )
        print(
            f"  Spalten erkannt: x='{kopf[ix]}', y='{kopf[iy]}'"
            + (f", Zeit='{kopf[it]}'" if it is not None else "")
        )
        daten = zeilen[1:]
    else:
        raise SystemExit(
            f"{pfad.name} hat keine Kopfzeile - bitte --spalten <index_x>,<index_y> angeben."
        )

    punkte = []
    for z in daten:
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
        punkte.append((t, x, y))
    if len(punkte) < 2:
        raise SystemExit(f"Zu wenige verwertbare Zeilen in {pfad}")
    if all(p[0] is not None for p in punkte):
        punkte.sort(key=lambda p: p[0])
    return np.array([(p[1], p[2]) for p in punkte], dtype=np.float64)


def _ist_zahl(s: str) -> bool:
    try:
        float(s.strip().replace(",", "."))
        return True
    except ValueError:
        return False


def sortiere_entlang_strecke(xy: np.ndarray) -> np.ndarray:
    """Bringt unsortierte Punkte (z.B. Zielmarken) in eine plausible
    Reihenfolge entlang der Strecke: Start am Punkt mit dem groessten Abstand
    zum Schwerpunkt, danach immer der naechste noch freie Nachbar."""
    n = len(xy)
    mitte = xy.mean(axis=0)
    start = int(np.argmax(np.hypot(*(xy - mitte).T)))
    frei = np.ones(n, dtype=bool)
    reihenfolge = [start]
    frei[start] = False
    aktuell = start
    for _ in range(n - 1):
        d = np.hypot(*(xy - xy[aktuell]).T)
        d[~frei] = np.inf
        naechster = int(np.argmin(d))
        reihenfolge.append(naechster)
        frei[naechster] = False
        aktuell = naechster
    return xy[reihenfolge]


def ausduennen(xy: np.ndarray, schritt: float) -> np.ndarray:
    """Behaelt nur Punkte mit mindestens `schritt` Metern Abstand zum letzten
    behaltenen Punkt - gegen Standzeiten und Ausreisser in der Trajektorie."""
    if schritt <= 0:
        return xy
    behalten = [xy[0]]
    for p in xy[1:]:
        if math.hypot(p[0] - behalten[-1][0], p[1] - behalten[-1][1]) >= schritt:
            behalten.append(p)
    return np.array(behalten, dtype=np.float64)


def cmd_achse(args: argparse.Namespace) -> None:
    if args.aus_csv:
        xy = achse_aus_csv(args.aus_csv, args.spalten)
        print(f"{len(xy)} Trajektorienpunkte gelesen")
    elif args.aus_gt:
        xy, crs = lade_gt(args.aus_gt, args.gt_layer, args.gt_spalten)
        xy = entferne_ausreisser(xy, args.gt_ausreisser_abstand)
        xy = sortiere_entlang_strecke(xy)
        print(
            f"{len(xy)} GT-Punkte entlang der Strecke sortiert (Ergebnis bitte in QGIS pruefen!)"
        )
    else:
        raise SystemExit("Bitte --aus-csv oder --aus-gt angeben.")

    # Die Achse MUSS im selben CRS liegen wie die LAS-Dateien, sonst landen
    # alle Punkte im selben Bin. Der index-Lauf sagt, welche Variante passt.
    if args.von_crs and args.nach_crs:
        from pyproj import Transformer  # noqa: PLC0415

        tf = Transformer.from_crs(args.von_crs, args.nach_crs, always_xy=True)
        x, y = tf.transform(xy[:, 0], xy[:, 1])
        xy = np.column_stack([x, y])
        print(f"  umprojiziert: {args.von_crs} -> {args.nach_crs}")
    if args.x_versatz or args.y_versatz:
        xy = xy + np.array([args.x_versatz, args.y_versatz], dtype=np.float64)
        print(f"  verschoben um X{args.x_versatz:+.1f} / Y{args.y_versatz:+.1f}")

    vorher = len(xy)
    xy = ausduennen(xy, args.schritt)
    d = np.hypot(*np.diff(xy, axis=0).T)
    print(
        f"  ausgeduennt: {vorher} -> {len(xy)} Stuetzstellen, Gesamtlaenge {d.sum():.1f} m, "
        f"groesster Stuetzstellenabstand {d.max():.1f} m"
    )
    if d.max() > 50 * max(args.schritt, 1.0):
        print(
            "  WARNUNG: sehr grosse Luecke in der Achse - moeglicherweise falsch sortiert "
            "oder mehrere getrennte Abschnitte in einer Datei."
        )

    with open(args.ziel, "w", encoding="utf-8") as fh:
        fh.write("# Achse fuer las_zu_copc_tiles.py (Format x;y)\n")
        for x, y in xy:
            fh.write(f"{x:.4f};{y:.4f}\n")
    print(f"Geschrieben: {args.ziel}")


# =========================== Unterbefehl: verarbeiten ===========================


def lade_chunk_modul(pfad_hinweis: Path | None):
    kandidaten = [pfad_hinweis] if pfad_hinweis else []
    kandidaten += [
        Path(__file__).resolve().parent / "las_zu_copc_tiles.py",
        Path.cwd() / "las_zu_copc_tiles.py",
    ]
    for k in kandidaten:
        if k and Path(k).is_file():
            spec = importlib.util.spec_from_file_location("las_zu_copc_tiles", k)
            modul = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modul)
            print(f"Chunking-Skript geladen: {k}")
            return modul
    raise SystemExit(
        "las_zu_copc_tiles.py nicht gefunden - bitte --chunk-skript angeben."
    )


def hole_datei(quelle: str, ziel_verzeichnis: Path) -> tuple[Path, bool]:
    """Stellt die Datei lokal bereit. Rueckgabe: (Pfad, temporaer?).
    Lokale/UNC-Pfade werden direkt gelesen (kein unnoetiges Kopieren)."""
    if not ist_url(quelle):
        return Path(quelle), False
    import urllib.request  # noqa: PLC0415

    ziel_verzeichnis.mkdir(parents=True, exist_ok=True)
    ziel = ziel_verzeichnis / quelle.rsplit("/", 1)[-1]
    print(f"  lade herunter -> {ziel}")
    start = time.monotonic()
    with urllib.request.urlopen(quelle, timeout=120) as resp, open(ziel, "wb") as fh:
        gesamt = int(resp.headers.get("Content-Length") or 0)
        geladen = 0
        while True:
            block = resp.read(8 << 20)
            if not block:
                break
            fh.write(block)
            geladen += len(block)
            if gesamt and geladen % (256 << 20) < (8 << 20):
                print(f"    {100.0 * geladen / gesamt:5.1f} %", flush=True)
    print(f"  {geladen / 1e9:.2f} GB in {time.monotonic() - start:.0f} s")
    return ziel, True


def cmd_verarbeiten(args: argparse.Namespace) -> None:
    with open(args.treffer, encoding="utf-8-sig") as fh:
        zeilen = list(csv.DictReader(fh))
    if not zeilen:
        raise SystemExit(f"{args.treffer} enthaelt keine Zeilen.")
    if args.km_von is not None or args.km_bis is not None:
        vor = len(zeilen)
        zeilen = [
            z
            for z in zeilen
            if z.get("km_von")
            and (args.km_von is None or float(z["km_von"]) >= args.km_von)
            and (args.km_bis is None or float(z["km_von"]) <= args.km_bis)
        ]
        print(f"Kilometerfilter: {vor} -> {len(zeilen)} Dateien")
    if args.limit:
        zeilen = zeilen[: args.limit]
    print(f"{len(zeilen)} Datei(en) zu verarbeiten")

    lzc = lade_chunk_modul(args.chunk_skript)
    achse_xy = achse_station = None
    if args.achsen_datei:
        achse_xy, achse_station = lzc.lade_achse_aus_datei(args.achsen_datei)
        print(f"Achse geladen: {len(achse_xy)} Stuetzstellen")
    else:
        print(
            "WARNUNG: keine --achsen-datei - die Achse wird je Datei aus der Punktwolke "
            "geschaetzt. Das kostet einen zusaetzlichen Lesevorgang je Datei."
        )

    args.ausgabe.mkdir(parents=True, exist_ok=True)
    frei = shutil.disk_usage(args.ausgabe).free
    print(f"Freier Platz unter {args.ausgabe}: {frei / 1e9:.1f} GB")

    worker = args.worker or os.cpu_count() or 4
    prio = args.start_prio
    for i, zeile in enumerate(zeilen, start=1):
        quelle = zeile["quelle"]
        prio_dir = args.ausgabe / f"Prio_{prio}"
        if (
            prio_dir.exists()
            and any(prio_dir.glob("segment_*.copc.laz"))
            and not args.ueberschreiben
        ):
            print(
                f"=== [{i}/{len(zeilen)}] {prio_dir.name} existiert bereits -> uebersprungen "
                f"(--ueberschreiben erzwingt Neuberechnung) ==="
            )
            prio += 1
            continue
        print(
            f"\n=== [{i}/{len(zeilen)}] {zeile['name']} -> {prio_dir.name} "
            f"({zeile.get('gt_punkte', '?')} GT-Punkte, {int(zeile['punkte']):,} Punkte) ==="
        )
        if args.trockenlauf:
            prio += 1
            continue
        pfad, temporaer = hole_datei(quelle, args.tmp_verzeichnis)
        try:
            lzc.verarbeite_datei(
                pfad,
                prio_dir,
                args.kantenlaenge,
                args.a_srs,
                args.tmp_verzeichnis,
                worker,
                achse_xy,
                achse_station,
                args.chunk_groesse,
                args.sample_schritt,
                args.fenster_laenge,
            )
        finally:
            if temporaer and pfad.exists():
                pfad.unlink()
                print(f"  temporaere LAS geloescht: {pfad.name}")
        prio += 1
    print("\nFertig.")


# =========================== CLI ===========================


def _spalten_paar(text: str) -> tuple[int, int]:
    a, b = text.split(",")
    return int(a), int(b)


def _namen_paar(text: str) -> tuple[str, str]:
    a, b = text.split(",")
    return a.strip(), b.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    unter = parser.add_subparsers(dest="befehl", required=True)

    # --- index ---
    p_idx = unter.add_parser(
        "index", help="LAS-Header lesen und mit der GT-Datei abgleichen"
    )
    p_idx.add_argument(
        "--gt", type=Path, required=True, help="GT-Datei (.gpkg oder .shp)"
    )
    p_idx.add_argument(
        "--gt-layer",
        default=None,
        help="Layername im GeoPackage (Default: erster Layer)",
    )
    p_idx.add_argument(
        "--gt-spalten",
        type=_namen_paar,
        default=None,
        help="Statt der Geometrie zwei Attributspalten verwenden, z.B. 'x,y'. "
        "Nuetzlich, wenn die Attribute in einem anderen CRS stehen als die Geometrie.",
    )
    p_idx.add_argument(
        "--las-verzeichnis",
        type=Path,
        default=None,
        help="Verzeichnis, rekursiv durchsucht",
    )
    p_idx.add_argument(
        "--url-liste",
        type=Path,
        default=None,
        help="Textdatei mit je einer LAS-URL pro Zeile",
    )
    p_idx.add_argument(
        "--muster",
        default="*.las",
        help="Dateimuster (Default *.las, *.laz wird mitgesucht)",
    )
    p_idx.add_argument(
        "--las-crs", default=None, help="CRS der LAS-Dateien erzwingen, z.B. EPSG:5683"
    )
    p_idx.add_argument(
        "--puffer",
        type=float,
        default=25.0,
        help="Suchpuffer um jede LAS-Box in Metern (Default 25)",
    )
    p_idx.add_argument(
        "--min-gt-punkte",
        type=int,
        default=1,
        help="Ab wie vielen GT-Punkten gilt eine Datei als Treffer (Default 1)",
    )
    p_idx.add_argument(
        "--gt-ausreisser-abstand",
        type=float,
        default=5000.0,
        help="GT-Punkte weiter als dieser Abstand vom Median werden ignoriert (0 = aus, Default 5000)",
    )
    p_idx.add_argument(
        "--ohne-crs-erkennung", action="store_true", help="VLRs nicht lesen (schneller)"
    )
    p_idx.add_argument(
        "--ausgabe-index",
        type=Path,
        default=Path("./index"),
        help="Zielverzeichnis (Default ./index)",
    )
    p_idx.set_defaults(funktion=cmd_index)

    # --- achse ---
    p_ach = unter.add_parser(
        "achse", help="Achsen-Datei fuer las_zu_copc_tiles.py bauen"
    )
    p_ach.add_argument(
        "--aus-csv",
        type=Path,
        default=None,
        help="Trajektorien-CSV (z.B. trajectory.csv)",
    )
    p_ach.add_argument(
        "--aus-gt",
        type=Path,
        default=None,
        help="Alternativ: GT-Datei als Achsenquelle",
    )
    p_ach.add_argument("--gt-layer", default=None)
    p_ach.add_argument("--gt-spalten", type=_namen_paar, default=None)
    p_ach.add_argument("--gt-ausreisser-abstand", type=float, default=5000.0)
    p_ach.add_argument(
        "--spalten",
        type=_spalten_paar,
        default=None,
        help="0-basierte Spaltenindizes 'x,y' fuer die CSV",
    )
    p_ach.add_argument(
        "--schritt",
        type=float,
        default=1.0,
        help="Mindestabstand der Stuetzstellen in Metern (Default 1)",
    )
    p_ach.add_argument(
        "--von-crs",
        default=None,
        help="Achse umprojizieren, Quell-CRS (z.B. EPSG:25832)",
    )
    p_ach.add_argument(
        "--nach-crs",
        default=None,
        help="Achse umprojizieren, Ziel-CRS = CRS der LAS-Dateien",
    )
    p_ach.add_argument(
        "--x-versatz",
        type=float,
        default=0.0,
        help="Konstanter Versatz auf X, z.B. 3000000 wenn die GK-Zonenkennziffer fehlt",
    )
    p_ach.add_argument(
        "--y-versatz", type=float, default=0.0, help="Konstanter Versatz auf Y"
    )
    p_ach.add_argument("--ziel", type=Path, required=True, help="Ausgabedatei")
    p_ach.set_defaults(funktion=cmd_achse)

    # --- verarbeiten ---
    p_ver = unter.add_parser(
        "verarbeiten", help="Treffer holen, kacheln, temporaere LAS loeschen"
    )
    p_ver.add_argument(
        "--treffer", type=Path, required=True, help="treffer.csv aus dem index-Lauf"
    )
    p_ver.add_argument(
        "--ausgabe", type=Path, required=True, help="COPC-Wurzelverzeichnis"
    )
    p_ver.add_argument("--start-prio", type=int, default=1)
    p_ver.add_argument(
        "--kantenlaenge",
        type=float,
        default=25.0,
        help="Segmentlaenge entlang der Achse in Metern (Default 25)",
    )
    p_ver.add_argument("--a-srs", default="EPSG:5683")
    p_ver.add_argument("--achsen-datei", type=Path, default=None)
    p_ver.add_argument("--tmp-verzeichnis", type=Path, default=Path("/tmp"))
    p_ver.add_argument("--worker", type=int, default=None)
    p_ver.add_argument("--chunk-groesse", type=int, default=2_000_000)
    p_ver.add_argument("--sample-schritt", type=int, default=20)
    p_ver.add_argument("--fenster-laenge", type=float, default=20.0)
    p_ver.add_argument(
        "--chunk-skript", type=Path, default=None, help="Pfad zu las_zu_copc_tiles.py"
    )
    p_ver.add_argument(
        "--limit", type=int, default=None, help="Nur die ersten N Treffer verarbeiten"
    )
    p_ver.add_argument("--km-von", type=float, default=None)
    p_ver.add_argument("--km-bis", type=float, default=None)
    p_ver.add_argument(
        "--ueberschreiben",
        action="store_true",
        help="Bestehende Prio-Ordner neu berechnen",
    )
    p_ver.add_argument(
        "--trockenlauf", action="store_true", help="Nur anzeigen, nichts tun"
    )
    p_ver.set_defaults(funktion=cmd_verarbeiten)

    args = parser.parse_args()
    args.funktion(args)


if __name__ == "__main__":
    main()
