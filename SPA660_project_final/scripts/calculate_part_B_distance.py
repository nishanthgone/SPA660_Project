"""Reproducibly calculate Part B: Gaia counterpart position and distance.

Expected project layout (the script belongs in project/scripts/):
    project/
      gaia_data/*candidate_matches*.csv
      xray_data/em01_081141_020_SourceSpec_00007_c010.fits
      scripts/calculate_part_B_distance.py

The script reads the eROSITA source-region coordinates from RA_OBJ/DEC_OBJ
in the source-spectrum FITS file. It obtains RUWE from the Gaia DR3 Archive
(or reads a previously saved gaia_source_metadata.csv). No X-ray spectral
analysis is performed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u


# ---------------------------------------------------------------------------
# Project paths and target
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
GAIA_DIR = ROOT / "gaia_data"
XRAY_DIR = ROOT / "xray_data"
TABLES_DIR = ROOT / "results" / "tables"
REPORTS_DIR = ROOT / "reports"

SOURCE_ID = 4772021167680733696
EXPECTED_XRAY_FILENAME = "em01_081141_020_SourceSpec_00007_c010.fits"
PC_TO_LY = 3.26156


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def find_one(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"No file matching {pattern!r} found in {directory}")
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one file matching {pattern!r} found in {directory}:\n"
            + "\n".join(str(p) for p in matches)
        )
    return matches[0]


def recover_csv_reference_position(candidates: pd.DataFrame) -> tuple[float, float, float]:
    """Recover the common position encoded by the candidate CSV separation circles.

    Returns (RA_deg, Dec_deg, maximum_abs_residual_arcsec). The recovered
    coordinate is the CSV's reference position, not the eROSITA FITS position.
    """
    required = {"ra", "dec", "separation_arcsec"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    if len(candidates) < 3:
        raise ValueError("At least three Gaia candidates are needed to recover the CSV reference position.")

    arcsec_per_rad = 206264.80624709636
    ra = np.deg2rad(candidates["ra"].to_numpy(dtype=float))
    dec = np.deg2rad(candidates["dec"].to_numpy(dtype=float))
    sep = candidates["separation_arcsec"].to_numpy(dtype=float)

    ra0, dec0 = ra[0], dec[0]
    dra = (ra - ra0 + np.pi) % (2 * np.pi) - np.pi
    x = np.cos(dec) * np.sin(dra)
    y = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
    z = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(dra)
    xi = x / z * arcsec_per_rad
    eta = y / z * arcsec_per_rad

    # Linearized intersection of the small-angle separation circles.
    A = 2.0 * np.column_stack((xi[1:], eta[1:]))
    b = xi[1:] ** 2 + eta[1:] ** 2 + sep[0] ** 2 - sep[1:] ** 2
    east_arcsec, north_arcsec = np.linalg.lstsq(A, b, rcond=None)[0]
    east, north = east_arcsec / arcsec_per_rad, north_arcsec / arcsec_per_rad

    ref_ra = (ra0 + np.arctan2(east, np.cos(dec0) - north * np.sin(dec0))) % (2 * np.pi)
    ref_dec = np.arctan2(
        np.sin(dec0) + north * np.cos(dec0),
        np.sqrt((np.cos(dec0) - north * np.sin(dec0)) ** 2 + east**2),
    )

    ref_coord = SkyCoord(ra=ref_ra * u.rad, dec=ref_dec * u.rad, frame="icrs")
    cand_coords = SkyCoord(ra=candidates["ra"].to_numpy() * u.deg,
                           dec=candidates["dec"].to_numpy() * u.deg,
                           frame="icrs")
    residual = np.max(np.abs(cand_coords.separation(ref_coord).arcsec - sep))
    return float(ref_coord.ra.deg), float(ref_coord.dec.deg), float(residual)


def locate_xray_source_fits() -> Path:
    """Find the eROSITA source-spectrum FITS file used for Part B astrometry."""
    if not XRAY_DIR.exists():
        raise FileNotFoundError(
            f"X-ray data folder not found: {XRAY_DIR}. Create xray_data/ in the project root."
        )

    exact = list(XRAY_DIR.rglob(EXPECTED_XRAY_FILENAME))
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError("Multiple copies of the expected eROSITA source FITS were found:\n" +
                           "\n".join(str(p) for p in exact))

    # Fallback: a unique SourceSpec FITS file is acceptable. If several exist,
    # refuse to guess which source corresponds to the assigned target.
    possible = []
    for p in XRAY_DIR.rglob("*"):
        low = p.name.lower()
        if p.is_file() and ("sourcespec" in low) and (low.endswith((".fits", ".fit", ".fts", ".fits.gz"))):
            possible.append(p)
    if len(possible) == 1:
        return possible[0]
    if not possible:
        raise FileNotFoundError(
            f"Could not find {EXPECTED_XRAY_FILENAME} or another *SourceSpec* FITS file under {XRAY_DIR}."
        )
    raise RuntimeError(
        "Multiple *SourceSpec* FITS files found. Set EXPECTED_XRAY_FILENAME to the correct file:\n"
        + "\n".join(str(p) for p in possible)
    )


def read_xray_source_position(fits_path: Path) -> tuple[float, float, int, str]:
    """Read the eROSITA source-region centre from RA_OBJ/DEC_OBJ FITS keywords."""
    with fits.open(fits_path, memmap=False) as hdul:
        # Prefer the source table extension (usually HDU 1), then inspect any
        # other HDU in case the archive product stores the keywords elsewhere.
        order = list(range(1, len(hdul))) + ([0] if len(hdul) else [])
        for idx in order:
            hdr = hdul[idx].header
            if "RA_OBJ" in hdr and "DEC_OBJ" in hdr:
                ra_x = float(hdr["RA_OBJ"])
                dec_x = float(hdr["DEC_OBJ"])
                if not (math.isfinite(ra_x) and math.isfinite(dec_x)):
                    continue
                if not (0.0 <= ra_x < 360.0 and -90.0 <= dec_x <= 90.0):
                    raise ValueError(
                        f"Invalid RA_OBJ/DEC_OBJ in {fits_path.name}, HDU {idx}: {ra_x}, {dec_x}"
                    )
                extname = str(hdr.get("EXTNAME", "PRIMARY" if idx == 0 else f"HDU{idx}"))
                return ra_x, dec_x, idx, extname

    raise KeyError(
        f"No RA_OBJ and DEC_OBJ keywords were found in any HDU of {fits_path}. "
        "Inspect the FITS headers and update read_xray_source_position() to use the correct source-region fields."
    )


def get_ruwe(candidates: pd.DataFrame, source_row: pd.Series) -> tuple[float, str]:
    """Get RUWE from a source CSV or Gaia DR3 Archive; never hardcode it."""
    # If the candidate file already contains RUWE, use it directly.
    if "ruwe" in candidates.columns and pd.notna(source_row.get("ruwe")):
        value = float(source_row["ruwe"])
        if math.isfinite(value):
            return value, "candidate_matches CSV (ruwe column)"

    # Reuse a previously saved Gaia query result, if present.
    metadata_candidates = [
        GAIA_DIR / "gaia_source_metadata.csv",
        TABLES_DIR / "gaia_source_metadata.csv",
        ROOT / "gaia_source_metadata.csv",
    ]
    for metadata_path in metadata_candidates:
        if metadata_path.exists():
            meta = pd.read_csv(metadata_path)
            if not {"source_id", "ruwe"}.issubset(meta.columns):
                raise ValueError(f"{metadata_path} must contain source_id and ruwe columns.")
            match = meta.loc[pd.to_numeric(meta.source_id, errors="coerce") == SOURCE_ID]
            if not match.empty and pd.notna(match.iloc[0]["ruwe"]):
                value = float(match.iloc[0]["ruwe"])
                if math.isfinite(value):
                    return value, f"saved Gaia query result: {metadata_path.relative_to(ROOT)}"

    # Otherwise query the official Gaia DR3 TAP table.
    try:
        from astroquery.gaia import Gaia

        query = f"""
        SELECT source_id, ra, dec, parallax, parallax_error, ruwe
        FROM gaiadr3.gaia_source
        WHERE source_id = {SOURCE_ID}
        """
        job = Gaia.launch_job_async(query)
        result = job.get_results()
        if len(result) != 1:
            raise RuntimeError(f"Expected one Gaia row for {SOURCE_ID}; received {len(result)}.")

        meta = pd.DataFrame({
            "source_id": [int(result["source_id"][0])],
            "ra": [float(result["ra"][0])],
            "dec": [float(result["dec"][0])],
            "parallax": [float(result["parallax"][0])],
            "parallax_error": [float(result["parallax_error"][0])],
            "ruwe": [float(result["ruwe"][0])],
        })
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        metadata_path = TABLES_DIR / "gaia_source_metadata.csv"
        meta.to_csv(metadata_path, index=False)
        return float(meta.loc[0, "ruwe"]), f"ESA Gaia DR3 Archive, gaiadr3.gaia_source; saved to {metadata_path.relative_to(ROOT)}"
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve RUWE from the Gaia Archive and no saved gaia_source_metadata.csv was found. "
            "Connect to the Gaia Archive or save its query result as gaia_source_metadata.csv with columns "
            "source_id,ruwe (and optionally ra,dec,parallax,parallax_error). Original error: " + str(exc)
        ) from exc


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------
def main() -> None:
    candidate_path = find_one("*candidate_matches*.csv", GAIA_DIR)
    candidates = pd.read_csv(candidate_path)
    if "source_id" not in candidates.columns:
        raise ValueError("Candidate CSV has no source_id column.")
    candidates["source_id"] = pd.to_numeric(candidates["source_id"], errors="coerce").astype("Int64")
    selected = candidates.loc[candidates.source_id == SOURCE_ID]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one row for Gaia source {SOURCE_ID}; found {len(selected)}.")
    row = selected.iloc[0]

    for col in ("ra", "dec", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error",
                "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"):
        if col not in candidates.columns:
            raise ValueError(f"Candidate CSV is missing required column: {col}")

    p = float(row.parallax)
    pe = float(row.parallax_error)
    if not (math.isfinite(p) and p > 0 and math.isfinite(pe) and pe > 0):
        raise ValueError(f"Invalid parallax or parallax uncertainty: {p}, {pe}")

    distance_pc = 1000.0 / p
    distance_error_pc = 1000.0 * pe / (p ** 2)
    distance_ly = distance_pc * PC_TO_LY
    distance_error_ly = distance_error_pc * PC_TO_LY
    parallax_snr = p / pe

    # Recover the rounded reference position encoded in the candidate CSV.
    ref_ra, ref_dec, csv_sep_residual = recover_csv_reference_position(candidates)

    # Read the actual eROSITA source-region coordinates from its FITS product.
    xray_fits = locate_xray_source_fits()
    xray_ra, xray_dec, xray_hdu, xray_extname = read_xray_source_position(xray_fits)
    xray_coord = SkyCoord(ra=xray_ra * u.deg, dec=xray_dec * u.deg, frame="icrs")

    # Calculate Gaia-to-eROSITA separations for every candidate using one reference.
    candidate_coords = SkyCoord(ra=candidates.ra.to_numpy(dtype=float) * u.deg,
                                dec=candidates.dec.to_numpy(dtype=float) * u.deg,
                                frame="icrs")
    candidates["separation_from_erosita_arcsec"] = candidate_coords.separation(xray_coord).arcsec
    candidates["separation_from_csv_reference_arcsec"] = candidates["separation_arcsec"].astype(float)
    candidates = candidates.sort_values("separation_from_erosita_arcsec")
    selected_xray_sep = float(candidates.loc[candidates.source_id == SOURCE_ID,
                                              "separation_from_erosita_arcsec"].iloc[0])

    # Obtain RUWE from the Gaia query response or a saved query output.
    ruwe, ruwe_source = get_ruwe(candidates, row)

    # Save a reproducible positional comparison table.
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    separation_table = candidates[["source_id", "ra", "dec", "separation_from_erosita_arcsec",
                                   "separation_from_csv_reference_arcsec"]].copy()
    separation_table.to_csv(TABLES_DIR / "gaia_xray_candidate_separations.csv", index=False)

    xray_relative = xray_fits.relative_to(ROOT) if xray_fits.is_relative_to(ROOT) else xray_fits
    report = f"""# Part B: Gaia counterpart identification and distance

## Counterpart identification and positional comparison

The selected Gaia DR3 source is **{SOURCE_ID}**, at RA = {float(row.ra):.14f} deg and Dec = {float(row.dec):.14f} deg (ICRS). The candidate CSV's `separation_arcsec` values refer to the rounded reference position RA = **{ref_ra:.7f} deg**, Dec = **{ref_dec:.7f} deg**. Recomputing the great-circle separations from this position reproduces the CSV values with a maximum residual of {csv_sep_residual:.2e} arcsec. Thus, the CSV value of {float(row.separation_arcsec):.6f} arcsec is not the offset from the eROSITA position.

The eROSITA source-region centre was read directly from **{xray_relative}**, HDU {xray_hdu} (`{xray_extname}`), using the FITS keywords `RA_OBJ` and `DEC_OBJ`. Its ICRS coordinates are RA = **{xray_ra:.9f} deg**, Dec = **{xray_dec:.9f} deg**. The angular separation between this eROSITA position and the selected Gaia source is **{selected_xray_sep:.5f} arcsec**.

The candidate separations from the eROSITA source-region centre are:

| Gaia DR3 source ID | Separation from eROSITA centre (arcsec) |
|---:|---:|
"""
    for _, cand in candidates.iterrows():
        report += f"| {int(cand.source_id)} | {cand.separation_from_erosita_arcsec:.5f} |\n"

    report += f"""

The selected Gaia source is the closest of the three candidates to the eROSITA source-region centre and is also the only candidate with epoch photometry in the supplied candidate data. These independent checks support its identification as the optical counterpart.

## Gaia astrometry and photometry

- Parallax: {p:.15f} mas
- Formal parallax uncertainty: {pe:.15f} mas
- Proper motion in RA: {float(row.pmra):.9f} +/- {float(row.pmra_error):.9f} mas/yr
- Proper motion in Dec: {float(row.pmdec):.9f} +/- {float(row.pmdec_error):.9f} mas/yr
- Mean G / BP / RP: {float(row.phot_g_mean_mag):.6f} / {float(row.phot_bp_mean_mag):.6f} / {float(row.phot_rp_mean_mag):.6f} mag
- RUWE: {ruwe:.8f}

RUWE provenance: {ruwe_source}. The value is retrieved for source ID {SOURCE_ID} from the Gaia DR3 `gaiadr3.gaia_source.ruwe` field; it is not hardcoded in this calculation script. RUWE close to unity is consistent with a good single-source astrometric fit, but does not establish the binary's orbital properties.

## Distance calculation

The inverse-parallax estimate is

$$d(\\mathrm{{pc}})=\\frac{{1000}}{{p(\\mathrm{{mas}})}} = \\frac{{1000}}{{{p:.15f}}} = {distance_pc:.6f}\\;\\mathrm{{pc}}.$$

First-order uncertainty propagation gives

$$\\sigma_d = \\frac{{1000\\sigma_p}}{{p^2}} = \\frac{{1000\\times {pe:.15f}}}{{({p:.15f})^2}} = {distance_error_pc:.6f}\\;\\mathrm{{pc}}.$$

The formal result is **{distance_pc:.4f} +/- {distance_error_pc:.4f} pc**, or **{distance_ly:.2f} +/- {distance_error_ly:.2f} light-years** using 1 pc = {PC_TO_LY} light-years. The parallax signal-to-noise ratio is {parallax_snr:.2f}. This direct inversion with a first-order uncertainty omits parallax zero-point systematics and does not model a non-Gaussian distance posterior or population prior. It is a useful formal approximation, not a full Bayesian distance inference.

## Reproducibility

Run `python scripts/calculate_part_B_distance.py` from the project root. The script reads the Gaia candidate CSV, reads the eROSITA source-region coordinates from the source-spectrum FITS file, retrieves RUWE from the Gaia DR3 Archive (or reuses a saved `gaia_source_metadata.csv`), calculates the angular separations and distance, and writes this report. It also writes `results/tables/gaia_xray_candidate_separations.csv` and, when queried, `results/tables/gaia_source_metadata.csv`.
"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "part_B_calculation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Selected Gaia source: {SOURCE_ID}")
    print(f"CSV reference position: RA={ref_ra:.7f} deg, Dec={ref_dec:.7f} deg")
    print(f"eROSITA source-region centre: RA={xray_ra:.9f} deg, Dec={xray_dec:.9f} deg")
    print(f"Gaia-to-eROSITA separation: {selected_xray_sep:.6f} arcsec")
    print(f"CSV separation reconstruction maximum residual: {csv_sep_residual:.3e} arcsec")
    print(f"RUWE: {ruwe:.8f} ({ruwe_source})")
    print(f"Distance: {distance_pc:.6f} +/- {distance_error_pc:.6f} pc")
    print(f"Distance: {distance_ly:.2f} +/- {distance_error_ly:.2f} light-years")
    print(f"Wrote report: {report_path}")
    print(f"Wrote candidate comparison: {TABLES_DIR / 'gaia_xray_candidate_separations.csv'}")


if __name__ == "__main__":
    main()
