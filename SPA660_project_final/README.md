# SPA660 Project: eROSITA X-ray and Gaia Optical Analysis

**Target:** 1eRASS J052113.8−520257 (EC 05200−5205)

This project combines eROSITA X-ray spectral analysis with Gaia DR3 astrometry and optical photometry to study a cataclysmic variable.

## Project components

- **Part A – X-ray spectral analysis:** The eROSITA source and background spectra, together with the response files, are stored in `xray_data/`. The spectrum was analysed and fitted in **XSPEC** using the absorbed thermal-plasma model `tbabs × apec` over 0.2–8.0 keV. The report contains the best-fit parameters, spectrum and residual plot. The X-ray fitting and plotting were performed in XSPEC.
- **Part B – Optical counterpart and distance:** Gaia DR3 source `4772021167680733696` was identified as the most likely optical counterpart. Its Gaia-to-eROSITA source-region separation is approximately 0.838 arcsec, and its formal inverse-parallax distance is 539.50 ± 7.13 pc.
- **Part C – Optical light curve:** The Gaia G-band epoch photometry was validated and used to produce the G-band light curve, including the photometric quality flags.
- **Part D – Period analysis:** Weighted and unweighted Lomb–Scargle analyses were compared, and the G-band observations were phase-folded. The unweighted candidate period is approximately 33.92 hours, close to the 34-hour literature value, but its global false-alarm probability is 0.257; therefore, the available Gaia data do not establish a statistically reliable orbital period.

## Main folders and files

- `xray_data/` – Raw eROSITA data used for Part A.
- `gaia_data/` – Gaia DR3 candidate, epoch-photometry FITS and derived G-band data.
- `scripts/` – Python scripts for Gaia data validation, Part B calculations, light-curve plotting, Lomb–Scargle calculations and phase folding.
- `results/figures/` – Final project figures.
- `results/tables/` – Validated Gaia data and calculated result tables.
- `reports/` – Supporting calculation and analysis reports, where included.
- `SPA660_project.pdf` – Final project report.

## Reproducing the Gaia analysis

Install the Python dependencies listed in `requirements.txt` (or, if not provided, install `numpy`, `pandas`, `astropy`, `scipy` and `matplotlib`). From the project root, run the scripts in this order:

1. `scripts/validate_and_prepare_gaia.py`
2. `scripts/calculate_part_B_distance.py`
3. `scripts/plot_g_band_lightcurve.py`
4. `scripts/calculate_lomb_scargle_comparison.py`
5. `scripts/plot_lomb_scargle_periodogram.py`
6. `scripts/fold_lightcurve.py`
7. `scripts/plot_phase_folded_lightcurve.py`

Part A's spectral fitting and spectrum/residual plot were produced in XSPEC; the raw inputs are retained in `xray_data/`.
