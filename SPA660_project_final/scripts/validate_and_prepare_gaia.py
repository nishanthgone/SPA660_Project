from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'gaia_data'
OUT = ROOT / 'results' / 'tables' / 'validated_gaia_G_band_data.csv'
EXPECTED_SOURCE = 4772021167680733696
EXPECTED = {'ra': 80.30761698342934, 'dec': -52.04978558145273, 'parallax': 1.8535512177559155, 'parallax_error': 0.024504205211997032}

def parse_vector(value, kind=float):
    text = str(value).strip().strip('()')
    parts = [part.strip() for part in text.split(',')]
    if kind is bool:
        return np.array([p.lower() == 'true' for p in parts], dtype=bool)
    if kind is int:
        return np.array([int(p) for p in parts], dtype=int)
    return np.array([float(p) if p.lower() != 'nan' else np.nan for p in parts], dtype=float)

candidate_path = next(DATA.glob('*candidate_matches*.csv'))
lightcurve_path = next(DATA.glob('*G_band_lightcurve*.csv'))
fits_path = next(DATA.glob('*epoch_photometry_raw*.fits'))
candidates = pd.read_csv(candidate_path)
lightcurve = pd.read_csv(lightcurve_path)
expected = candidates.loc[candidates.source_id == EXPECTED_SOURCE]
assert len(expected) == 1
assert int(expected.iloc[0].source_id) == EXPECTED_SOURCE
assert expected.iloc[0].separation_arcsec == candidates.separation_arcsec.min()
for key in ('ra', 'dec', 'parallax'):
    assert np.isclose(float(expected.iloc[0][key]), EXPECTED[key], rtol=0, atol=1e-12), key
# The candidate CSV stores parallax_error to nine decimal places.
assert np.isclose(float(expected.iloc[0].parallax_error), EXPECTED['parallax_error'], rtol=0, atol=5e-10)
assert len(lightcurve) == 39
with fits.open(fits_path) as hdul:
    row = hdul[1].data[0]
    assert int(row['source_id']) == EXPECTED_SOURCE
    vectors = {name: parse_vector(row[name], kind) for name, kind in [
        ('transit_id', str), ('g_transit_time', float), ('g_transit_flux', float),
        ('g_transit_flux_error', float), ('g_transit_flux_over_error', float), ('g_transit_mag', float),
        ('g_transit_n_obs', int), ('photometry_flag_noisy_data', bool),
        ('variability_flag_g_reject', bool)]}
    vectors['transit_id'] = np.array(str(row['transit_id']).strip().strip('()').split(', '), dtype=str)
valid = (vectors['g_transit_n_obs'] > 0) & np.isfinite(vectors['g_transit_time']) & np.isfinite(vectors['g_transit_mag']) & np.isfinite(vectors['g_transit_flux'])
assert valid.sum() == 39 and len(valid) == 40
raw_valid = pd.DataFrame({
    'transit_id': vectors['transit_id'][valid],
    'time_days_from_BJD2010': vectors['g_transit_time'][valid],
    'g_mag_vega_fits': vectors['g_transit_mag'][valid],
    'g_flux_electron_per_s_fits': vectors['g_transit_flux'][valid],
    'g_flux_error_electron_per_s_fits': vectors['g_transit_flux_error'][valid],
    'g_flux_over_error_fits': vectors['g_transit_flux_over_error'][valid],
    'g_n_ccd_observations_fits': vectors['g_transit_n_obs'][valid],
    'photometry_flag_noisy_data': vectors['photometry_flag_noisy_data'][valid],
    'variability_flag_g_reject': vectors['variability_flag_g_reject'][valid],
})
# Match by transit time, never by source array position.
merged = lightcurve.merge(raw_valid, on='time_days_from_BJD2010', how='outer', validate='one_to_one', indicator=True)
assert (merged['_merge'] == 'both').all(), 'FITS/CSV transit-time match failed'
assert np.allclose(merged.g_mag_vega, merged.g_mag_vega_fits, rtol=0, atol=1e-9)
assert np.allclose(merged.g_flux_electron_per_s, merged.g_flux_electron_per_s_fits, rtol=0, atol=1e-7)
assert np.allclose(merged.g_flux_error_electron_per_s, merged.g_flux_error_electron_per_s_fits, rtol=0, atol=1e-7)
assert np.allclose(merged.g_flux_over_error, merged.g_flux_over_error_fits, rtol=0, atol=1e-4)
assert np.array_equal(merged.g_n_ccd_observations.astype(int), merged.g_n_ccd_observations_fits.astype(int))
merged = merged.drop(columns=['_merge'])
merged['sigma_mag'] = (2.5 / np.log(10)) * (merged.g_flux_error_electron_per_s / merged.g_flux_electron_per_s)
merged['primary_period_sample'] = ~merged.variability_flag_g_reject
merged.to_csv(OUT, index=False)
print(f'Candidate selected: {EXPECTED_SOURCE}; separation={expected.iloc[0].separation_arcsec:.9f} arcsec')
print(f'Raw FITS transits={len(valid)}; valid={valid.sum()}; invalid={len(valid)-valid.sum()}')
print(f'CSV observations={len(lightcurve)}; time-matched valid FITS records={len(merged)}')
print(f'Noisy valid={int(merged.photometry_flag_noisy_data.sum())}; G-rejected valid={int(merged.variability_flag_g_reject.sum())}; valid rejected transit(s) in invalid record={int((vectors["variability_flag_g_reject"] & ~valid).sum())}')
print(f'Primary period analysis: retained={int(merged.primary_period_sample.sum())}; excluded={int((~merged.primary_period_sample).sum())}; noisy observations retained={int((merged.primary_period_sample & merged.photometry_flag_noisy_data).sum())}')
print(f'Wrote {OUT}')
