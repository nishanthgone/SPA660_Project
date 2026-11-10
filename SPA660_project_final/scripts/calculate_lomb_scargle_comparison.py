from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, peak_widths

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
df = pd.read_csv(TABLES / "validated_gaia_G_band_data.csv")
primary = df.loc[~df.variability_flag_g_reject].copy()
samples = {"primary_37": primary, "all_valid_39": df}

baseline = float(df.bjd_tcb.max() - df.bjd_tcb.min())
dfreq = 1.0 / (5.0 * baseline)
# Preserve the original 0.02/day grid exactly, and add a separate regular low-
# frequency segment for the 100- and 200-day extensions. Astropy's fast method
# requires each evaluated segment to be regular.
freq_high = np.arange(0.02, 24.0, dfreq)
freq_low = np.arange(1 / 200.0, 0.02, dfreq)
freq = np.concatenate([freq_low, freq_high])
period_ranges = {
    "max_period_50d": (1 / 50.0, 24.0),
    "max_period_100d": (1 / 100.0, 24.0),
    "max_period_200d": (1 / 200.0, 24.0),
    "period_1_to_1000h": (24 / 1000.0, 24.0),
}
literature_period_h = 34.0
literature_frequency = 24.0 / literature_period_h

def make_ls(sample: pd.DataFrame, weighted: bool) -> LombScargle:
    dy = sample.sigma_mag.to_numpy() if weighted else None
    return LombScargle(
        sample.bjd_tcb.to_numpy(),
        sample.g_mag_vega.to_numpy(),
        dy=dy,
        fit_mean=True,
        center_data=True,
        nterms=1,
    )

def power_at(ls: LombScargle, f: float) -> float:
    return float(ls.power(np.array([f]), method="chi2", normalization="standard")[0])

def refine(ls: LombScargle, frequencies: np.ndarray, powers: np.ndarray) -> dict:
    i = int(np.argmax(powers))
    maxima, _ = find_peaks(powers)
    if i not in maxima:
        maxima = np.append(maxima, i)
    candidates = maxima[np.argsort(powers[maxima])[-30:]]
    refined = []
    for j in candidates:
        lo = max(float(frequencies[0]), float(frequencies[j] - 1.5 * dfreq))
        hi = min(float(frequencies[-1]), float(frequencies[j] + 1.5 * dfreq))
        if hi <= lo:
            continue
        opt = minimize_scalar(lambda x: -power_at(ls, float(x)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
        refined.append((float(-opt.fun), float(opt.x), int(j), bool(opt.success)))
    best_power, best_frequency, parent_i, ok = max(refined)
    return {
        "grid_index": i,
        "grid_frequency_per_day": float(frequencies[i]),
        "grid_power": float(powers[i]),
        "refined_parent_grid_frequency_per_day": float(frequencies[parent_i]),
        "refined_frequency_per_day": best_frequency,
        "refined_period_days": float(1 / best_frequency),
        "refined_period_hours": float(24 / best_frequency),
        "refined_power": best_power,
        "local_refinement_success": ok,
    }

def fast_power(ls: LombScargle) -> np.ndarray:
    return np.concatenate([
        ls.power(freq_low, method="fast", normalization="standard"),
        ls.power(freq_high, method="fast", normalization="standard"),
    ])

periodograms: dict[tuple[str, bool], np.ndarray] = {}
objects: dict[tuple[str, bool], LombScargle] = {}
peaks: dict[tuple[str, bool], dict] = {}
diagnostic_rows = []
comparison_rows = []
fap_levels = [0.1, 0.05, 0.01]
for sample_name, sample in samples.items():
    for weighted in (True, False):
        method_name = "weighted" if weighted else "unweighted"
        ls = make_ls(sample, weighted)
        full_power = fast_power(ls)
        objects[(sample_name, weighted)] = ls
        periodograms[(sample_name, weighted)] = full_power
        if sample_name == "primary_37":
            peaks[(sample_name, weighted)] = refine(ls, freq, full_power)
        for range_name, (fmin, fmax) in period_ranges.items():
            mask = (freq >= fmin) & (freq <= fmax)
            f_sub, p_sub = freq[mask], full_power[mask]
            peak = refine(ls, f_sub, p_sub)
            grid_fap = float(ls.false_alarm_probability(
                peak["grid_power"], method="baluev", minimum_frequency=fmin, maximum_frequency=fmax
            ))
            fap = float(ls.false_alarm_probability(
                peak["refined_power"], method="baluev", minimum_frequency=fmin, maximum_frequency=fmax
            ))
            levels = ls.false_alarm_level(
                fap_levels, method="baluev", minimum_frequency=fmin, maximum_frequency=fmax
            )
            comparison_rows.append({
                "sample": sample_name,
                "weighting": method_name,
                "search_range": range_name,
                "n_retained": len(sample),
                "n_g_rejected_excluded": int(df.variability_flag_g_reject.sum()) if sample_name == "primary_37" else 0,
                "n_noisy_retained": int(sample.photometry_flag_noisy_data.sum()),
                "frequency_min_per_day": float(fmin),
                "frequency_max_per_day": float(fmax),
                "frequency_step_per_day": float(dfreq),
                "grid_frequency_per_day": peak["grid_frequency_per_day"],
                "grid_period_days": float(1 / peak["grid_frequency_per_day"]),
                "grid_period_hours": float(24 / peak["grid_frequency_per_day"]),
                "grid_power": peak["grid_power"],
                "refined_frequency_per_day": peak["refined_frequency_per_day"],
                "refined_period_days": peak["refined_period_days"],
                "refined_period_hours": peak["refined_period_hours"],
                "refined_power": peak["refined_power"],
                "grid_peak_global_fap_baluev": grid_fap,
                "global_fap_baluev": fap,
                "fap_0_10_power_level": float(levels[0]),
                "fap_0_05_power_level": float(levels[1]),
                "fap_0_01_power_level": float(levels[2]),
                "boundary_peak": bool(abs(peak["grid_frequency_per_day"] - fmin) <= dfreq * 1.01),
            })

        if sample_name == "primary_37":
            out = pd.DataFrame({
                "frequency_per_day": freq,
                "period_days": 1 / freq,
                "power": full_power,
            })
            out.to_csv(TABLES / f"lomb_scargle_{method_name}.csv", index=False)

# Weight diagnostics preserve all valid observations; no observation is removed.
df["inverse_variance_weight"] = 1.0 / df.sigma_mag**2
weight_rows = df.sort_values("inverse_variance_weight", ascending=False).copy()
weight_rows.to_csv(TABLES / "weighted_observation_weights.csv", index=False)

# Local maxima and exact benchmark calculations on the shared grid.
secondary_rows = []
for method_name, weighted in (("weighted", True), ("unweighted", False)):
    ls = objects[("primary_37", weighted)]
    power = periodograms[("primary_37", weighted)]
    peaks_idx, _ = find_peaks(power)
    peaks_idx = peaks_idx[np.argsort(power[peaks_idx])[-15:][::-1]]
    for rank, i in enumerate(peaks_idx[:10], start=1):
        secondary_rows.append({"weighting": method_name, "rank": rank, "frequency_per_day": freq[i], "period_hours": 24 / freq[i], "power": power[i]})
    # Sampling window is computed from the same selected observation times.
    times = primary.bjd_tcb.to_numpy()
    window = np.abs(np.exp(2j * np.pi * np.outer(freq, times - times.min())).sum(axis=1) / len(times)) ** 2
    pd.DataFrame({"frequency_per_day": freq, "period_days": 1 / freq, "sampling_window_power": window}).to_csv(
        TABLES / f"sampling_window_{method_name}.csv", index=False
    )
    best = peaks[("primary_37", weighted)]
    bench_power = power_at(ls, literature_frequency)
    # Re-evaluate a fine local grid about the selected short-range maximum.
    short_mask = (freq >= 24 / 1000) & (freq <= 24)
    f_short, p_short = freq[short_mask], power[short_mask]
    ix_short = int(np.argmax(p_short))
    refined_short = refine(ls, f_short, p_short)
    fine_short = np.linspace(refined_short["refined_frequency_per_day"] - 0.01,
                             refined_short["refined_frequency_per_day"] + 0.01, 20001)
    fine_power = ls.power(fine_short, method="fast", normalization="standard")
    fine_ix = int(np.argmax(fine_power))
    widths = peak_widths(fine_power, [fine_ix], rel_height=0.5)
    left_f = float(np.interp(widths[2][0], np.arange(len(fine_short)), fine_short))
    right_f = float(np.interp(widths[3][0], np.arange(len(fine_short)), fine_short))
    secondary_rows.append({"weighting": method_name, "rank": "short_range_main", "frequency_per_day": f_short[ix_short], "period_hours": 24 / f_short[ix_short], "power": p_short[ix_short]})
    peaks[("primary_37", weighted)]["power_at_exact_34h"] = bench_power
    peaks[("primary_37", weighted)]["short_range_grid_frequency_per_day"] = float(f_short[ix_short])
    peaks[("primary_37", weighted)]["short_range_grid_period_hours"] = float(24 / f_short[ix_short])
    peaks[("primary_37", weighted)]["short_range_grid_power"] = float(p_short[ix_short])
    peaks[("primary_37", weighted)]["short_range_refined"] = refined_short
    peaks[("primary_37", weighted)]["short_range_half_prominence_width_per_day"] = right_f - left_f
    peaks[("primary_37", weighted)]["short_range_half_prominence_left_frequency_per_day"] = left_f
    peaks[("primary_37", weighted)]["short_range_half_prominence_right_frequency_per_day"] = right_f

pd.DataFrame(comparison_rows).to_csv(TABLES / "lomb_scargle_comparison.csv", index=False)
pd.DataFrame(secondary_rows).to_csv(TABLES / "lomb_scargle_secondary_peaks.csv", index=False)

# Statistical period error conditional on a coherent sinusoid and independent,
# homoscedastic residuals: parametric residual bootstrap, retaining observation
# times and refitting only within +/- 0.05 cycles/day of the unweighted peak.
ls_u = objects[("primary_37", False)]
u = peaks[("primary_37", False)]["short_range_refined"]
t = primary.bjd_tcb.to_numpy()
y = primary.g_mag_vega.to_numpy()
fit = ls_u.model(t, u["refined_frequency_per_day"])
resid = y - fit
rng = np.random.default_rng(92311)
f_local = np.linspace(u["refined_frequency_per_day"] - 0.05, u["refined_frequency_per_day"] + 0.05, 2001)
boot_freq = []
for _ in range(400):
    ysim = fit + rng.normal(0, float(np.std(resid, ddof=3)), size=len(y))
    lsim = LombScargle(t, ysim, fit_mean=True, center_data=True, nterms=1)
    p_sim = lsim.power(f_local, method="fast", normalization="standard")
    j = int(np.argmax(p_sim))
    lo, hi = f_local[max(0, j - 2)], f_local[min(len(f_local) - 1, j + 2)]
    opt = minimize_scalar(lambda x: -power_at(lsim, float(x)), bounds=(lo, hi), method="bounded")
    boot_freq.append(float(opt.x))
boot_period_h = 24 / np.asarray(boot_freq)

summary = {
    "time_reference": "BJD_TCB days",
    "primary_n": len(primary), "sensitivity_n": len(df),
    "noisy_n_39": int(df.photometry_flag_noisy_data.sum()),
    "g_rejected_n_39": int(df.variability_flag_g_reject.sum()),
    "invalid_fit_record_g_reject_n": 1,
    "baseline_days": baseline, "frequency_step_per_day": dfreq,
    "frequency_grid_points": int(len(freq)), "search_ranges": period_ranges,
    "configuration": "Astropy LombScargle, fit_mean=True, center_data=True, nterms=1, normalization=standard; same times/grid/config for weighted and unweighted",
    "weighted_definition": "dy=sigma_mag; inverse variance weights 1/sigma_mag^2",
    "unweighted_definition": "dy=None",
    "literature_period_hours": literature_period_h,
    "literature_frequency_per_day": literature_frequency,
    "results_primary_37": {"weighted": peaks[("primary_37", True)], "unweighted": peaks[("primary_37", False)]},
    "max_weight": float(df.inverse_variance_weight.max()),
    "min_weight": float(df.inverse_variance_weight.min()),
    "weight_ratio_max_over_min": float(df.inverse_variance_weight.max() / df.inverse_variance_weight.min()),
    "sigma_mag_min": float(df.sigma_mag.min()), "sigma_mag_max": float(df.sigma_mag.max()),
    "unweighted_34h_bootstrap": {
        "replicates": 400, "seed": 92311,
        "noise_model": "best-fit sinusoid plus iid Gaussian residual noise with standard deviation equal to observed unweighted sinusoid-fit residual scatter; fixed observation times",
        "residual_std_mag": float(np.std(resid, ddof=3)),
        "period_hours_median": float(np.median(boot_period_h)),
        "period_hours_16_84_percentile": [float(np.percentile(boot_period_h, 16)), float(np.percentile(boot_period_h, 84))],
        "frequency_search_window_per_day": [float(f_local.min()), float(f_local.max())],
        "assumption_warning": "Conditional formal uncertainty only; it does not include correlated/transient variability or model misspecification."
    },
    "significance_note": "Baluev FAP is computed independently for each weighting and each searched frequency interval; weighted and unweighted FAPs are not directly interchangeable."
}
(TABLES / "lomb_scargle_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"results_primary_37": summary["results_primary_37"], "weight_diagnostics": {k: summary[k] for k in ("sigma_mag_min", "sigma_mag_max", "min_weight", "max_weight", "weight_ratio_max_over_min")}, "bootstrap": summary["unweighted_34h_bootstrap"]}, indent=2))
print("Wrote matched weighted/unweighted periodograms and comparison tables to", TABLES)
