from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
df = pd.read_csv(TABLES / "validated_gaia_G_band_data.csv")
summary = json.loads((TABLES / "lomb_scargle_comparison_summary.json").read_text(encoding="utf-8"))
periods = {
    "unweighted_refined_peak_primary37": summary["results_primary_37"]["unweighted"]["short_range_refined"]["refined_period_days"],
    "weighted_refined_peak_primary37": summary["results_primary_37"]["weighted"]["short_range_refined"]["refined_period_days"],
    "literature_exact_34h_exploratory": 34.0 / 24.0,
}
epoch = float(df.bjd_tcb.min())
frames = []
for label, period in periods.items():
    part = df.copy()
    part["fold_label"] = label
    part["adopted_period_days"] = float(period)
    part["reference_epoch_bjd_tcb"] = epoch
    part["phase"] = ((part.bjd_tcb - epoch) / period) % 1
    part["included_in_primary_sample"] = ~part.variability_flag_g_reject
    part["retained_primary_n"] = int((~df.variability_flag_g_reject).sum())
    part["g_rejected_excluded_primary_n"] = int(df.variability_flag_g_reject.sum())
    frames.append(part)
out = pd.concat(frames, ignore_index=True).sort_values(["fold_label", "phase"])
out.to_csv(TABLES / "phase_folded_lightcurve.csv", index=False)
print("Reference epoch (BJD_TCB):", f"{epoch:.9f}")
for label, period in periods.items():
    print(f"{label}: P={period:.12f} d ({period*24:.8f} h); retained=37, excluded G-rejected=2; noisy flags retained")
print("Wrote", TABLES / "phase_folded_lightcurve.csv")
