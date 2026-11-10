from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
df = pd.read_csv(TABLES / "phase_folded_lightcurve.csv")
df = df.loc[df.fold_label == "unweighted_refined_peak_primary37"].copy()
period_h = float(df.adopted_period_days.iloc[0]) * 24
fig, ax = plt.subplots(figsize=(7.6, 4.5))
for mask, color, label in [
    (~df.photometry_flag_noisy_data & ~df.variability_flag_g_reject, "#245a81", "Unflagged"),
    (df.photometry_flag_noisy_data & ~df.variability_flag_g_reject, "#d58a2b", "Noisy flagged"),
    (df.variability_flag_g_reject, "#b44a3a", "G-rejected (shown; excluded from fit)"),
]:
    part = df.loc[mask]
    if not part.empty:
        ax.errorbar(part.phase, part.g_mag_vega, yerr=part.sigma_mag, fmt="o", ms=4, capsize=1.5, color=color, alpha=.85, label=label)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Phase")
ax.set_ylabel("Gaia G-band Vega magnitude")
ax.set_title(f"Exploratory phase fold at the unweighted peak (P = {period_h:.3f} h)")
ax.grid(alpha=.22)
ax.legend(fontsize=8)
fig.tight_layout()
out = FIGURES / "phase_folded_lightcurve.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
