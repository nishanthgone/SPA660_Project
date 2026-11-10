from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

df = pd.read_csv(TABLES / "lomb_scargle_unweighted.csv")
# Refined maximum and 1% level are the existing reported analysis outputs.
peak_frequency = 0.7074999332967404
peak_power = 0.5260875527822431
one_percent_level = 0.615879
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(df.frequency_per_day, df.power, color="#164a72", lw=0.8, label="Unweighted periodogram (37 observations)")
ax.axhline(one_percent_level, color="#b44a3a", ls="--", lw=1.2, label="1% false-alarm power level")
ax.scatter([peak_frequency], [peak_power], color="#d38323", edgecolor="black", zorder=4, label="Strongest refined peak")
ax.annotate(f"{peak_frequency:.4f} d$^{{-1}}\nP = {24/peak_frequency:.3f} h", (peak_frequency, peak_power), xytext=(12, 12), textcoords="offset points", fontsize=9, arrowprops={"arrowstyle":"->", "lw":0.8})
ax.set_xscale("log")
ax.set_xlim(0.02, 24)
ax.set_xlabel("Frequency (cycles per day)")
ax.set_ylabel("Lomb–Scargle standard power")
ax.set_title("Unweighted Gaia G-band Lomb–Scargle periodogram")
ax.grid(alpha=0.22)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
out = FIGURES / "lomb_scargle_periodogram.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
