from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'results'/'tables'/'validated_gaia_G_band_data.csv').sort_values('bjd_tcb')
fig,ax=plt.subplots(figsize=(10,5.5))
styles=[(~df.photometry_flag_noisy_data & ~df.variability_flag_g_reject,'tab:blue','Quality flags false'),(df.photometry_flag_noisy_data & ~df.variability_flag_g_reject,'tab:orange','Noisy flagged'),(df.variability_flag_g_reject,'tab:red','G-rejected (excluded from primary period analysis)')]
for mask,color,label in styles:
 if mask.any(): ax.errorbar(df.loc[mask,'bjd_tcb'],df.loc[mask,'g_mag_vega'],yerr=df.loc[mask,'sigma_mag'],fmt='o',ms=5,capsize=2,color=color,label=label,alpha=.85)
ax.invert_yaxis(); ax.set_xlabel('Observation time (BJD_TCB, days)'); ax.set_ylabel('Gaia G-band Vega magnitude')
ax.set_title('Gaia DR3 G-band light curve: 1eRASS J052113.8−520257')
ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
out=ROOT/'results'/'figures'/'gaia_G_band_lightcurve.png'; fig.savefig(out,dpi=180); print(out)
