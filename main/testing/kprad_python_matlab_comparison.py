# kprad_python_matlab_comparison.py
"""
Script runs the python version of KPRAD, compares it to the output of the MATLAB version
"""


import numpy as np
from kprad.main_script import main
import scipy.io
import matplotlib.pyplot as plt

_PRAD_COLOR = {
    "H": "tab:red",
    "He": "tab:olive",
    "C": "tab:blue",
    "Ne": "tab:green",
    "Ar": "tab:purple",
    "W": "tab:brown",
}
_SPECIES_COLOR = {
    "H": "C0",
    "He": "C4",
    "C": "C1",
    "Ne": "C2",
    "Ar": "C3",
    "W": "C5",
}

def _fmt_val(v, fmt=".2f"):
    return (
        "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)
    )

kprad_py = main(config_path="/Users/plh/Documents/python_code/kprad/configs/180016_SPI.yaml", show = False)
kprad_ml = scipy.io.loadmat("/Users/plh/Documents/python_code/kprad/matlab/Jeffs_kprad.mat")

tV = kprad_py['tV']
tV_m = kprad_ml['tV']

fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.5))

# ---- (0,0)  Ip / Iw ------------------------------------------------------------
ax = axes[0, 0]
ax.plot(tV, kprad_py["Ip"], "C0", linewidth=1.8, label=r"$I_p$")
ax.plot(tV, kprad_py["Iw"], "C2", linewidth=1.2, label=r"$I_w$")

# MATLAB
ax.plot(tV_m, kprad_ml['Ip'], color = 'red', linewidth = 1.8, label=r"$I_p$" + "MATLAB" )
ax.plot(tV_m, kprad_ml['Iw'], color = 'red', linewidth = 1.8, label=r"$I_w$" + "MATLAB" )

ax.set(xlabel="time [ms]", ylabel="I [MA]", title="Current")
ax.set_ylim(0, 1.65)
ax.set_xlim(0, 20)
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)



# ---- (0,1)  ne ------------------------------------------------------------
ax = axes[0, 1]
ax.plot(tV, kprad_py["dense"], "C0", linewidth=1.8, label=r"$n_e$ free")

ax.plot(tV_m, kprad_ml["dense"][::1].transpose(), "red", linewidth=1.0, label=r"$n_e$ free MATLAB")

ax.set(xlabel="time [ms]", ylabel=r"$n_e$ [cm$^{-3}$]", title="Electron density")
ax.legend(loc="best", fontsize=9)
ax.grid(True, which="both", alpha=0.3)


# ---- (0,2)  Te/Ti ------------------------------------------------------------
ax = axes[0, 2]
ax.semilogy(tV, kprad_py["Te"], "C0", linewidth=1.8, label=r"$T_e$")
ax.semilogy(tV, kprad_py["Ti"], "C2", linewidth=1.2, label=r"$T_i$")

ax.semilogy(tV_m, kprad_ml["Te"], "red", linewidth=1.0, label=r"$T_e$" + "MATLAB")
ax.semilogy(tV_m, kprad_ml["Ti"], "red", linewidth=1.0, label=r"$T_i$" + "MATLAB")

ax.set(xlabel="time [ms]", ylabel=r"$T_{e,i}$ [eV]", title="Temperature")
ax.legend(loc="best", fontsize=9)
ax.grid(True, which="both", alpha=0.3)


# ---- (1, 0)  Zeff/Zbar ------------------------------------------------------------
ax = axes[1, 0]
ax.plot(tV, kprad_py["Zbar"], "C4", linewidth=1.8, label=r"$\langle Z \rangle$")
ax.plot(tV, kprad_py["Zeff"], "C1", linewidth=1.2, label=r"$Z_\mathrm{eff}$")


ax.plot(tV_m, kprad_ml["Zbar"][::1].transpose(), "red", linewidth=1.0, label=r"$\langle Z \rangle$" + "MATLAB")
ax.plot(tV_m, kprad_ml["Zeff"], "red", linewidth=1.0, label=r"$Z_\mathrm{eff}$" + "MATLAB")

ax.set(
    xlabel="time [ms]",
    ylabel=r"$\langle Z \rangle,\; Z_\mathrm{eff}$",
    title="Mean ionization",
)
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)



# ---- (1, 1)  Thermal energy ------------------------------------------------------------
ax = axes[1, 1]
ax.plot(tV, kprad_py["Wthe"], "C0", linewidth=1.8, label=r"$W_{th,e}$")
ax.plot(tV, kprad_py["Wthi"], "C2", linewidth=1.2, label=r"$W_{th,i}$")
ax.plot(
    tV, kprad_py["Wthe"] + kprad_py["Wthi"], "k--", linewidth=1.0, label=r"$W_{th}$ total"
)


ax.plot(tV_m, kprad_ml["Wthe"], "red", linewidth=1.0, label=r"$W_{th,e}$"+ "MATLAB")
ax.plot(tV_m, kprad_ml["Wthi"], "red", linewidth=1.0, label=r"$W_{th,i}$"+ "MATLAB")
ax.plot(
    tV_m, kprad_ml["Wthe"] + kprad_ml["Wthi"], "r--", linewidth=1.0, label=r"$W_{th}$ total"+ "MATLAB")


ax.set(xlabel="time [ms]", ylabel=r"$W_{th}$ [MJ]", title="Thermal energy")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)


# ---- (1, 2)  radiated power ------------------------------------------------------------
ax = axes[1, 2]
Prad = kprad_py['Prad']
if Prad:
    total_prad = np.zeros(len(tV))
    for sym, p in Prad.items():
        total_prad += p
        ax.semilogy(
            tV,
            np.maximum(p, 1e-10),
            color=_PRAD_COLOR.get(sym, "tab:gray"),
            linewidth=1.5,
            label=sym,
        )
    ax.semilogy(
        tV, np.maximum(total_prad, 1e-10), "k--", linewidth=1.2, label="total"
    )


total_prad_ml = np.zeros(kprad_ml['Prad_H'].shape)
for key in ['H', "C", "Ne"]:
    total_prad_ml += np.maximum(kprad_ml[f"Prad_{key}"], 1e-10)
    ax.semilogy(
        tV_m,
        np.maximum(kprad_ml[f"Prad_{key}"][::1].transpose(), 1e-10),
        color='red',
        linewidth=1.0,
        label=f"{key} MATLAB",
    )
ax.semilogy(
    tV_m, np.maximum(total_prad_ml[::1].transpose(), 1e-10), "r--", linewidth=1.2, label="total MATLAB"
)


ax.set(
    xlabel="time [ms]",
    ylabel=r"$P_\mathrm{rad}$ [GW]",
    title="Radiated power",
    ylim=(1e-4, None),
)
ax.legend(loc="best", fontsize=9)
ax.grid(True, which="both", alpha=0.3)


# ---- (2,1)  injected particles ------------------------------------------------------------
"""
NOTE: The cummalitive python version is slower than MATLAB's version by dt / 2.0. If you subtract
by that amount, they will line up perfectly. 
"""
ax = axes[2, 0]
injection = kprad_py['injection']

ax2 = ax.twinx()
ax2.set_ylabel(r"Rate [$10^{20}$ atoms/ms]", fontsize=9, color="gray")
ax2.tick_params(axis="y", colors="gray", labelsize=8)
for sym, data in injection.items():
    color = _SPECIES_COLOR.get(sym, "C7")
    # Rate as a light shaded area (right axis)
    ax2.fill_between(tV, data["rate"], alpha=0.18, color=color)
    ax2.plot(tV, data["rate"], color=color, linewidth=1.2)
    # Cumulative on left axis
    total = float(data["cumulative"][-1])
    ax.plot(
        tV,
        data["cumulative"],
        linewidth=2.0,
        color=color,
        label=f"{sym}  ({total:.2f}×10²⁰)",
    )


ax2.plot(kprad_ml['tsplineV'][0], kprad_ml["NdotSPI"][0], color='red', linewidth=1.0)
ax2.plot(kprad_ml['tsplineV'][0], kprad_ml["NdotSPIint"][0], color='red', linewidth=1.0, label="MATLAB")


ax2.set(
    xlabel="time [ms]",
    ylabel=r"Delivered [$10^{20}$ atoms]",
    title="Injected particles vs time",
)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)







# ---- (2,2)  summary text panel ------------------------------------------------------------
ax = axes[2, 2]
ax.axis("off")
s = kprad_py or {}
m = kprad_ml
tTQ = kprad_py.get("tTQ", float("nan"))
tCQ = kprad_py.get("tCQ", float("nan"))
Pradtot = s.get("Pradtot")
lines = [
    ("Simulation Summary    PYTHON   | MATLAB", True),
    ("─" * 46, False),
    (f"t_TQ                = {_fmt_val(tTQ)}     | {_fmt_val(m['tTQ'].item())} ms", False),
    (f"t_CQ                = {_fmt_val(tCQ)}    | {_fmt_val(m['tCQ'].item())} ms", False),
    (
        f"E_rad               = {(_fmt_val(Pradtot*1e3, '.0f') if Pradtot else 'N/A')}     | {_fmt_val(m['Pradtot'].item())} kJ",
        False,
    ),
    ("─" * 46, False),
    (f"I_p,0               = {_fmt_val(s.get('Ip0'))}     | {_fmt_val(m['Ip0'].item())} MA", False),
    (f"T_e,0               = {_fmt_val(s.get('Te0_keV'))}     | {_fmt_val(m['Te0'].item())} keV", False),
    (f"first light         = {_fmt_val(s.get('tfirst'))}     | {_fmt_val(m['tfirst'].item())} ms", False),
    #(f"E_φ (mid-CQ).       = {_fmt_val(s.get('Ephi_CQ'))} | {_fmt_val(m['Ei_C'].item())} V/m", False),
    (f"n_e,crit            = {_fmt_val(s.get('densecrit'), '.2e')} | {_fmt_val(m['densecrit'].item(), '.2e')} cm⁻³", False),
]
text = "\n".join(t for t, _ in lines)
ax.text(
    0.08,
    0.95,
    text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5dc", alpha=0.7),
)

fig.suptitle("KPRAD Python/MATLAB Simulation Comparison", fontsize=14, y=0.998)
fig.tight_layout(rect=[0, 0.0, 1, 0.996])  # type: ignore
