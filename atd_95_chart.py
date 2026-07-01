import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# (dataset, h_adj, GAT mean, GAT std, ELAS mean, ELAS std) — order matches tab:atd
rows = [
    ("Walker (2018)",   0.377, 0.0683, 0.0012, 0.0708, 0.0008),
    ("Santos (2018)",   0.333, 0.0436, 0.0080, 0.0223, 0.0002),
    ("Sep (2021)",      0.318, 0.3281, 0.0192, 0.2023, 0.0041),
    ("Lewowski (2021)", 0.247, 0.0865, 0.0154, 0.0498, 0.0036),
    ("Burska (2023)",   0.221, 0.1443, 0.0021, 0.0463, 0.0005),
    ("Lauper (2021)",   0.187, 0.1141, 0.0034, 0.0531, 0.0002),
    ("Nelson (2002)",   0.101, 0.2918, 0.0349, 0.1937, 0.0015),
    ("Dolinska (2022)", 0.098, 0.0583, 0.0028, 0.0146, 0.0023),
    ("Brouwer (2019)",  0.097, 0.0327, 0.0025, 0.0055, 0.0001),
    ("Muthu (2020)",    0.070, 0.1932, 0.0072, 0.1952, 0.0010),
    ("Leenaars (2020)", 0.002, 0.2485, 0.0055, 0.0967, 0.0002),
]

WIN  = "#1a9850"
LOSS = "#d73027"
GAT_C  = "#222222"
ELAS_C = "#888888"

n = len(rows)
ys = np.arange(n)[::-1]

fig, ax = plt.subplots(figsize=(11, 8.5))

for (name, hadj, g, gsd, e, esd), y in zip(rows, ys):
    delta = g - e
    # ATD: LOWER is better -> GAT wins when g < e  (delta < 0)
    col = WIN if delta < 0 else LOSS
    ax.plot([e, g], [y, y], color=col, lw=3.2, zorder=1, solid_capstyle="round")
    ax.errorbar(e, y, xerr=esd, fmt="s", ms=12, mfc="white", mec=ELAS_C,
                mew=2.2, ecolor=ELAS_C, elinewidth=1.6, capsize=4, zorder=2)
    ax.errorbar(g, y, xerr=gsd, fmt="o", ms=13, mfc=GAT_C, mec=GAT_C,
                ecolor=GAT_C, elinewidth=1.6, capsize=4, zorder=3)
    ax.text(0.40, y, f"{delta:+.4f}", color=col, fontsize=13.5,
            fontweight="bold", va="center", ha="left")

labels = [f"{r[0]}\n$h_{{adj}}$ = {r[1]:.3f}" for r in rows]
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=13)
ax.set_ylim(-0.7, n - 0.3)

ax.set_xlim(0, 0.39)
ax.set_xticks(np.arange(0, 0.351, 0.05))
ax.tick_params(axis="x", labelsize=13)
ax.set_xlabel("ATD   (further left = relevant papers found earlier, better)", fontsize=14)

ax.text(0.40, n - 0.15, r"$\Delta$", fontsize=15, fontweight="bold",
        va="center", ha="left")

ax.grid(axis="x", color="#dddddd", lw=1)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)

legend = [
    Line2D([0],[0], marker="o", color="w", mfc=GAT_C, mec=GAT_C, ms=13, label="GAT (this work)"),
    Line2D([0],[0], marker="s", color="w", mfc="white", mec=ELAS_C, mew=2.2, ms=12, label="ELAS u4 baseline"),
    Line2D([0],[0], color=WIN,  lw=3.2, label="GAT better"),
    Line2D([0],[0], color=LOSS, lw=3.2, label="Baseline better"),
]
ax.legend(handles=legend, fontsize=12, loc="lower right", frameon=True,
          framealpha=0.95, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("outputs/atd_dumbbell.pdf", bbox_inches="tight")
plt.savefig("outputs/atd_dumbbell.png", dpi=150, bbox_inches="tight")
print("done")