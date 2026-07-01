import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# (dataset, h_adj, GAT mean, GAT std, ELAS mean, ELAS std) — order matches tab:wss95 (top = highest h_adj)
rows = [
    ("Walker (2018)",   0.377, 0.718, 0.004, 0.705, 0.001),
    ("Santos (2018)",   0.333, 0.867, 0.006, 0.898, 0.004),
    ("Sep (2021)",      0.318, 0.216, 0.107, 0.464, 0.032),
    ("Lewowski (2021)", 0.247, 0.642, 0.012, 0.735, 0.002),
    ("Burska (2023)",   0.221, 0.593, 0.009, 0.803, 0.004),
    ("Lauper (2021)",   0.187, 0.534, 0.012, 0.782, 0.001),
    ("Nelson (2002)",   0.101, 0.248, 0.013, 0.461, 0.012),
    ("Dolinska (2022)", 0.098, 0.791, 0.017, 0.910, 0.000),
    ("Brouwer (2019)",  0.097, 0.852, 0.013, 0.930, 0.000),
    ("Muthu (2020)",    0.070, 0.311, 0.039, 0.416, 0.007),
    ("Leenaars (2020)", 0.002, 0.361, 0.003, 0.665, 0.001),
]

WIN  = "#1a9850"   # colourblind-safe green
LOSS = "#d73027"   # colourblind-safe red
GAT_C  = "#222222"
ELAS_C = "#888888"

n = len(rows)
ys = np.arange(n)[::-1]  # top row = first entry

fig, ax = plt.subplots(figsize=(11, 8.5))

for (name, hadj, g, gsd, e, esd), y in zip(rows, ys):
    delta = g - e
    col = WIN if delta > 0 else LOSS
    # connecting line (direction cue)
    ax.plot([e, g], [y, y], color=col, lw=3.2, zorder=1, solid_capstyle="round")
    # ELAS = open square, GAT = filled circle (distinguishable by SHAPE, not just colour)
    ax.errorbar(e, y, xerr=esd, fmt="s", ms=12, mfc="white", mec=ELAS_C,
                mew=2.2, ecolor=ELAS_C, elinewidth=1.6, capsize=4, zorder=2)
    ax.errorbar(g, y, xerr=gsd, fmt="o", ms=13, mfc=GAT_C, mec=GAT_C,
                ecolor=GAT_C, elinewidth=1.6, capsize=4, zorder=3)
    # delta value at right margin, sign-coloured
    ax.text(1.13, y, f"{delta:+.3f}", color=col, fontsize=14,
            fontweight="bold", va="center", ha="left")

# y labels: dataset + h_adj on second line
labels = [f"{r[0]}\n$h_{{adj}}$ = {r[1]:.3f}" for r in rows]
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=13)
ax.set_ylim(-0.7, n - 0.3)

ax.set_xlim(0, 1.12)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.tick_params(axis="x", labelsize=13)
ax.set_xlabel("WSS@95   (further right = fewer papers to screen, better)", fontsize=14)

# delta column header
ax.text(1.13, n - 0.15, r"$\Delta$", fontsize=15, fontweight="bold",
        va="center", ha="left")

ax.grid(axis="x", color="#dddddd", lw=1)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)

# legend by shape
legend = [
    Line2D([0],[0], marker="o", color="w", mfc=GAT_C, mec=GAT_C, ms=13, label="GAT (this work)"),
    Line2D([0],[0], marker="s", color="w", mfc="white", mec=ELAS_C, mew=2.2, ms=12, label="ELAS u4 baseline"),
    Line2D([0],[0], color=WIN,  lw=3.2, label="GAT better"),
    Line2D([0],[0], color=LOSS, lw=3.2, label="Baseline better"),
]
ax.legend(handles=legend, fontsize=12, loc="lower right", frameon=True,
          framealpha=0.95, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("outputs/wss95_dumbbell.pdf", bbox_inches="tight")
plt.savefig("outputs/wss95_dumbbell.png", dpi=150, bbox_inches="tight")
print("done")