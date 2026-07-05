import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def intensity_color(intensity):
    """Return color based on TC intensity regime."""
    if intensity < 70:
        return "orange"
    elif intensity <= 90:
        return "red"
    else:
        return "magenta"


# ---------------------------------------------------
# Paths
# ---------------------------------------------------
online = False
base_path = "/rstor/jmayhall/" if online else "//uahdata/rstor/"

file_list = glob.glob(
    f"{base_path}cataloging/nc_process/shear_process/shear_process_all/*.npz"
)

ships_path = (
    f"{base_path}cataloging/nc_process/"
    "violin_plots/shear_process_all/ships_interp.txt"
)

ships_df = pd.read_csv(
    ships_path,
    sep="\t",
    index_col=0,
    parse_dates=True
)


# ---------------------------------------------------
# Build list of SHIPS observations that have NPZ files
# ---------------------------------------------------
valid_cases = []

for file in file_list:

    atcf_id = file[-41:-33]
    date = file[-32:-24]
    time = file[-23:-19]

    valid_cases.append(
        (
            atcf_id,
            pd.to_datetime(f"{date}{time}", format="%Y%m%d%H%M")
        )
    )

valid_cases = pd.DataFrame(
    valid_cases,
    columns=["atcf_id", "time"]
)


# ---------------------------------------------------
# Keep only observations used in composite analysis
# ---------------------------------------------------
ships = (
    ships_df.reset_index()
            .rename(columns={"index": "time"})
            .merge(valid_cases,
                   on=["time", "atcf_id"],
                   how="inner")
)


# ---------------------------------------------------
# Compute shear and keep Low Shear cases
# ---------------------------------------------------
ships["Shear"] = np.sqrt(
    ships["shear_u"]**2 +
    ships["shear_v"]**2
) * 0.514444

ships = ships[ships["Shear"] < 5]


# ---------------------------------------------------
# Split by basin
# ---------------------------------------------------
atl = ships[ships["atcf_id"].str.startswith("AL")]
ep = ships[ships["atcf_id"].str.startswith("EP")]

# ---------------------------------------------------
# Plot
# ---------------------------------------------------
bin_width = 10
bins = np.arange(-5, 185 + 5, bin_width)

fig, ax = plt.subplots(
    1, 2,
    figsize=(12, 4.5),
    sharey=True
)

fig.suptitle(
    r"Low Shear (<5 m s$^{{-1}}$) TC Intensity Distribution",
    fontsize=18
)
for axis, data, title in zip(
        ax,
        [atl, ep],
        ["Atlantic", "Eastern Pacific"]):
    axis.hist(
        data["max_winds"],
        bins=bins,
        edgecolor="black",
        linewidth=0.7
    )

    median = np.nanmedian(data["max_winds"])
    axis.axvline(median, linestyle="--", color="black", linewidth=3)
    axis.axvline(x=65, color='orange', linestyle='--', lw=3)
    axis.axvline(x=95, color='magenta', linestyle='--', lw=3)

    axis.set_xlim(20, 170)
    axis.set_xlabel("Maximum Wind (kt)")
    axis.set_title(title, fontsize=15)

    axis.grid(
        True,
        linestyle="--",
        linewidth=1,
        alpha=1
    )
    axis.set_xticks(np.arange(20, 161, 10))

    axis.set_axisbelow(True)

ax[0].set_ylabel("# of Images")

legend_lines = [
    Line2D(
        [0], [0],
        color=intensity_color(69),
        linestyle="--",
        linewidth=2,
        label="TS/HUR Transition"
    ),
    Line2D(
        [0], [0],
        color=intensity_color(91),
        linestyle="--",
        linewidth=2,
        label="HUR/MAJ HUR Transition"
    ),
    Line2D(
        [0], [0],
        color='black',
        linestyle="--",
        linewidth=2,
        label="Median Intensity"
    )
]


fig.legend(
    handles=legend_lines,
    loc="upper right",
    ncol=1,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.98, 1)
)

plt.tight_layout(rect=[0, 0, 0.98, 0.97])
plt.savefig('low_shear_dist.png', dpi=300, bbox_inches="tight")