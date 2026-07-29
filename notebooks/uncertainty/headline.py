import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def make_systematic_quadrature_table(csv_files, output_md="systematic_table.md"):
    """
    Combine systematic drift tables with different binning.

    Parameters
    ----------
    csv_files : list
        List of CSV filenames.
        Each CSV must contain:
        p_lo/p_low, p_hi/p_high,
        theta_lo/theta_low, theta_hi/theta_high,
        and either relative_drift or stat_unc

    output_md : str
        Output markdown filename.
    """

    tables = []

    for file in csv_files:
        df = pd.read_csv(file)

        # ---------------------------------------------
        # Normalize bin column names
        # ---------------------------------------------

        rename_map = {
            "p_low": "p_lo",
            "p_high": "p_hi",
            "theta_low": "theta_lo",
            "theta_high": "theta_hi"
        }

        df.rename(
            columns=rename_map,
            inplace=True
        )

        required_cols = [
            "p_lo",
            "p_hi",
            "theta_lo",
            "theta_hi"
        ]

        missing = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{file} missing columns: {missing}"
            )


        # ---------------------------------------------
        # Determine systematic column
        # ---------------------------------------------

        if "relative_drift" in df.columns:
            sys_col = "relative_drift"

        elif "stat_unc" in df.columns:
            sys_col = "stat_unc"

        else:
            raise ValueError(
                f"{file} has neither relative_drift nor stat_unc"
            )


        name = os.path.splitext(
            os.path.basename(file)
        )[0]


        df = df[
            [
                "p_lo",
                "p_hi",
                "theta_lo",
                "theta_hi",
                sys_col
            ]
        ].copy()


        df.rename(
            columns={sys_col: name},
            inplace=True
        )


        tables.append(df)


    # -------------------------------------------------
    # Make master binning from all unique boundaries
    # -------------------------------------------------

    p_edges = sorted(
        set(
            [x for df in tables for x in df.p_lo] +
            [x for df in tables for x in df.p_hi]
        )
    )

    theta_edges = sorted(
        set(
            [x for df in tables for x in df.theta_lo] +
            [x for df in tables for x in df.theta_hi]
        )
    )


    master = []

    for i in range(len(p_edges)-1):
        for j in range(len(theta_edges)-1):

            master.append(
                {
                    "p_lo": p_edges[i],
                    "p_hi": p_edges[i+1],
                    "theta_lo": theta_edges[j],
                    "theta_hi": theta_edges[j+1]
                }
            )

    master = pd.DataFrame(master)


    # -------------------------------------------------
    # Match each source onto master bins
    # -------------------------------------------------

    for df in tables:

        source = [
            c for c in df.columns
            if c not in [
                "p_lo",
                "p_hi",
                "theta_lo",
                "theta_hi"
            ]
        ][0]


        values = []

        for _, row in master.iterrows():

            match = df[
                (df.p_lo <= row.p_lo) &
                (df.p_hi >= row.p_hi) &
                (df.theta_lo <= row.theta_lo) &
                (df.theta_hi >= row.theta_hi)
            ]


            if len(match):

                # Use the first matching bin
                # (assumes no ambiguous overlaps)
                values.append(
                    match.iloc[0][source]
                )

            else:
                values.append(np.nan)


        master[source] = values


    # -------------------------------------------------
    # Quadrature sum
    # -------------------------------------------------

    sys_cols = [
        c for c in master.columns
        if c not in [
            "p_lo",
            "p_hi",
            "theta_lo",
            "theta_hi"
        ]
    ]


    master["quadrature"] = np.sqrt(
        np.nansum(
            master[sys_cols].values**2,
            axis=1
        )
    )


    # -------------------------------------------------
    # Write markdown
    # -------------------------------------------------

    md_table = master.copy()

    # Replace missing systematic values with N/A for display
    md_table = md_table.replace(np.nan, "N/A")

    with open(output_md, "w") as f:
        f.write(
            md_table.to_markdown(index=False)
        )


    return master


def plot_uncertainty_vs_momentum(
    master,
    theta_range,
    value_col="quadrature",
    output_png="uncertainty_vs_momentum.png",
    title=None
):
    """
    Plot uncertainty vs momentum for a representative theta slice.

    Parameters
    ----------
    master : pd.DataFrame
        The table returned by make_systematic_quadrature_table.
    theta_range : tuple(float, float)
        (theta_lo, theta_hi) window. Only bins fully inside this
        window are used. If more than one theta bin per momentum
        bin falls in the window, their values are averaged so you
        get one point per momentum bin.
    value_col : str
        Which column to plot on the y-axis (default: "quadrature").
    output_png : str
        Output image filename.
    title : str, optional
        Plot title. If not given, an automatic title is generated
        from value_col and theta_range.

    Returns
    -------
    pd.DataFrame
        The per-momentum-bin data that was plotted.
    """


    theta_lo, theta_hi = theta_range

    subset = master[
        (master.theta_lo >= theta_lo) &
        (master.theta_hi <= theta_hi)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No bins found with theta fully inside {theta_range}"
        )

    subset["p_center"] = (subset.p_lo + subset.p_hi) / 2

    # Average across any theta bins that fall within the window,
    # so there's one representative point per momentum bin
    plot_data = (
        subset.groupby(["p_lo", "p_hi", "p_center"], as_index=False)[value_col]
        .mean()
        .sort_values("p_center")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        plot_data["p_center"],
        plot_data[value_col],
        marker="o",
        linestyle="-"
    )
    ax.set_xlabel("Momentum p")
    ax.set_ylabel(value_col)
    ax.set_title(
        title if title is not None
        else f"{value_col} vs momentum (theta in [{theta_lo}, {theta_hi}])"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    return plot_data


files_Mc = []
files_RICH = []
files_Mx = []

files_Mc.append("calibration_sensitivity.csv")
files_Mc.append("weighted_contamination_comparison_granular.csv")
files_Mx.append("calibration_sensitivity.csv")
files_Mx.append("weighted_contamination_comparison_granular.csv")
files_RICH.append("calibration_sensitivity.csv")
files_RICH.append("weighted_contamination_comparison_granular.csv")
files_Mc.append("threshold_sensitivity.csv")
files_Mx.append("threshold_sensitivity.csv")
files_RICH.append("threshold_sensitivity.csv")



files_Mc.append("per_bin_sweep_uncertianties.csv")
files_RICH.append("rich_contamination_binned.csv")
files_Mx.append("mx_width_sensitivity.csv")

# NOTE: `files` was undefined in the original script (bug) - it never
# actually referenced any of files_Mc / files_RICH / files_Mx. Also,
# "weighted_contamination_comparison.csv" is appended to files_Mc twice,
# which would double-count it in the quadrature sum. Using files_Mc
# below as a placeholder - swap in whichever list (or combination) is
# actually correct, and remove the duplicate append.
master_Mc = make_systematic_quadrature_table(
    files_Mc,
    output_md="systematic_table_MC.md"
)
master_Mx = make_systematic_quadrature_table(
    files_Mx,
    output_md="systematic_table_MX.md"
)
master_RICH = make_systematic_quadrature_table(
    files_RICH,
    output_md="systematic_table_RICH.md"
)

plot_uncertainty_vs_momentum(
    master_MC,
    theta_range=(5, 35),
    output_png="uncertainty_vs_momentum_MC.png",
    title="Systematic Uncertainty vs Momentum, MC"
)
plot_uncertainty_vs_momentum(
    master_MX,
    theta_range=(5, 20),
    output_png="uncertainty_vs_momentum_MX.png",
    title="Systematic Uncertainty vs Momentum, MX"
)
plot_uncertainty_vs_momentum(
    master_RICH,
    theta_range=(0, 20),
    output_png="uncertainty_vs_momentum_RICH.png",
    title="Systematic Uncertainty vs Momentum, RICH"
)