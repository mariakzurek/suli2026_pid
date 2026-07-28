import pandas as pd
import numpy as np
import os


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



files = []

files.append("calibration_sensitivity.csv")
files.append("rich_contamination_binned.csv")
files.append("weighted_contamination_comparison.csv")
files.append("threshold_sensitivity.csv")
files.append("mx_width_sensitivity.csv")
files.append("per_bin_sweep_uncertianties.csv")


make_systematic_quadrature_table(
    files,
    output_md="systematic_table.md"
)