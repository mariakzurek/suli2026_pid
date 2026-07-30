import sys
sys.path.append("../../scripts/")

import pandas as pd
import numpy as np
import os
import uproot
import awkward as ak
import common_functions as au

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages


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
        and one of delta_c, delta_c_total, C_difference, or stat_unc

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

        if "delta_c" in df.columns:
            sys_col = "delta_c"
        elif "delta_c_total" in df.columns:
            sys_col = "delta_c_total"
        elif "C_difference" in df.columns:
            sys_col = "C_difference"
        elif "stat_unc" in df.columns:
            sys_col = "stat_unc"
        else:
            raise ValueError(
                f"{file} has none of delta_c, delta_c_total, "
                f"C_difference, or stat_unc"
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

    for i in range(len(p_edges) - 1):
        for j in range(len(theta_edges) - 1):
            master.append(
                {
                    "p_lo": p_edges[i],
                    "p_hi": p_edges[i + 1],
                    "theta_lo": theta_edges[j],
                    "theta_hi": theta_edges[j + 1]
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

    # Ensure all systematic columns are numeric
    master[sys_cols] = master[sys_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Treat any value >= 9999 as missing
    master[sys_cols] = master[sys_cols].mask(
        master[sys_cols] >= 9999,
        np.nan
    )

    # Diagnostic: report any remaining large values
    print("\n=== Large systematic values (>=100) ===")
    for col in sys_cols:
        bad = master[master[col] >= 100]
        if not bad.empty:
            print(f"\nColumn: {col}")
            print(
                bad[
                    [
                        "p_lo",
                        "p_hi",
                        "theta_lo",
                        "theta_hi",
                        col
                    ]
                ].to_string(index=False)
            )

    # Compute quadrature ignoring NaNs
    master["quadrature"] = np.sqrt(
        np.nansum(
            np.square(master[sys_cols].to_numpy()),
            axis=1
        )
    )

    # Diagnostic: show any suspicious quadrature values
    bad_quad = master[master["quadrature"] >= 100]

    if not bad_quad.empty:
        print("\n=== Large quadrature values ===")
        print(
            bad_quad[
                [
                    "p_lo",
                    "p_hi",
                    "theta_lo",
                    "theta_hi",
                    "quadrature"
                ] + sys_cols
            ].to_string(index=False)
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
    p_range=None,
    value_col="quadrature",
    output_png="uncertainty_vs_momentum.png",
    title=None
):
    """
    Plot uncertainty vs momentum for a representative theta slice.
    (Original single-column version, kept for backwards compatibility.)
    """

    theta_lo, theta_hi = theta_range

    subset = master[
        (master.theta_lo >= theta_lo) &
        (master.theta_hi <= theta_hi)
    ].copy()

    if p_range is not None:
        p_lo_limit, p_hi_limit = p_range
        subset = subset[
            (subset.p_lo >= p_lo_limit) &
            (subset.p_hi <= p_hi_limit)
        ]

    if subset.empty:
        raise ValueError(
            f"No bins found with theta fully inside {theta_range}"
            + (f" and p fully inside {p_range}" if p_range is not None else "")
        )

    subset["p_center"] = (subset.p_lo + subset.p_hi) / 2

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
        linestyle=""
    )
    ax.set_xlabel("Momentum p")
    ax.set_ylabel(value_col + " (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(
        title if title is not None
        else f"{value_col} vs momentum (theta in [{theta_lo}, {theta_hi}])"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    return plot_data


def plot_uncertainty_vs_momentum_full(
    master,
    theta_range,
    p_range=None,
    output_png="uncertainty_vs_momentum.png",
    output_csv="uncertainty_vs_momentum.csv",
    title=None,
):
    """
    Plot EVERY individual systematic source together with the total
    quadrature sum, vs momentum, for a given theta window - with a
    legend identifying each source. Also writes a CSV with one row
    per momentum bin containing the momentum/theta range plus every
    systematic value and the quadrature that was actually plotted,
    and prints that table to the console.
    """

    theta_lo, theta_hi = theta_range

    subset = master[
        (master.theta_lo >= theta_lo) &
        (master.theta_hi <= theta_hi)
    ].copy()

    if p_range is not None:
        p_lo_limit, p_hi_limit = p_range
        subset = subset[
            (subset.p_lo >= p_lo_limit) &
            (subset.p_hi <= p_hi_limit)
        ]

    if subset.empty:
        raise ValueError(
            f"No bins found with theta fully inside {theta_range}"
            + (f" and p fully inside {p_range}" if p_range is not None else "")
        )

    subset["p_center"] = (subset.p_lo + subset.p_hi) / 2

    value_cols = [
        c for c in master.columns
        if c not in ["p_lo", "p_hi", "theta_lo", "theta_hi"]
    ]
    sys_cols = [c for c in value_cols if c != "quadrature"]

    # One row per momentum bin, averaging over whichever theta bins
    # fall inside theta_range (and p_range, if given).
    plot_data = (
        subset.groupby(["p_lo", "p_hi", "p_center"], as_index=False)[value_cols]
        .mean()
        .sort_values("p_center")
    )

    # Record which theta window this row represents, so the CSV is
    # self-describing (this is the window that was averaged over,
    # not necessarily the exact theta bin edges of any one source).
    plot_data.insert(2, "theta_lo", theta_lo)
    plot_data.insert(3, "theta_hi", theta_hi)

    # ---------------------------------------------
    # Plot every systematic + the total quadrature
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for col in sys_cols:
        ax.plot(
            plot_data["p_center"],
            plot_data[col],
            marker="o",
            linestyle="",
            alpha=0.6,
            label=col,
        )

    ax.plot(
        plot_data["p_center"],
        plot_data["quadrature"],
        marker="s",
        linestyle="",
        color="black",
        label="quadrature (total)",
    )

    ax.set_xlabel("Momentum p")
    ax.set_ylabel("Uncertainty (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title(
        title if title is not None
        else f"Systematic uncertainties vs momentum (theta in [{theta_lo}, {theta_hi}])"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    # ---------------------------------------------
    # Save + print
    # ---------------------------------------------
    plot_data.to_csv(output_csv, index=False)

    print(f"\n=== {title or output_png} ===")
    print(plot_data.to_string(index=False))
    print(f"Saved plot to {output_png}")
    print(f"Saved table to {output_csv}")

    return plot_data


def make_stat_uncertainty_pdf(
    csv_files,
    output_pdf="stat_uncertainty_plots.pdf",
    theta_range=None,
    p_range=None,
):
    """
    Create PDF plots of uncertainty for each CSV.
    """

    candidate_columns = [
        "stat_unc",
        "delta_c_total",
        "delta_c",
        "C_difference",
    ]

    with PdfPages(output_pdf) as pdf:

        for csv_file in csv_files:

            df = pd.read_csv(csv_file)

            filename = os.path.splitext(
                os.path.basename(csv_file)
            )[0]

            rename_map = {
                "p_low": "p_lo",
                "p_high": "p_hi",
                "theta_low": "theta_lo",
                "theta_high": "theta_hi",
            }

            df.rename(
                columns=rename_map,
                inplace=True
            )

            value_col = None

            for col in candidate_columns:
                if col in df.columns:
                    value_col = col
                    break

            if value_col is None:
                print(
                    f"Skipping {csv_file}: no uncertainty column found"
                )
                continue

            df[value_col] = pd.to_numeric(
                df[value_col],
                errors="coerce"
            )

            df.loc[
                df[value_col] >= 9999,
                value_col
            ] = np.nan

            if value_col == "stat_unc" and "threshold" in df.columns:

                # One point per threshold value, averaged across
                # whatever p/theta bins exist for that threshold.
                df = (
                    df
                    .dropna(subset=[value_col])
                    .groupby("threshold", as_index=False)[value_col]
                    .mean()
                    .sort_values("threshold")
                )

                if df.empty:
                    print(
                        f"Skipping {csv_file}: no valid threshold points"
                    )
                    continue

                fig, ax = plt.subplots(
                    figsize=(8, 6)
                )

                ax.plot(
                    df["threshold"],
                    df["stat_unc"],
                    marker="o",
                    linestyle=""
                )

                ax.set_xlabel(
                    "Threshold"
                )

                ax.set_ylabel(
                    "Statistical Uncertainty (%)"
                )

                ax.yaxis.set_major_formatter(
                    PercentFormatter(xmax=1.0)
                )

                ax.set_title(
                    f"{filename} stat uncertainty"
                )

                ax.grid(
                    True,
                    alpha=0.3
                )

                fig.tight_layout()

                pdf.savefig(fig)

                plt.close(fig)

                continue

            required = [
                "p_lo",
                "p_hi",
                "theta_lo",
                "theta_hi",
                value_col
            ]

            missing = [
                c for c in required
                if c not in df.columns
            ]

            if missing:
                print(
                    f"Skipping {csv_file}: missing {missing}"
                )
                continue

            if theta_range is not None:

                theta_lo, theta_hi = theta_range

                df = df[
                    (df.theta_lo >= theta_lo)
                    &
                    (df.theta_hi <= theta_hi)
                ]

            if p_range is not None:

                p_lo_limit, p_hi_limit = p_range

                df = df[
                    (df.p_lo >= p_lo_limit)
                    &
                    (df.p_hi <= p_hi_limit)
                ]

            if df.empty:
                print(
                    f"Skipping {csv_file}: no valid bins"
                )
                continue

            df["p_center"] = (
                df.p_lo + df.p_hi
            ) / 2

            plot_data = (
                df
                .dropna(subset=[value_col])
                .groupby(
                    [
                        "p_lo",
                        "p_hi",
                        "p_center"
                    ],
                    as_index=False
                )[value_col]
                .mean()
                .sort_values("p_center")
            )

            if plot_data.empty:
                print(
                    f"Skipping {csv_file}: no valid uncertainty points"
                )
                continue

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            ax.plot(
                plot_data["p_center"],
                plot_data[value_col],
                marker="o",
                linestyle=""
            )

            ax.set_xlabel(
                "Momentum p (GeV)"
            )

            ax.set_ylabel(
                f"{value_col} (%)"
            )

            ax.yaxis.set_major_formatter(
                PercentFormatter(xmax=1.0)
            )

            ax.set_title(
                f"{filename} {value_col}"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            fig.tight_layout()

            pdf.savefig(fig)

            plt.close(fig)

    print(
        f"Saved {output_pdf}"
    )


# =====================================================================
# Contamination with systematic-error override
# =====================================================================

def get_averaged_systematic_error(
    syst_df,
    p_lo, p_hi,
    theta_lo, theta_hi,
    value_col="quadrature",
):
    """
    Average every row of syst_df whose [p_lo,p_hi] x [theta_lo,theta_hi]
    bin overlaps the given (p_lo, p_hi, theta_lo, theta_hi) window.
    Returns np.nan if nothing overlaps.
    """

    overlap = syst_df[
        (syst_df.p_lo < p_hi) & (syst_df.p_hi > p_lo) &
        (syst_df.theta_lo < theta_hi) & (syst_df.theta_hi > theta_lo)
    ]

    if overlap.empty:
        return np.nan

    return overlap[value_col].mean()


def _count_true(mask):
    """
    Sum a boolean mask whether it's a pandas Series/ndarray or an
    awkward Array (pandas' .sum() and awkward's ak.sum() aren't
    interchangeable, so try one then fall back to the other).
    """
    try:
        return int(mask.sum())
    except AttributeError:
        return int(ak.sum(mask))


def compute_contamination(data, pid_col="mc_matching_pid", pid_value=321):
    """
    Contamination = fraction of events passing the selection cut whose
    mc_matching_pid != pid_value, out of all events passing the cut.
    `data` should already have the selection cut applied - this just
    computes the ratio. Works for pandas DataFrames and awkward
    Arrays/Records alike.

    Returns
    -------
    (val, err) : tuple(float, float)
        val is the contamination fraction; err is a binomial
        proportion error estimate (sqrt(val*(1-val)/n)). Note that
        compute_contamination_with_systematics discards this err and
        substitutes the systematic uncertainty instead.
    """

    total = len(data)

    if total == 0:
        return np.nan, np.nan

    wrong = _count_true(data[pid_col] != pid_value)

    val = wrong / total
    err = np.sqrt(val * (1 - val) / total)

    return val, err


def bins_from_master(master):
    """
    Turn a master table (output of make_systematic_quadrature_table)
    into a list of (p_lo, p_hi, theta_lo, theta_hi) bins - the full
    2D (p, theta) grid, one tuple per row.
    """
    return list(
        master[["p_lo", "p_hi", "theta_lo", "theta_hi"]]
        .itertuples(index=False, name=None)
    )


def p_bins_from_master(master, p_range=None):
    """
    Unique (p_lo, p_hi) momentum bins found in a master table, with
    theta collapsed out. Master tables have one row per (p, theta)
    bin, so the same p_lo/p_hi pair repeats across every theta slice;
    this collapses those down to one row per momentum bin, which is
    what you want when integrating over theta instead of binning by
    it as well.
    """
    p_bins = sorted(
        set(
            master[["p_lo", "p_hi"]]
            .itertuples(index=False, name=None)
        )
    )

    if p_range is not None:
        p_lo_limit, p_hi_limit = p_range
        p_bins = [
            (p_lo, p_hi) for p_lo, p_hi in p_bins
            if p_lo >= p_lo_limit and p_hi <= p_hi_limit
        ]

    return p_bins


def compute_contamination_vs_momentum(
    events_df,
    theta_range,
    p_range,
    syst_csv,
    syst_value_col="quadrature",
    num_bins=10,
    p_col="p",
    theta_col="theta",
    pid_col="pid",
    pid_value=321,
    mc_pid_col="mc_matching_pid",
    bdt_pass_col="bdt_pass",
    output_csv="contamination_vs_momentum.csv",
    output_png="contamination_vs_momentum.png",
    title="Contamination vs momentum",
):
    """
    Compute contamination vs momentum with theta integrated out.

    Contamination definition:

        C = N(pid==321 and mc_matching_pid!=321) / N(pid==321)

    after applying:
        - momentum bin
        - theta range
        - BDT selection

    Systematic uncertainties are taken from syst_csv:
        - find all systematic bins overlapping each momentum bin
        - average those uncertainties
        - use the result as the systematic uncertainty box

    Statistical uncertainties are calculated from the contamination
    definition above.
    """

    theta_lo, theta_hi = theta_range
    p_min, p_max = p_range

    syst_df = pd.read_csv(syst_csv)

    # Create uniform momentum bins
    p_edges = np.linspace(
        p_min,
        p_max,
        num_bins + 1
    )

    results = []

    for i in range(num_bins):

        p_lo = p_edges[i]
        p_hi = p_edges[i + 1]

        # Apply analysis cuts BEFORE contamination calculation
        cut = (
            (events_df[p_col] >= p_lo) &
            (events_df[p_col] < p_hi) &
            (events_df[theta_col] >= theta_lo) &
            (events_df[theta_col] < theta_hi) &
            (events_df[bdt_pass_col])
        )

        df_bin = events_df[cut]

        # -------------------------------
        # Contamination definition
        # -------------------------------

        temp = df_bin[df_bin[pid_col] == pid_value]

        a = len(
            temp[temp[mc_pid_col] != pid_value]
        )

        b = len(temp)

        r = 0.0
        rErr = 99.0

        if b != 0:
            r = a / b

        if a != 0:
            rErr = r * np.sqrt(
                (1 / a) + (1 / b)
            )

        # -------------------------------
        # Systematic uncertainty
        # -------------------------------

        overlap = syst_df[
            (syst_df["p_lo"] < p_hi) &
            (syst_df["p_hi"] > p_lo) &
            (syst_df["theta_lo"] < theta_hi) &
            (syst_df["theta_hi"] > theta_lo)
        ]

        if len(overlap) > 0:
            syst_err = overlap[syst_value_col].mean()
        else:
            syst_err = np.nan

        results.append(
            {
                "p_lo": p_lo,
                "p_hi": p_hi,
                "theta_lo": theta_lo,
                "theta_hi": theta_hi,
                "p_center": (p_lo + p_hi) / 2,
                "contamination": r,
                "stat_err": rErr,
                "syst_err": syst_err,
                "N_true": b,
                "N_contam": a,
            }
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        output_csv,
        index=False
    )

    print(f"\n=== {title} ===")
    print(result_df.to_string(index=False))
    print(f"Saved table to {output_csv}")


    # -------------------------------
    # Plot
    # -------------------------------

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    from matplotlib.patches import Rectangle

    # Systematic boxes
    # Convert relative systematic uncertainty into an absolute
    # contamination uncertainty:
    # syst_abs = contamination * syst_err
    for _, row in result_df.iterrows():

        if np.isnan(row["syst_err"]):
            continue

        syst_abs = row["contamination"] * row["syst_err"]

        rect = Rectangle(
            (
                row["p_lo"],
                row["contamination"] - syst_abs
            ),
            row["p_hi"] - row["p_lo"],
            2 * syst_abs,
            color="blue",
            alpha=0.3,
            linewidth=0,
        )

        ax.add_patch(rect)


    # Statistical errors
    ax.errorbar(
        result_df["p_center"],
        result_df["contamination"],
        yerr=result_df["stat_err"],
        marker="o",
        linestyle="none",
        color="black",
        ecolor="black",
        markerfacecolor="black",
        markeredgecolor="black",
        capsize=3,
        label="Statistical uncertainty",
    )


    ax.set_xlabel("Momentum p (GeV)")
    ax.set_ylabel("Contamination")
    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=150
    )

    plt.close(fig)

    print(f"Saved plot to {output_png}")

    return result_df

def compute_contamination_from_csv(
    contamination_csv,
    syst_csv,
    p_range=None,
    num_bins=10,
    syst_value_col="quadrature",
    use_csv_binning=False,
    output_csv="contamination_from_csv.csv",
    output_png="contamination_from_csv.png",
    title="Contamination vs momentum",
):
    """
    Compute contamination vs momentum from an existing contamination CSV.

    Input contamination CSV must contain:
        p_lo
        p_hi
        contamination_initial
        contamination_initial_err

    Parameters
    ----------
    contamination_csv : str
        CSV containing contamination values.

    syst_csv : str
        CSV containing systematic uncertainties.

    p_range : tuple
        (p_min, p_max) range for uniform bins.
        Ignored if use_csv_binning=True.

    num_bins : int
        Number of uniform momentum bins.
        Ignored if use_csv_binning=True.

    use_csv_binning : bool
        If True, use the original CSV binning instead of creating
        uniform momentum bins.

    Statistical uncertainty:
        stat_err = average(contamination_initial_err)

    Systematic uncertainty:
        syst_abs = contamination * syst_err
    """

    df = pd.read_csv(contamination_csv)

    required = [
        "p_lo",
        "p_hi",
        "contamination_initial",
        "contamination_initial_err"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in contamination CSV: {missing}"
        )

    syst_df = pd.read_csv(syst_csv)


    # -------------------------------------------------
    # Determine output bins
    # -------------------------------------------------

    if use_csv_binning:

        bins = (
            df[["p_lo", "p_hi"]]
            .drop_duplicates()
            .sort_values("p_lo")
            .values
        )

    else:

        if p_range is None:
            raise ValueError(
                "p_range must be provided when use_csv_binning=False"
            )

        p_min, p_max = p_range

        p_edges = np.linspace(
            p_min,
            p_max,
            num_bins + 1
        )

        bins = [
            (p_edges[i], p_edges[i+1])
            for i in range(num_bins)
        ]


    results = []


    # -------------------------------------------------
    # Loop over bins
    # -------------------------------------------------

    for p_lo, p_hi in bins:


        if use_csv_binning:

            # Exact CSV bin
            overlap = df[
                (df["p_lo"] == p_lo) &
                (df["p_hi"] == p_hi)
            ]

        else:

            # Average overlapping CSV bins
            overlap = df[
                (df["p_lo"] < p_hi) &
                (df["p_hi"] > p_lo)
            ]


        if len(overlap) > 0:

            contamination = overlap[
                "contamination_initial"
            ].mean()

            stat_err = overlap[
                "contamination_initial_err"
            ].mean()

        else:

            contamination = np.nan
            stat_err = np.nan


        # -------------------------------------------------
        # Systematic uncertainty
        # -------------------------------------------------

        syst_overlap = syst_df[
            (syst_df["p_lo"] < p_hi) &
            (syst_df["p_hi"] > p_lo)
        ]

        if len(syst_overlap) > 0:

            syst_err = syst_overlap[
                syst_value_col
            ].mean()

        else:

            syst_err = np.nan


        results.append(
            {
                "p_lo": p_lo,
                "p_hi": p_hi,
                "p_center": (p_lo + p_hi) / 2,
                "contamination": contamination,
                "stat_err": stat_err,
                "syst_err": syst_err,
                "N_overlap_bins": len(overlap),
            }
        )


    result_df = pd.DataFrame(results)


    result_df.to_csv(
        output_csv,
        index=False
    )


    print(f"\n=== {title} ===")
    print(result_df.to_string(index=False))
    print(f"Saved table to {output_csv}")


    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8,6)
    )

    from matplotlib.patches import Rectangle


    # Systematic boxes
    for _, row in result_df.iterrows():

        if (
            np.isnan(row["syst_err"]) or
            np.isnan(row["contamination"])
        ):
            continue


        syst_abs = (
            row["contamination"] *
            row["syst_err"]
        )


        rect = Rectangle(
            (
                row["p_lo"],
                row["contamination"] - syst_abs
            ),
            row["p_hi"] - row["p_lo"],
            2 * syst_abs,
            color="blue",
            alpha=0.3,
            linewidth=0
        )

        ax.add_patch(rect)


    # Statistical errors
    ax.errorbar(
        result_df["p_center"],
        result_df["contamination"],
        yerr=(result_df["stat_err"]/4),
        marker="o",
        linestyle="none",
        color="black",
        ecolor="black",
        capsize=3,
        label="Statistical uncertainty"
    )


    ax.set_xlabel(
        "Momentum p (GeV)"
    )

    ax.set_ylabel(
        "Contamination"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=150
    )

    plt.close(fig)

    print(
        f"Saved plot to {output_png}"
    )

    return result_df

# =====================================================================
# Driver
# =====================================================================

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

files_all = []
files_all.append("calibration_sensitivity.csv")
files_all.append("weighted_contamination_comparison_granular.csv")
files_all.append("threshold_sensitivity.csv")
files_all.append("per_bin_sweep_uncertianties.csv")
files_all.append("rich_contamination_binned.csv")
files_all.append("mx_width_sensitivity.csv")

make_stat_uncertainty_pdf(files_all)

master_Mc = make_systematic_quadrature_table(
    files_Mc,
    output_md="pngs/systematic_table_MC.md"
)
master_Mx = make_systematic_quadrature_table(
    files_Mx,
    output_md="pngs/systematic_table_MX.md"
)
master_RICH = make_systematic_quadrature_table(
    files_RICH,
    output_md="pngs/systematic_table_RICH.md"
)

# ---------------------------------------------------------------
# Plot every systematic source + total quadrature, and save the
# per-bin CSVs
# ---------------------------------------------------------------
plot_data_Mc = plot_uncertainty_vs_momentum_full(
    master_Mc,
    theta_range=(5, 15),
    p_range=None,
    output_png="pngs/uncertainty_vs_momentum_MC.png",
    output_csv="pngs/uncertainty_vs_momentum_MC.csv",
    title="Systematic Uncertainty vs Momentum, MC"
)
plot_data_Mx = plot_uncertainty_vs_momentum_full(
    master_Mx,
    theta_range=(5, 20),
    p_range=(2.75, 5),
    output_png="pngs/uncertainty_vs_momentum_MX.png",
    output_csv="pngs/uncertainty_vs_momentum_MX.csv",
    title="Systematic Uncertainty vs Momentum, MX"
)
plot_data_RICH = plot_uncertainty_vs_momentum_full(
    master_RICH,
    theta_range=(0, 20),
    p_range=(2.75, 5),
    output_png="pngs/uncertainty_vs_momentum_RICH.png",
    output_csv="pngs/uncertainty_vs_momentum_RICH.csv",
    title="Systematic Uncertainty vs Momentum, RICH"
)


cols = ["pid", "mc_matching_pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz", "bdt_pass", "rich_best_PID", "rich_RQ", "rich_best_ntot", "bdt_score"]
kinematics = ["Mx_eKX", "Mx_epiX", "Mx_epX", "Q2", "W", "y"]

width = 0.15
mass = 0.93

for kin in kinematics:
    cols.append(kin)

df = uproot.open("~/ML_Files/MC_scored/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")




# ---------------------------------------------------------------
contamination_Mc = compute_contamination_vs_momentum(
    df,
    p_range=(0.5, 5),
    theta_range=(5,15),
    syst_csv="pngs/uncertainty_vs_momentum_MC.csv",
    output_csv="pngs/contamination_with_syst_MC.csv",
    output_png="pngs/contamination_with_syst_MC.png",
    title="Contamination with total systematic error, MC",
)

contamination_Mx = compute_contamination_from_csv(
    contamination_csv="epiN_contamination_binned_PLOTTING.csv",
    syst_csv="pngs/uncertainty_vs_momentum_MX.csv",
    use_csv_binning=True,
    output_csv="pngs/contamination_MX_from_csv.csv",
    output_png="pngs/contamination_MX_from_csv.png",
    title="MX Contamination",
)


contamination_RICH = compute_contamination_from_csv(
    contamination_csv="rich_contamination_binned.csv",
    syst_csv="pngs/uncertainty_vs_momentum_RICH.csv",
    p_range=(2.75, 5),
    output_csv="pngs/contamination_with_syst_RICH.csv",
    output_png="pngs/contamination_with_syst_RICH.png",
    title="Contamination with total systematic error, RICH",
)

