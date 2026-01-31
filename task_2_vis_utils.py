import pandas as pd
import matplotlib.pyplot as plt


def plot_series_grid(
    dataframe: pd.DataFrame,
    metric_col: str,
    plot_col: str,
    color_col: str,
    smoothing_win: int = 0,
) -> None:
    """
    For each unique value in `plot_col` creates a separate figure.
    Within each figure plots `metric_col` over index (or time if index is datetime),
    split by `color_col`.

    If smoothing_win > 1:
    - applies rolling median
    - drops first and last `smoothing_win` points to avoid edge effects
    """
    df = dataframe.copy()

    for c in [metric_col, plot_col, color_col]:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in dataframe")

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col, plot_col, color_col])

    plot_vals = sorted(df[plot_col].astype(str).unique())

    for pv in plot_vals:
        sub = df[df[plot_col].astype(str) == pv].copy()

        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        for cv, g in sub.groupby(color_col, sort=True):
            g = g.sort_index()

            y = g[metric_col]

            if smoothing_win and smoothing_win > 1:
                y = (
                    y.rolling(window=int(smoothing_win), min_periods=smoothing_win)
                     .median()
                )

                # drop edge effects
                y = y.iloc[smoothing_win:-smoothing_win]
                g = g.iloc[smoothing_win:-smoothing_win]

            if len(y) == 0:
                continue

            ax.plot(g.index, y.values, label=str(cv), linewidth=1)

        ax.set_title(f"{plot_col} = {pv}")
        ax.set_xlabel("index (set datetime index for time)")
        ax.set_ylabel(metric_col)
        ax.legend(title=color_col, fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()



def plot_dist_grid(
    dataframe: pd.DataFrame,
    metric_col: str,
    plot_col: str,
    color_col: str,
    smoothing_win: int = 0,
    plot_type: str = "hist",   # "hist" | "box"
) -> None:
    """
    For each unique value in `plot_col` creates a separate figure.
    Within each figure shows distribution of `metric_col` split by `color_col`.

    plot_type:
      - "hist": overlapping histograms
      - "box" : boxplots (one box per color_col)

    smoothing_win:
      If > 0, trims first and last `smoothing_win` values inside each
      (plot_col, color_col) group after sorting by index.
    """
    df = dataframe.copy()

    for c in [metric_col, plot_col, color_col]:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in dataframe")

    if plot_type not in {"hist", "box"}:
        raise ValueError("plot_type must be 'hist' or 'box'")

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col, plot_col, color_col])

    plot_vals = sorted(df[plot_col].astype(str).unique())

    for pv in plot_vals:
        sub = df[df[plot_col].astype(str) == pv].copy()

        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        grouped = []
        labels = []

        for cv, g in sub.groupby(color_col, sort=True):
            g = g.sort_index()
            x = g[metric_col].dropna()

            if smoothing_win and smoothing_win > 0 and len(x) > 2 * smoothing_win:
                x = x.iloc[smoothing_win:-smoothing_win]

            if len(x) == 0:
                continue

            if plot_type == "hist":
                ax.hist(x.values, bins=30, alpha=0.5, label=str(cv))
            else:  # box
                grouped.append(x.values)
                labels.append(str(cv))

        if plot_type == "box" and grouped:
            ax.boxplot(grouped, labels=labels, showfliers=True)

        ax.set_title(f"{plot_col} = {pv}")
        ax.set_ylabel(metric_col)

        if plot_type == "hist":
            ax.set_xlabel(metric_col)
            ax.legend(title=color_col, fontsize=8, ncol=2)
        else:
            ax.set_xlabel(color_col)

        plt.tight_layout()
        plt.show()