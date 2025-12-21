import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from typing import Optional, Sequence, Tuple

axis_labels = defaultdict(lambda: "No text available")
axis_labels.update(
    {
        "pred_set_size": "Prediction set size",
        "target_in_pred_set": "Target in prediction set",
        "target_prob": "Predicted probability of target",
        "correct": "Correct prediction",
        "step_idx": "Step index",
        "annotation_id": "Annotation ID",
        "method": "Method",
        "seed": "Random seed",
        "risk_level": "Target risk level (α)",
        "fp_rate_per_task": "Empirical risk (FPR per task)",
        "pred_set_avg": "Average prediction set size",
        "accuracy": "Accuracy",
    }
)

plt.rcParams['axes.labelsize'] = 14

def nonconformity_histogram(scores, threshold, naive_threshold=None, threshold2=None, naive_threshold2=None):
    plt.hist(scores, bins=30, edgecolor='k', linewidth=1)
    plt.axvline(
        x=threshold, linestyle='--', color='r', label=r'Quantile value ({threshold:.5f}), $\alpha=0.1$'.format(threshold=threshold),
    )
    if naive_threshold is not None:
        plt.axvline(
            x=naive_threshold, linestyle='--', color='g', label=r'Naive quantile value ({naive_threshold:.5f}), $\alpha=0.1$'.format(naive_threshold=naive_threshold),
        )
    if threshold2 is not None:
        plt.axvline(
            x=threshold2, linestyle=':', color='r', label=r'Quantile value ({threshold2:.5f}), $\alpha=0.2$'.format(threshold2=threshold2),
        )
    if naive_threshold2 is not None:
        plt.axvline(
            x=naive_threshold2, linestyle=':', color='g', label=r'Naive quantile value ({naive_threshold2:.5f}), $\alpha=0.2$'.format(naive_threshold2=naive_threshold2),
        )
    plt.title(
        'Histogram of non-conformity scores in the calibration set'
    )
    plt.xlabel('Non-conformity score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    print('A good predictor should have low non-conformity scores, concentrated at the left side of the figure')


def accuracy_vs_length(df, correct_col="correct", ann_col="annotation_id"):
    # Per-annotation stats
    ann_stats = (
        df.groupby(ann_col)[correct_col]
        .agg(step_acc="mean", task_acc="min", length="size")
        .reset_index()
    )
    # Aggregate by length
    len_stats = (
        ann_stats.groupby("length")
        .agg(
            step_acc_mean=("step_acc", "mean"),
            task_acc_mean=("task_acc", "mean"),
            n_annotations=("length", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots()
    ax.plot(len_stats["length"], len_stats["step_acc_mean"], marker="o", label="Avg step acc")
    ax.plot(len_stats["length"], len_stats["task_acc_mean"], marker="s", label="Task acc (all steps correct)")
    ax.set_xlabel("Annotation length (number of steps)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.show()
    return len_stats




def coverage_vs_set_size(df, alpha_line: float = 0.9, method_col: str = 'method'):
    """
    Plot average coverage (target in pred_set) versus pred_set_size.
    If method_col is provided and present in df, draws one curve per method.
    """

    palette = "tab10"
    order = list(df['method'].unique())

    group_cols = ["pred_set_size"]
    hue = None
    if method_col and method_col in df.columns:
        group_cols = [method_col, "seed"] + group_cols
        hue = method_col

    agg = (
        df.groupby(group_cols, observed=True)["covered"]
        .mean()
        .reset_index()
        .rename(columns={"covered": "coverage"})
    )
    # aggregate over seeds
    if hue:
        methods = df[method_col].unique()
        sizes = np.sort(df["pred_set_size"].unique())
        idx = pd.MultiIndex.from_product([methods, sizes], names=[method_col, "pred_set_size"])
        agg = (
            agg.groupby([method_col, "pred_set_size"], observed=True)["coverage"]
               .mean()
               .reset_index()
        )

    # Left: coverage vs set size; Right: pred set size distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.lineplot(data=agg, x="pred_set_size", y="coverage", hue=hue, marker="o", ax=axes[0], palette=palette, hue_order=order)
    min_coverage = agg["coverage"].min()
    # axes[0].set_ylim(max(min_coverage - 0.1, 0), 1)
    axes[0].set_xlabel("Prediction set size")
    axes[0].set_ylabel("Coverage")
    # axes[0].set_title("Coverage vs prediction set size" + (f" by {method_col}" if hue else ""))
    if alpha_line is not None:
        axes[0].axhline(alpha_line, linestyle="--", color="gray", alpha=0.7, label=f"alpha={alpha_line}")
        if hue:
            axes[0].legend(title=hue)
        else:
            axes[0].legend()

    # Distribution plot (use barplot for clearer method colors)
    if hue:
        count_df = (
            df.groupby([method_col, "seed", "pred_set_size"], observed=True)
            .size()
            .reset_index(name="count")
        ).groupby([method_col, "pred_set_size"], observed=True)["count"].mean().reset_index()
        sns.barplot(data=count_df, x="pred_set_size", y="count", hue=method_col, ax=axes[1], palette=palette, hue_order=order)
        axes[1].legend(title=hue)
    else:
        sns.histplot(data=df, x="pred_set_size", binwidth=1, discrete=True, ax=axes[1], color="steelblue")
    axes[1].set_xlabel("Prediction set size")
    axes[1].set_ylabel("Count")
    # axes[1].set_title("Prediction set size distribution")

    plt.tight_layout()
    plt.show()



########
# def ask_rate_vs_target_prob(df):
#     df = df.copy()
#     bins = [0.0, 0.2, 0.4, 0.7, 0.9, 1.0] 
#     df["ask"] = df["pred_set_size"] > 1
#     df["bin"] = pd.cut(df["target_prob"], bins=4,include_lowest=True,right=True,)

#     ask_rate = df.groupby("bin")["ask"].mean()

#     # plt.figure(figsize=(5,4))

#     ask_rate.plot(marker="o")
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Ask rate")
#     plt.title("Ask probability vs target_prob bin")
#     plt.tight_layout()




def a_in_pred_set(df, method_col: str = "method", iter_col: str = "seed"):
    """
    Left: P(A in pred_set | Y != A) vs pred_set_size (FPR curve).
    Right: overall class-conditional coverage P(A in pred_set | Y = A) per method.
    """
    df = df.copy()
    df["A_in_set"] = df["pred_set"].apply(lambda s: "A" in s )
    no_A = df[df['target_label'] != 'A']

    order = [method for method in df['method'].unique()]
 
    # 1) per-(method, seed, size) mean for A-in-set
    group_cols = [method_col, iter_col, "pred_set_size"]
    per_seed = (
        no_A.groupby(group_cols, observed=True)
        .agg(
            p_A_in_set=("A_in_set", "mean"),
        )
        .reset_index()
    )
    # 2) average over seeds
    agg = (
        per_seed.groupby([method_col, "pred_set_size"], observed=True)
        .agg(p_A_in_set=("p_A_in_set", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    sns.lineplot(
        data=agg,
        x="pred_set_size",
        y="p_A_in_set",
        palette="tab10",
        hue=method_col,
        hue_order=order,
        marker="o",
        ax=axes[0],
    )
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Class A FPR - P(A in pred set | Y != A)")
    axes[0].set_xlabel("Prediction set size")
   
    only_a = df[df['target_label'] == 'A']
    coverage = (
        only_a.groupby([method_col, iter_col], observed=True)["A_in_set"]
        .mean()
        .reset_index(name="coverage")
    )
    coverage_agg = (
        coverage.groupby(method_col, observed=True)["coverage"]
        .mean()
        .reset_index()
    )
    coverage_agg[method_col] = pd.Categorical(coverage_agg[method_col], categories=order, ordered=True)
    coverage_agg = coverage_agg.sort_values(method_col)

    sns.barplot(
        data=coverage_agg,
        x=method_col,
        y="coverage",
        order=order,
        palette="tab10",
        hue=method_col,
        hue_order=order,
        ax=axes[1],
    )
    axes[1].axhline(0.9, linestyle="--", color="gray", alpha=0.7, label="Target 0.9")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Class A coverage - P(A in pred set | Y = A)")
    axes[1].set_xlabel("Method")
    axes[1].tick_params(axis="x", rotation=15)
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].text(
            p.get_x() + p.get_width() / 2,
            min(height + 0.03, 1.02),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].legend()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title=method_col)

    plt.tight_layout()
    plt.show()




def failure_rate_vs_step_length(df, ann_col: str = "annotation_id", correct_col: str = "correct"):
    """
    Plot failure rate versus step index (within an annotation), stratified by pred_set_size.
    Assumes df has a 'pred_set_size' column and a per-row 'step_id' or relative index.
    """

    df = df.copy()
    
    df["fail"] = ~df[correct_col].astype(bool)
    agg = (
        df.groupby(["step_idx", "pred_set_size"])["fail"]
        .mean()
        .reset_index()
        .rename(columns={"fail": "failure_rate"})
    )
   
    sns.lineplot(data=agg, x="step_idx", y="failure_rate", hue="pred_set_size", marker="o")
    plt.xlabel("Step index")
    plt.ylabel("Failure rate")
    plt.title("Failure rate vs step index by pred_set_size")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def set_size_vs_task_length(df, ann_col: str = "annotation_id", method_col: str = "method", alpha: float = 0.9):
    """
    Plot average pred_set_size versus step index (within an annotation), optionally split by method.
    Right panel shows conditional coverage versus step index.
    """

    df = df.copy()
    group_cols = ["step_idx"]
    hue = None
    if method_col and method_col in df.columns:
        group_cols.insert(0, method_col)
        hue = method_col

    agg_size = (
        df.groupby(group_cols, observed=True)["pred_set_size"]
        .mean()
        .reset_index()
        .rename(columns={"pred_set_size": "avg_pred_set_size"})
    )
    agg_cov = (
        df.groupby(group_cols, observed=True)["covered"]
        .mean()
        .reset_index()
        .rename(columns={"covered": "coverage"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.lineplot(data=agg_size, x="step_idx", y="avg_pred_set_size", hue=hue, marker="o", ax=axes[0])
    axes[0].set_xlabel("Step index")
    axes[0].set_ylabel("Average pred set size")
    axes[0].set_title("Avg pred set size vs step index")

    sns.lineplot(data=agg_cov, x="step_idx", y="coverage", hue=hue, marker="o", ax=axes[1], legend=True)
    axes[1].set_xlabel("Step index")
    axes[1].set_ylabel("Avg Coverage")
    axes[1].set_ylim(agg_cov["coverage"].min()-0.05, 1.05)
    axes[1].set_title("Coverage vs step index" )
    axes[1].axhline(alpha, linestyle="--", color="gray", alpha=0.7, label=f"Target coverage ({alpha})")

    if hue:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, title=hue)
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels, title=hue)

    plt.tight_layout()
    plt.show()


def coverage_boxplot(df_all, iter_col: str = "seed", method_col: str = "method", alpha: float = 0.9):
    """
    Box plots by method:
      - coverage per iteration (target in pred_set)
      - average pred_set_size per iteration
    """

    group_df = df_all.groupby([method_col, iter_col], observed=True)
    cov = (
        group_df["covered"]
        .mean()
        .reset_index(name="coverage")
    )
    size = (
        group_df["pred_set_size"]
        .mean()
        .reset_index(name="pred_set_size")
    )
    # dummy column so all boxes sit at one x position; color by method via hue
    cov["dummy"] = "all"
    size["dummy"] = "all"
    
    fig, axes = plt.subplots(1, 2,)

    sns.boxplot(data=cov, x="dummy", y="coverage", hue=method_col, ax=axes[0])
    lower_bound = min(max(0, cov["coverage"].min() - 0.05), alpha - 0.01)
    axes[0].set_ylim(lower_bound, 1)
    axes[0].hlines(
        y=alpha,
        xmin=-0.5,
        xmax=0.5,
        colors="r",
        linestyles="--",
        label=f"Target coverage ({alpha})",
    )
    axes[0].set_ylabel("Coverage")
    axes[0].set_xlabel("")
    axes[0].set_xticks([])
    

    sns.boxplot(data=size, x="dummy", y="pred_set_size", hue=method_col, ax=axes[1])
    axes[1].set_ylabel("Prediction set size")
    axes[1].set_xlabel("")
    axes[1].set_xticks([])


    # Move a single legend underneath the plots
    handles, labels = axes[0].get_legend_handles_labels()
    if axes[0].legend_:
        axes[0].legend_.remove()
    if axes[1].legend_:
        axes[1].legend_.remove()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=max(1, min(len(labels), 4)), title="Method")
    fig.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    plt.show()


def risk_plot(
    metrics_df: pd.DataFrame,
    risk_col="risk_level",   # target risk α
    fp_col="fp_rate_per_task",     # empirical risk: FP per task
    method_col="method",           # optional
):
    df = metrics_df.copy()
    df["target_risk"] = df[risk_col]
    df["emp_risk"] = df[fp_col]

    sns.lineplot(
        data=df.sort_values(["target_risk", 'fp_rate_per_task']),
        x="target_risk",
        y="emp_risk",
        hue=method_col if method_col in df.columns else None,
        marker="o",
    )
    plt.plot([0, 1], [0, 1], "k--", label="y = x")
    xylim = max(df["target_risk"].max(), df["emp_risk"].max())
    plt.xlim(0, xylim + 0.05)
    plt.ylim(0, xylim + 0.05)
    plt.xlabel("Target risk α")
    plt.ylabel("Empirical risk (FPR per task)")
    # plt.title("Empirical Risk vs Target Risk")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


def risk_and_fpr_plots(
    metrics_df: pd.DataFrame,
    target_risk_col: str = "risk_level",
    emp_risk_col: str = "fp_rate_per_task",
    size_col: str = "pred_set_avg",
    method_col: str = "method",
):
    """
    Two-panel plot:
      - Left: avg prediction set size vs empirical risk
      - Right: empirical coverage vs average prediction set size
    """
    df = metrics_df.copy()
    hue = method_col if method_col in df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  
    # Left: avg pred set size vs emp risk
    sns.lineplot(
        data=df.sort_values([emp_risk_col, method_col]),
        x=emp_risk_col,
        y=size_col,
        hue=hue,
        marker="o",
        ax=axes[0],
        legend=False,
    )
    
    axes[0].set_xlabel("Empirical risk (FPR per task)")
    axes[0].set_ylabel("Prediction set size")
    # axes[0].set_title("Pred set size vs Empirical Risk")
    axes[0].grid(alpha=0.3)

    # Right: empirical coverage vs emp risk
    sns.lineplot(
        data=df.sort_values([emp_risk_col, "target_in_pred_set"]),
        x=emp_risk_col,
        y="target_in_pred_set",
        hue=method_col if method_col in df.columns else None,
        marker="o",
    )


    # plt.title("FPR for class A vs coverage")
    # axes[1].set_xlabel(r"FPR for A ($Pr(\hat{Y}=A | Y\neq A)$)")
    axes[1].set_xlabel("Empirical risk (FPR per task)")
    axes[1].set_ylabel("Empirical coverage")
    # axes[1].set_title("FPR for class A vs coverage")

    if hue:
        handles, labels = axes[1].get_legend_handles_labels()
        axes[0].legend(handles, labels, title=method_col)
        axes[1].legend(title=method_col)
    else:
        axes[0].legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def conditional_risk_by_step(
    df: pd.DataFrame,
    step_col: str = "step_idx",
    method_col: str = "method",
    pred_col: str = "pred_label",
    true_col: str = "target_label",
    pos_label: str = "A",
    target_alpha: float | None = None,
):
    """
    Bar plot of empirical risk (FPR for pos_label) by difficulty group (step index).
    If method_col is present, produces grouped bars per method.
    """
    df = df.copy()
    df["fp"] = (df[pred_col] == pos_label) & (df[true_col] != pos_label)
    df["neg"] = df[true_col] != pos_label

    group_cols = [step_col]
    if method_col and method_col in df.columns:
        group_cols.insert(0, method_col)

    agg = (
        df.groupby(group_cols, observed=True)
        .apply(lambda x: x["fp"].sum() / x["neg"].sum() if x["neg"].sum() > 0 else np.nan)
        .reset_index(name="risk")
    )

    plt.figure(figsize=(6, 4))
    sns.barplot(data=agg, x=step_col, y="risk", hue=method_col if method_col in agg.columns else None, palette="tab10")
    if target_alpha is not None:
        plt.axhline(target_alpha, linestyle="--", color="gray", alpha=0.7, label=f"alpha={target_alpha}")
    plt.ylabel(f"Empirical risk (P(pred={pos_label} | true≠{pos_label}))")
    plt.xlabel("Step index / difficulty group")
    plt.title("Conditional risk by difficulty group")
    plt.tight_layout()
    plt.show()

 

def quick_plot(metrics_df: pd.DataFrame,x='fp_rate_per_task',y='risk_level', method_col="method"):
    sns.lineplot(
        data=metrics_df.sort_values([x, y]),
        x=x,
        y=y,
        hue=method_col if method_col in metrics_df.columns else None,
        marker="o",
    )
    
   
    plt.xlabel(axis_labels[x])
    plt.ylabel(axis_labels[y])
    # plt.title("Empirical Risk vs Target Risk")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()



def plot_episode_reliability_multi(
    df: pd.DataFrame,
    methods: Optional[Sequence[str]] = None,
    risk_level: Optional[float] = None,
    threshold_alpha: Optional[float] = None,
    seed: Optional[int] = None,
    n_bins: int = 10,
    min_episodes_in_bin: int = 5,
    conf_col: str = "pred_prob",
    correct_col: str = "correct",
    annotation_col: str = "annotation_id",
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an episode-level reliability diagram with one curve per method.

    Episode definitions (customize inside if needed):
    - Episode confidence c_i: min(pred_prob over steps in that episode).
    - Episode success y_i^ep: 1 if all steps are correct, else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
        ['annotation_id', 'pred_prob', 'correct', 'method',
         'risk_level', 'threshold_alpha', 'seed'].
    methods : list of str or None
        Which methods to plot. If None, uses all unique methods in df
        after filtering by risk_level / threshold_alpha / seed.
    risk_level : float, optional
        Filter rows by risk_level (exact match).
    threshold_alpha : float, optional
        Filter rows by threshold_alpha (exact match).
    seed : int, optional
        Filter rows by seed (exact match).
    n_bins : int
        Number of confidence bins.
    min_episodes_in_bin : int
        Minimum episodes per bin per method to include that bin.
    conf_col : str
        Column name for per-step confidence (default: 'pred_prob').
    correct_col : str
        Column name for per-step correctness flag (default: 'correct').
    annotation_col : str
        Column identifying episodes (trajectories), e.g. 'annotation_id'.
    title : str, optional
        Plot title. If None, a default is constructed.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    df_filt = df.copy()

    # Global filters
    if risk_level is not None:
        df_filt = df_filt[df_filt["risk_level"] == risk_level]

    if threshold_alpha is not None:
        df_filt = df_filt[df_filt["threshold_alpha"] == threshold_alpha]

    if seed is not None:
        df_filt = df_filt[df_filt["seed"] == seed]

    # Determine which methods to plot
    methods_to_plot = sorted(df_filt["method"].unique().tolist())
    
    # Set up bin edges in [0,1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    fig, ax = plt.subplots()

    # Use different markers / linestyles to distinguish methods
    markers = ["o", "s", "D", "^", "v", "x", "*", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]
    marker_cycle = 0
    linestyle_cycle = 0

    for method_name in methods_to_plot:
        df_m = df_filt[df_filt["method"] == method_name]
        if df_m.empty:
            continue

        # --- Aggregate to episode level for this method ---

        # Episode confidence: min(pred_prob) over steps
        ep_conf = df_m.groupby(annotation_col)[conf_col].min()

        # Episode success: all steps correct -> 1, else 0
        ep_success = df_m.groupby(annotation_col)[correct_col].all().astype(int)

        # Align to be safe
        ep_conf, ep_success = ep_conf.align(ep_success, join="inner")

        conf_values = ep_conf.to_numpy()
        success_values = ep_success.to_numpy()

        # Bin by confidence
        bin_idx = np.digitize(conf_values, bin_edges, right=False) - 1  # 0..n_bins-1

        bin_centers = []
        bin_empirical_success = []
        bin_counts = []

        for b in range(n_bins):
            in_bin = bin_idx == b
            n_in_bin = in_bin.sum()
            if n_in_bin < min_episodes_in_bin:
                continue

            avg_conf = float(conf_values[in_bin].mean())
            avg_success = float(success_values[in_bin].mean())

            bin_centers.append(avg_conf)
            bin_empirical_success.append(avg_success)
            bin_counts.append(n_in_bin)

        if not bin_centers:
            # Nothing to plot for this method with the min_episodes_in_bin constraint
            continue

        # Pick marker + linestyle
        marker = markers[marker_cycle % len(markers)]
        linestyle = linestyles[linestyle_cycle % len(linestyles)]
        marker_cycle += 1
        if marker_cycle % len(markers) == 0:
            linestyle_cycle += 1

        ax.plot(
            bin_centers,
            bin_empirical_success,
            marker=marker,
            linestyle=linestyle,
            label=f"{method_name}",
        )

    # Add perfect calibration diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted task (episode) confidence")
    ax.set_ylabel("Empirical task (episode) success frequency")

    if title is None:
        pieces = ["Episode-level Reliability"]
        if risk_level is not None:
            pieces.append(f"risk={risk_level}")
        if threshold_alpha is not None:
            pieces.append(f"α={threshold_alpha}")
        if seed is not None:
            pieces.append(f"seed={seed}")
        title = " | ".join(pieces)

   
    ax.legend(title="method", loc="best")
    fig.tight_layout()
    for e in bin_edges:
        ax.axvline(e, color="lightgray", linewidth=0.5)


    plt.show()
    # return fig, ax
