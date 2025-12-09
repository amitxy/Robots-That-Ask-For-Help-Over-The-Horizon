import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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



def reliability_plot(
    df,
    prob_col: str = "target_prob",
    correct_col: str = "correct",
    split_col: str = "test_split",
    n_bins: int = 10,
    include_overall: bool = True,
):
    """
    Calibration plot with one line per split (plus optional overall).
    """
    df = df.copy()
    edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df[prob_col], bins=edges, include_lowest=True)
    agg = (
        df.groupby([split_col, "bin"], observed=True)
        .agg(conf=(prob_col, "mean"), acc=(correct_col, "mean"))
        .reset_index()
    )
    if include_overall:
        overall = (
            df.groupby("bin", observed=True)
            .agg(conf=(prob_col, "mean"), acc=(correct_col, "mean"))
            .reset_index()
        )
        overall[split_col] = "overall"
        agg = pd.concat([agg, overall], ignore_index=True)

    plt.figure(figsize=(5, 4))
    sns.lineplot(data=agg, x="conf", y="acc", hue=split_col, marker="o")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted prob (bin)")
    plt.ylabel("Empirical accuracy")
    plt.ylim(0, 1)
    plt.title("Reliability by split")
    plt.tight_layout()


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

    # Left: coverage vs set size; Right: pred set size distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(data=agg, x="pred_set_size", y="coverage", hue=hue, marker="o", ax=axes[0], palette=palette, hue_order=order)
    min_coverage = agg["coverage"].min()
    # axes[0].set_ylim(max(min_coverage - 0.1, 0), 1)
    axes[0].set_xlabel("Pred set size")
    axes[0].set_ylabel("Avg coverage (target in set)")
    axes[0].set_title("Coverage vs pred set size" + (f" by {method_col}" if hue else ""))
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
    axes[1].set_xlabel("Pred set size")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Pred set size distribution")

    plt.tight_layout()
    plt.show()




########
def ask_rate_vs_target_prob(df):
    df = df.copy()
    bins = [0.0, 0.2, 0.4, 0.7, 0.9, 1.0] 
    df["ask"] = df["pred_set_size"] > 1
    df["bin"] = pd.cut(df["target_prob"], bins=4,include_lowest=True,right=True,)

    ask_rate = df.groupby("bin")["ask"].mean()

    # plt.figure(figsize=(5,4))

    ask_rate.plot(marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ask rate")
    plt.title("Ask probability vs target_prob bin")
    plt.tight_layout()



def fpr_vs_lambda(metrics_df):
    plt.figure(figsize=(5,4))
    plt.plot(metrics_df["lambda"], metrics_df["fp_rate"], marker="o", color="red")
    plt.xlabel("lambda")
    plt.ylabel("FP rate")
    plt.title("False-positive rate vs lambda")
    plt.tight_layout()

###############

def a_in_pred_set_bar_plot(df, method_col: str = "method"):
    """
    Plot P(A in pred_set) by pred_set_size, optionally split by method.
    Supports df_all-style inputs with a 'method' column; if absent, plots overall.
    """
    df = df.copy()
    df["A_in_set"] = df["pred_set"].apply(lambda s: "A" in s if isinstance(s, (list, tuple, set)) else False)

    group_cols = ["pred_set_size"]
    if method_col and method_col in df.columns:
        # group_cols.insert(0,"seed")
        group_cols.insert(0, method_col)

    agg = (
        df.groupby(group_cols, observed=True)["A_in_set"]
        .mean()
        .reset_index(name="p_A_in_set")
    )

    if method_col in agg.columns:
        sns.barplot(data=agg, x="pred_set_size", y="p_A_in_set", hue=method_col)
        plt.legend(title="method")
    else:
        sns.barplot(data=agg, x="pred_set_size", y="p_A_in_set")
    plt.ylabel("P(A in pred_set)")
    plt.xlabel("Pred set size")
    plt.title("How often 'A' is in the prediction set vs set size")
    plt.tight_layout()
    plt.show()


def a_in_pred_set(df, method_col: str = "method", iter_col: str = "seed"):
    """
    Plot P(A in pred_set) vs pred_set_size, plus accuracy vs pred_set_size.
    If method_col and iter_col (seed) exist:
      1) compute mean per (method, seed, pred_set_size)
      2) then average across seeds -> 1 curve per method.
    """
    df = df.copy()
    df["A_in_set"] = df["pred_set"].apply(lambda s: "A" in s )

    order = [method for method in df['method'].unique()]
    if "no_A" in order:
        order = [method for method in df['method'].unique() if method != "no_A" ]
        order.append("no_A")
    
    
    # 1) per-(method, seed, size) mean for A-in-set and accuracy (if present)
    group_cols = [method_col, iter_col, "pred_set_size"]
    per_seed = (
        df.groupby(group_cols, observed=True)
        .agg(
            p_A_in_set=("A_in_set", "mean"),
            acc=("correct", "mean") ,
        )
        .reset_index()
    )
    # 2) average over seeds
    agg = (
        per_seed.groupby([method_col, "pred_set_size"], observed=True)
        .agg(p_A_in_set=("p_A_in_set", "mean"), acc=("acc", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(
        data=agg,
        x="pred_set_size",
        y="p_A_in_set",
        hue=method_col,
        hue_order=order,
        marker="o",
        ax=axes[0],
    )
    axes[0].set_ylim(-0.01, 1.05)
    axes[0].set_ylabel("P(A in pred_set)")
    axes[0].set_xlabel("Pred set size")
    axes[0].set_title("How often 'A' is in the prediction set vs set size")

    sns.lineplot(
        data=agg,
        x="pred_set_size",
        y="acc",
        hue=method_col,
        hue_order=order,
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Pred set size")
    axes[1].set_title("Accuracy vs set size")

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
    if "step_id" not in df.columns:
        df["step_id"] = df.groupby(ann_col).cumcount()
    df["fail"] = ~df[correct_col].astype(bool)
    agg = (
        df.groupby(["step_id", "pred_set_size"])["fail"]
        .mean()
        .reset_index()
        .rename(columns={"fail": "failure_rate"})
    )
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=agg, x="step_id", y="failure_rate", hue="pred_set_size", marker="o")
    plt.xlabel("Step index")
    plt.ylabel("Failure rate")
    plt.title("Failure rate vs step index by pred_set_size")
    plt.ylim(0, 1)
    plt.tight_layout()

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
    lower_bound = min(max(0, cov["coverage"].min() - 0.05), alpha - 0.05)
    axes[0].set_ylim(lower_bound, 1)
    axes[0].hlines(
        y=alpha,
        xmin=-0.5,
        xmax=0.5,
        colors="r",
        linestyles="--",
        label=f"Target coverage ({alpha})",
    )
    axes[0].set_ylabel("Coverage (target in set)")
    axes[0].set_xlabel("")
    axes[0].set_xticks([])
    axes[0].set_title("Coverage per iteration by method")
    axes[0].legend(title=method_col)

    sns.boxplot(data=size, x="dummy", y="pred_set_size", hue=method_col, ax=axes[1])
    axes[1].set_ylabel("Pred set size (avg per iter)")
    axes[1].set_xlabel("")
    axes[1].set_xticks([])
    axes[1].set_title("Pred set size per iteration by method")
    axes[1].legend(title=method_col)

    plt.tight_layout()
    plt.show()
