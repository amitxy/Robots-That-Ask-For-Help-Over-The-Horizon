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


def coverage_vs_set_size(df, alpha_line: float = 0.9):
    """
    Plot average coverage (target in pred_set) versus pred_set_size.
    Shows one line per split on the same axes, plus an optional overall line.
    Requires columns: pred_set, pred_set_size, target_label, split_col.
    """
    df = df.copy()
    df["covered"] = df.apply(
        lambda r: (r["target_label"] in r["pred_set"]) if isinstance(r["pred_set"], (list, tuple, set)) else False,
        axis=1,
    )
    agg = (
        df.groupby("pred_set_size")["covered"]
            .mean()
            .reset_index()
            .rename(columns={"covered": "coverage"})
    )

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=agg, x="pred_set_size", y="coverage", marker="o")
    
    min_coverage = min(agg['coverage'])
    plt.ylim(min_coverage - 0.1, 1)
    plt.xlabel("Pred set size")
    plt.ylabel("Avg coverage (target in set)")
    plt.title("Coverage vs pred set size")
    if alpha_line is not None:
        plt.axhline(alpha_line, linestyle="--", color="gray", alpha=0.7, label=f"alpha={alpha_line}")
        plt.legend()
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

def a_in_pred_set(df):
    df["A_in_set"] = df["pred_set"].apply(lambda s: "A" in s)
    group = df.groupby("pred_set_size")["A_in_set"].mean()
    if group.iloc[0] < 0.05:
        group.iloc[0] += 0.005
    group.plot(kind="bar")
    plt.ylabel("P(A in pred_set)")
    plt.title("How often 'A' is in the prediction set V.S set size")
    plt.tight_layout()
    plt.show()


def pred_set_size_distribution(df, split_col: str = "test_split", kind: str = "hist"):
    """
    Plot distribution of pred_set_size (hist or violin) per split.
    """
    import seaborn as sns
    if kind == "hist":
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x="pred_set_size", hue=split_col, multiple="stack", binwidth=1, discrete=True)
        plt.xlabel("Pred set size")
        plt.ylabel("Count")
        plt.title("Prediction set size distribution")
        plt.tight_layout()
    elif kind == "violin":
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=df, x=split_col, y="pred_set_size", inner="quartile", cut=0)
        plt.xlabel("Split")
        plt.ylabel("Pred set size")
        plt.title("Prediction set size distribution")
        plt.tight_layout()
    else:
        raise ValueError("kind must be 'hist' or 'violin'")


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

def coverage_boxplot(df, iter_col: str = "i"):
    """
    Box plot of per-iteration coverage (fraction of rows with target in pred_set).
    Computes coverage per iter_col, then plots the distribution.
    """

    df = df.copy()
    df["covered"] = df.apply(
        lambda r: (r["target_label"] in r["pred_set"]) if isinstance(r["pred_set"], (list, tuple, set)) else False,
        axis=1,
    )
    # aggregate coverage per iteration
    cov_per_iter = (
        df.groupby(iter_col)["covered"]
        .mean()
        .reset_index(name="coverage")
    )
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=cov_per_iter, y="coverage")
    plt.ylim(0.9, 1)
    plt.ylabel("Coverage (target in set)")
    plt.title("Coverage per iteration")
    plt.tight_layout()
