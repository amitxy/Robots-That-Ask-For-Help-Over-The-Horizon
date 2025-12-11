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
    # aggregate over seeds and fill missing method/set_size combos with 0 coverage
    if hue:
        methods = df[method_col].unique()
        sizes = np.sort(df["pred_set_size"].unique())
        idx = pd.MultiIndex.from_product([methods, sizes], names=[method_col, "pred_set_size"])
        agg = (
            agg.groupby([method_col, "pred_set_size"], observed=True)["coverage"]
               .mean()
               .reindex(idx, fill_value=0)
               .reset_index()
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
    # plt.xlim(0, df["target_risk"].max() * 1.05)
    # plt.ylim(0, df["emp_risk"].max() * 1.05)
    plt.xlabel("Target risk (α)")
    plt.ylabel("Empirical risk (FPR per task)")
    plt.title("Empirical Risk vs Target Risk")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


def risk_vs_pred_size(
    metrics_df: pd.DataFrame,
    risk_col="fp_rate_per_task",   # target risk α
    target_col="pred_set_avg",   
    method_col="method",           # optional
):
    df = metrics_df.copy()
    
    sns.lineplot(
        data=df.sort_values([risk_col, method_col]),
        x=risk_col,
        y=target_col,
        hue=method_col if method_col in df.columns else None,
        marker="o",
    )
   
    # plt.xlim(0, df[risk_col].max() * 1.05)
    # plt.ylim(0, df[target_col].max() * 1.05)
    plt.set_xlabel(r"Empirical risk $(\hat{R})$")
    plt.ylabel("Avg prediction set size")
    plt.title("Avg prediction set size vs Empirical Risk")
    plt.legend()
    plt.tight_layout()
    plt.show()


def risk_and_size_plots(
    metrics_df: pd.DataFrame,
    target_risk_col: str = "risk_level",
    emp_risk_col: str = "fp_rate_per_task",
    size_col: str = "pred_set_avg",
    method_col: str = "method",
):
    """
    Two-panel plot:
      - Left: target risk vs empirical risk (should lie on/below y=x)
      - Right: empirical risk vs average prediction set size
    """
    df = metrics_df.copy()
    hue = method_col if method_col in df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  
    # Left: risk vs set size
    sns.lineplot(
        data=df.sort_values([emp_risk_col, method_col]),
        x=emp_risk_col,
        y=size_col,
        hue=hue,
        marker="o",
        ax=axes[0],
        legend=False,
    )
    
    axes[0].set_xlabel(r"Empirical risk $(\hat{R})$")
    axes[0].set_ylabel("Avg prediction set size")
    axes[0].set_title("Pred set size vs Empirical Risk")
    axes[0].grid(alpha=0.3)

    # Right: risk vs set size
    sns.lineplot(
        data=df.sort_values([emp_risk_col, method_col]),
        x=emp_risk_col,
        y='ask_prob',
        hue=method_col if method_col in df.columns else None,
        marker="o",
    )
    axes[1].set_xlabel(r"Empirical risk $(\hat{R})$")
    axes[1].set_ylabel("Ask probability")
    axes[1].set_title("Ask prob vs Empirical Risk")

    if hue:
        handles, labels = axes[1].get_legend_handles_labels()
        axes[0].legend(handles, labels, title=method_col)
        axes[1].legend(title=method_col)
    else:
        axes[0].legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def fpr_a_vs_coverage(metrics_df: pd.DataFrame, method_col: str = "method", risk_col: str = "fp_rate_per_task",  fp_col="fp_rate"):
    """
    Scatter/line plot of FPR for class A (pred=A | true!=A) vs overall empirical coverage.
    Computes per (method, seed) then plots one curve per method.
    """
    
    df = metrics_df.copy()
    
    sns.lineplot(
        data=df.sort_values(["target_in_pred_set", fp_col]),
        x=fp_col,
        y="target_in_pred_set",
        hue=method_col if method_col in df.columns else None,
        marker="o",
    )
    
    # plt.xlim(0, df["target_risk"].max() * 1.05)
    # plt.ylim(0, df["emp_risk"].max() * 1.05)
    max_x = df[risk_col].max()
    # plt.plot([0, max_x + 0.05], [0, 1], "k--", label="y = x")
    plt.xlabel("Empirical coverage (Pr(target in pred_set))")
    plt.ylabel(r"FPR for A ($Pr(\hat{Y}=A | Y\neq A)$)")
    plt.title("FPR for class A vs coverage")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
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

 
