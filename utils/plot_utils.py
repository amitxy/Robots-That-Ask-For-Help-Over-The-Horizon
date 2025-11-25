import matplotlib.pyplot as plt


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
