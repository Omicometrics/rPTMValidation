import csv
import operator
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .. import PSM

from .constants import *


def split_target_decoy_scores(psms: List[PSM]) \
        -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Splits the target and decoy scores into two lists of tuples.

    Args:
        psms (list): The list of validated PSMs.

    Returns:
        Two lists of tuples, one list of (score, prob) combinations for target
        peptide identifications and one for decoy.

    """
    target, decoy = [], []
    for psm in psms:
        if psm.lda_score is not None and psm.lda_prob is not None:
            target.append((psm.lda_score, psm.lda_prob))
        if psm.decoy_lda_score is not None and psm.decoy_lda_prob is not None:
            decoy.append((psm.decoy_lda_score, psm.decoy_lda_prob))
    return target, decoy


def plot_scores(psms: List[PSM], lda_threshold: float,
                label_prefix: str = "",
                save_path: Optional[str] = None):
    """
    Plots the score distributions of the target and decoy peptide
    identifications.

    Args:
        psms (list): The list of validated PSMs.
        lda_threshold (float): The LDA score threshold for validation.
        label_prefix (str, optional): The 'peptide' prefix in the legend
                                      labels.
        save_path (str, optional): The path to which to save the figure.

    """
    target, decoy = split_target_decoy_scores(psms)

    target_scores = sorted([t[0] for t in target], reverse=True)
    decoy_scores = sorted([d[0] for d in decoy], reverse=True)

    if label_prefix and not label_prefix.endswith(" "):
        label_prefix += " "

    # Plot the target and decoy score distributions
    plt.scatter(range(len(target_scores)), target_scores, marker='o',
                facecolors="none", edgecolors=TARGET_COLOR, linewidths=1,
                label=f"Target {label_prefix}")

    plt.scatter(range(len(decoy_scores)), decoy_scores, marker='x',
                facecolors=DECOY_COLOR, linewidths=1,
                label=f"Decoy {label_prefix}")

    plt.xlabel("Spectrum No.", fontproperties=FONT)
    plt.ylabel("LDA Score", fontproperties=FONT)

    # Calculate the x-position for the score annotation
    ann_x_pos = decoy_scores[0] + 0.05 * len(decoy_scores)

    plt.annotate(f"$s_{{LDA}}$={lda_threshold:.2f}",
                 (ann_x_pos, lda_threshold + 0.05 * max(target_scores)),
                 fontproperties=FONT)

    ax = plt.gca()

    # Horizontal line for probability threshold
    ax.axhline(lda_threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    ax.legend(prop=FONT, frameon=False, handletextpad=0.0001, loc=1)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def plot_score_similarity(psms: Sequence[PSM], lda_threshold: float,
                          save_path: Optional[str] = None,
                          use_benchmarks: bool = True,
                          sim_threshold: Optional[float] = None):
    """
    Plots the rPTMDetermine scores against the unmodified/modified spectrum
    similarity scores.

    Args:
        psms (list): The list of validated and compared PSMs.
        lda_threshold (float):
        save_path (str, optional): The path to which to save the plot.
        use_benchmarks (bool, optional)
        sim_threshold (float, optional)

    """
    if not use_benchmarks and sim_threshold is None:
        print("sim_threshold must be passed to plot_score_similarity when "
              "use_benchmarks is set to False")
        sys.exit(1)

    bench_sims, bench_ldas, sims, ldas = [], [], [], []
    for psm in psms:
        if psm.lda_score is None:
            continue
        if use_benchmarks and psm.benchmark:
            bench_sims.append(psm.max_similarity)
            bench_ldas.append(psm.lda_score)
        else:
            sims.append(psm.max_similarity)
            ldas.append(psm.lda_score)

    ax = plt.gca()

    if use_benchmarks and sim_threshold is None:
        sim_score = min([sim for sim, lda in zip(bench_sims, bench_ldas)
                         if lda >= lda_threshold])
    else:
        sim_score = sim_threshold

    if use_benchmarks:
        ax.scatter(sims, ldas, marker="^",
                   facecolors="grey", linewidths=1,
                   label="Other")
    else:
        ax.scatter(sims, ldas, marker="o", facecolors="none",
                   edgecolors=TARGET_COLOR, linewidths=1)

    if use_benchmarks:
        ax.scatter(bench_sims, bench_ldas, marker="o",
                   facecolors=TARGET_COLOR,
                   label="Benchmark")

    ax.set_xlabel("Similarity Score", fontproperties=FONT)
    # ax.set_xlim(0., 1.)
    ax.set_ylabel("LDA Score", fontproperties=FONT)

    ax.axhline(lda_threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    ax.axvline(sim_score, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    if use_benchmarks:
        handles, labels = ax.get_legend_handles_labels()
        order = [1, 0]
        ax.legend([handles[idx] for idx in order],
                  [labels[idx] for idx in order],
                  prop=FONT, loc=2,
                  bbox_to_anchor=(0.0, 0.5, 0.5, 0.5), handletextpad=0.0001)

    max_lda = max(bench_ldas + ldas)
    ax.annotate(f"$s_{{LDA}}$={lda_threshold:.2f}",
                (0.17, lda_threshold + 0.05 * max_lda),
                fontproperties=FONT)

    ax.annotate(f"$s_{{similarity}}$={sim_score:.2f}",
                (sim_score - 0.1, 1.1 * max_lda), fontproperties=FONT,
                annotation_clip=False)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def plot_recovered_score_similarity(psms: Sequence[PSM],
                                    sim_threshold: float,
                                    lda_threshold: float,
                                    save_path: Optional[str] = None):
    """
    Plots the rPTMDetermine scores against the unmodified/modified spectrum
    similarity scores for the recovered identifications.

    Args:
        psms (list): The list of validated PSMs as tuples with their
                     normal/decoy indicator.
        sim_threshold (float):
        lda_threshold (float):
        save_path (str, optional): The path to which to save the plot.

    """
    target_sims, target_ldas, decoy_sims, decoy_ldas = [], [], [], []
    for psm in psms:
        if psm.lda_score is None:
            continue
        if psm.target:
            target_sims.append(psm.max_similarity)
            target_ldas.append(psm.lda_score)
        else:
            decoy_sims.append(psm.max_similarity)
            decoy_ldas.append(psm.lda_score)

    plt.scatter(target_sims, target_ldas, marker="o", facecolors="none",
                edgecolors=TARGET_COLOR,
                label="Target")

    plt.scatter(decoy_sims, decoy_ldas, marker="x",
                facecolors=DECOY_COLOR,
                label="Decoy")

    plt.xlabel("Similarity Score", fontproperties=FONT)
    plt.ylabel("LDA Score", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(lda_threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    ax.axvline(sim_threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    ax.legend(prop=FONT, loc=4, handletextpad=0.01)

    box = ax.get_position()
    # Reduce the size of the plot to allow space for annotation on RHS
    ax.set_position([box.x0, box.y0,
                     box.width * 0.95, box.height])

    max_lda = max(target_ldas + decoy_ldas)
    plt.annotate(f"$s_{{LDA}}$={lda_threshold:.2f}",
                 (max(target_sims) + 0.06, lda_threshold - 1.),
                 fontproperties=FONT,
                 annotation_clip=False)

    plt.annotate(f"$s_{{similarity}}$={sim_threshold:.2f}",
                 (sim_threshold - 0.1, 1.17 * max_lda), fontproperties=FONT,
                 annotation_clip=False)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def _filter_psms_site_probs(psms: Sequence[PSM]) -> List[float]:
    """
    """
    return sorted([p.site_prob for p in psms if p.site_prob is not None])


def plot_site_probabilities(psms: Sequence[PSM], threshold: float = 0.99,
                            save_path: Optional[str] = None):
    """
    Plots the site localization probabilities for the given PSMs.

    Args:
        psms (list): The list of validated and localized PSMs.
        threshold (float): The site probability threshold for localization.
        save_path (str, optional): The path to which to save the plot.

    """
    probs = _filter_psms_site_probs(psms)

    plt.scatter(range(len(probs)), probs, color=TARGET_COLOR, marker="x",
                linewidth=2, s=100)

    plt.xlabel("Identification No.", fontproperties=FONT)
    plt.ylabel("Site Probability", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2)

    # Configure the x-axis to use integer tick labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.annotate(f"$p_{{site}}$={threshold}",
                 (0.75 * len(probs), threshold - 0.05), fontproperties=FONT)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def plot_validated_recovered_site_probabilities(
        val_psms: Sequence[PSM], rec_psms: Sequence[PSM],
        threshold: float = 0.99, save_path: Optional[str] = None):
    """
    Plots the site probabilities for validated and recovered identifications.

    """
    val_probs = _filter_psms_site_probs(val_psms)
    rec_probs = _filter_psms_site_probs(rec_psms)

    plt.scatter(range(len(val_probs)), val_probs, color="#6c0e08", marker="x",
                linewidth=2, s=100, label="Validated")
    plt.scatter(range(len(rec_probs)), rec_probs, color="#16086c", marker="x",
                linewidth=2, s=100, label="Missed")

    plt.xlabel("Identification No.", fontproperties=FONT)
    plt.ylabel("Site Probability", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2)

    # Configure the x-axis to use integer tick labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.annotate(f"$p_{{site}}$={threshold}",
                 (0.75 * len(rec_probs), threshold - 0.05),
                 fontproperties=FONT)

    ax.legend(prop=FONT, frameon=False, loc=4, handletextpad=0.01)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def plot_similarity_score_dist(scores: Sequence[float], kde: bool = False,
                               save_path: Optional[str] = None):
    """
    Plots the distribution of similarity scores using a histogram or a
    kernel density estimate plot.

    Note that this function requires seaborn to be installed.

    Args:
        scores (list): The similarity scores.
        kde (bool, optional): Whether to plot using kernel density estimation
                              rather than as a histogram.
        save_path (str, optional): The path to which to save the plot.

    """
    import seaborn as sns

    scores = sorted(scores)

    if kde:
        sns.distplot(scores, hist=True, kde=True,
                     bins=20, color='darkblue',
                     hist_kws={'edgecolor': 'black'},
                     kde_kws={'linewidth': 4})
        plt.ylabel("Probability Density", fontproperties=FONT)
    else:
        sns.distplot(scores, hist=True, kde=False,
                     bins=20, color='darkblue',
                     hist_kws={'edgecolor': 'black'})
        plt.ylabel("Frequency", fontproperties=FONT)

    plt.xlabel("Similarity Score", fontproperties=FONT)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def plot_fisher_scores(
        scores: Dict[str, float],
        threshold: Optional[float] = None,
        save_path: Optional[str] = None
):
    """
    Plots Fisher scores against their features.

    Args:
        scores (dict); A dictionary of feature to Fisher score.
        threshold (float, optional): The feature selection threshold. If None,
                                     no threshold will be plotted.
        save_path (str, optional): The path to which to save the plot.

    """
    features, feature_scores = zip(*[(k, v) for k, v in scores.items()])

    features = tuple([f if f != "n_missed_cleavages" else "NMC"
                      for f in features])

    # Sort the features by score
    features, feature_scores = zip(*sorted(list(zip(features, feature_scores)),
                                           key=operator.itemgetter(1),
                                           reverse=True))

    plt.plot(features, feature_scores, "o")
    ax = plt.gca()

    if threshold is not None:
        # Horizontal line for score threshold
        ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--",
                   linewidth=2)

    box = ax.get_position()
    # Reduce the size of the plot to allow space for axis labels
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Rotate the feature labels for visibility
    plt.xticks(rotation=90)

    plt.ylabel("Fisher Score", fontproperties=FONT)

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()


def plot_fisher_scores_file(scores_file: str, threshold: float,
                            save_path: Optional[str] = None):
    """
    Plots the Fisher scores calculated during model construction.

    Args:
        scores_file (str): The path to the Fisher scores CSV file.
        threshold (float): The feature selection threshold.
        save_path (str, optional): The path to which to save the plot.

    """
    scores = {}
    with open(scores_file, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            scores[row[0]] = float(row[1])

    plot_fisher_scores(scores, threshold=threshold, save_path=save_path)