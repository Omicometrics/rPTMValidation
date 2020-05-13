import csv
import operator
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .. import PSM

from .constants import *


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
