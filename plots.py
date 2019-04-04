#! /usr/bin/env python3
"""
This module provides some functions useful for visualizing the peptide
identification validation results.

"""
import bisect
import operator
from typing import List, Tuple

from matplotlib import font_manager
import matplotlib.pyplot as plt

from psm import PSM


TARGET_COLOR = "#4472C4"
DECOY_COLOR = "#C00000"
THRESHOLD_COLOR = "green"

FONTFAMILY = "Times New Roman"
FONTSIZE = 16
FONT = font_manager.FontProperties(family=FONTFAMILY, size=FONTSIZE)


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
        # Exclude those results from deamidation correction since they don't
        # have assigned probabilities
        if psm.lda_prob is not None:
            target.append((psm.lda_score, psm.lda_prob))
            decoy.append((psm.decoy_lda_score, psm.decoy_lda_prob))
    return target, decoy


def get_validation_threshold(val_results: List[Tuple[float, float]],
                             prob_threshold: float) -> float:
    """
    Finds the minimum score associated with the given probability threshold.

    Args:
        val_results (list): A list of tuples of (score, probability).
        prob_threshold (float): The probability threshold.

    Returns:
        The score threshold for the probability as a float.

    """
    val_results = sorted(val_results, key=operator.itemgetter(0))
    idx = bisect.bisect_left([r[1] for r in val_results], prob_threshold) + 1
    return val_results[idx][0]


def plot_scores(psms: List[PSM], prob_threshold=0.99, label_prefix="",
                save_path=None, **kwargs):
    """
    Plots the score distributions of the target and decoy peptide
    identifications.

    Args:
        psms (list): The list of validated PSMs.
        prob_threshold (float, optional): The probability threshold for
                                          validation.
        label_prefix (str, optional): The 'peptide' prefix in the legend
                                      labels.
        save_path (str, optional): The path to which to save the figure.

    """
    target, decoy = split_target_decoy_scores(psms)

    # Find the position of the last validated PSM
    val_score = get_validation_threshold(target, prob_threshold)

    target_scores = sorted([t[0] for t in target], reverse=True)
    decoy_scores = sorted([d[0] for d in decoy], reverse=True)

    if label_prefix and not label_prefix.endswith(" "):
        label_prefix += " "

    # Plot the target and decoy score distributions
    plt.scatter(range(len(target_scores)), target_scores, marker='o',
                facecolors="none", edgecolors=TARGET_COLOR, linewidths=1,
                label=f"Target {label_prefix}peptide identifications")
    plt.scatter(range(len(decoy_scores)), decoy_scores, marker='x',
                facecolors=DECOY_COLOR, linewidths=1,
                label=f"Decoy {label_prefix}peptide identifications")

    plt.xlabel("Spectrum No.", fontproperties=FONT)
    plt.ylabel("rPTMDetermine Score", fontproperties=FONT)
    
    # Calculate the x-position for the score annotation
    ann_x_pos = ([(ii, ds) for ii, ds in enumerate(decoy_scores)
                  if ds > val_score][-1][0] + 0.05 * len(decoy_scores))
    
    plt.annotate(f"$s_{{PD}}$={val_score:.2f}",
                 (ann_x_pos, val_score + 0.05 * max(target_scores)),
                 fontproperties=FONT)

    ax = plt.gca()

    # Horizontal line for probability threshold
    ax.axhline(val_score, color=THRESHOLD_COLOR, linestyle="--", linewidth=2)

    ax.legend(prop=FONT, frameon=False, loc=1, handletextpad=0.01)

    if save_path is not None:
        plt.savefig(save_path, **kwargs)

    plt.show()


def plot_score_similarity(psms: List[PSM], prob_threshold=0.99):
    """
    Plots the rPTMDetermine scores against the unmodified/modified spectrum
    similarity scores.

    Args:
        psms (list): The list of validated and compared PSMs.

    """
    bench_sims, bench_ldas, sims, ldas = [], [], [], []
    for ii, psm in enumerate(psms):
        sim = (max(s[2] for s in psm.similarity_scores)
               if psm.similarity_scores else 0.)
        if psm.benchmark:
            bench_sims.append(sim)
            bench_ldas.append(psm.lda_score)
        else:
            sims.append(sim)
            ldas.append(psm.lda_score)
            
    print(psms[46])

    plt.scatter(sims, ldas, marker="^",
                facecolors="grey", linewidths=1)
                
    plt.scatter(bench_sims, bench_ldas, marker="o", facecolors=TARGET_COLOR)

    plt.xlabel("Similarity Score", fontproperties=FONT)
    plt.ylabel("rPTMDetermine Score", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(get_validation_threshold(split_target_decoy_scores(psms)[0],
                                        prob_threshold),
               color=THRESHOLD_COLOR, linestyle="--")

    ax.axvline(min(bench_sims), color=THRESHOLD_COLOR, linestyle="--")
    ax.axvline(0.55, color="red", linestyle="--")

    plt.show()
