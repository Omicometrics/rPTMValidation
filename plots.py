#! /usr/bin/env python3
"""
This module provides some functions useful for visualizing the peptide
identification validation results.

"""
from typing import List, Tuple

from matplotlib import font_manager
import matplotlib.pyplot as plt

import lda
from peptide_spectrum_match import PSM


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


def plot_scores(psms: List[PSM], prob_threshold=0.99, label_prefix="",
                save_path=None):
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
    val_score = lda.get_validation_threshold(target, prob_threshold)

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
        plt.savefig(save_path)

    plt.show()

    return val_score


def plot_score_similarity(psms: List[PSM], prob_threshold=0.99,
                          save_path=None, use_benchmarks=True,
                          sim_threshold=None):
    """
    Plots the rPTMDetermine scores against the unmodified/modified spectrum
    similarity scores.

    Args:
        psms (list): The list of validated and compared PSMs.

    """
    if not use_benchmarks and sim_threshold is None:
        print("sim_threshold must be passed to plot_score_similarity when "
              "use_benchmarks is set to False")
        sys.exit(1)        
    
    bench_sims, bench_ldas, sims, ldas = [], [], [], []
    for psm in psms:
        if use_benchmarks and psm.benchmark:
            bench_sims.append(psm.max_similarity)
            bench_ldas.append(psm.lda_score)
        else:
            sims.append(psm.max_similarity)
            ldas.append(psm.lda_score)

    val_score = lda.get_validation_threshold(split_target_decoy_scores(psms)[0],
                                         prob_threshold)
    sim_score = min(bench_sims) if use_benchmarks else sim_threshold

    if use_benchmarks:
        plt.scatter(bench_sims, bench_ldas, marker="o", facecolors=TARGET_COLOR,
                    label="Benchmark Identifications")

    if use_benchmarks:
        plt.scatter(sims, ldas, marker="^",
                    facecolors="grey", linewidths=1,
                    label="Other Identifications")
    else:
        plt.scatter(sims, ldas, marker="o", facecolors="none",
                    edgecolors=TARGET_COLOR, linewidths=1)

    plt.xlabel("Similarity Score", fontproperties=FONT)
    plt.ylabel("rPTMDetermine Score", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(val_score, color=THRESHOLD_COLOR, linestyle="--", linewidth=2)

    ax.axvline(sim_score, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    if use_benchmarks:
        ax.legend(prop=FONT, frameon=False, loc=0,
                  bbox_to_anchor=(0.14, 0.5, 0.5, 0.5), handletextpad=0.01)

    max_lda = max(bench_ldas + ldas)
    plt.annotate(f"$s_{{PD}}$={val_score:.2f}",
                 (0.05, val_score + 0.05 * max_lda),
                 fontproperties=FONT)

    plt.annotate(f"$s_{{similarity}}$={sim_score:.2f}",
                 (sim_score - 0.1, 1.1 * max_lda), fontproperties=FONT,
                 annotation_clip=False)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
    
    
def plot_recovered_score_similarity(psms: List[Tuple[PSM, str]],
                                    sim_threshold,
                                    lda_threshold,
                                    save_path=None):
    """
    Plots the rPTMDetermine scores against the unmodified/modified spectrum
    similarity scores for the recovered identifications.

    Args:
        psms (list): The list of validated PSMs as tuples with their
                     normal/decoy indicator.
        sim_threshold (float):
        lda_threshold (float):
        save_path (str, optional):

    """
    target_sims, target_ldas, decoy_sims, decoy_ldas = [], [], [], []
    for psm, pep_type in psms:
        if pep_type == "normal":
            target_sims.append(psm.max_similarity)
            target_ldas.append(psm.lda_score)
        else:
            decoy_sims.append(psm.max_similarity)
            decoy_ldas.append(psm.lda_score)

    plt.scatter(target_sims, target_ldas, marker="o", facecolors="none",
                edgecolors=TARGET_COLOR,
                label="Target Identifications")

    plt.scatter(decoy_sims, decoy_ldas, marker="x",
                facecolors=DECOY_COLOR,
                label="Decoy Identifications")

    plt.xlabel("Similarity Score", fontproperties=FONT)
    plt.ylabel("rPTMDetermine Score", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(lda_threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2)

    ax.axvline(sim_threshold, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=2)

    ax.legend(prop=FONT, frameon=False, loc=0,
              bbox_to_anchor=(0.07, 0.5, 0.5, 0.5), handletextpad=0.01)

    max_lda = max(target_ldas + decoy_ldas)
    plt.annotate(f"$s_{{PD}}$={lda_threshold:.2f}",
                 (0.05, lda_threshold + 0.05 * max_lda),
                 fontproperties=FONT)

    plt.annotate(f"$s_{{similarity}}$={sim_threshold:.2f}",
                 (sim_threshold - 0.1, 1.14 * max_lda), fontproperties=FONT,
                 annotation_clip=False)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_site_probabilities(psms: List[PSM], threshold=0.99, save_path=None):
    """
    Plots the site localization probabilities for the given PSMs.

    Args:
        psms (list): The list of validated and localized PSMs.
        threshold (float): The site probability threshold for localization.

    """
    probs = [p.site_prob for p in psms if p.site_prob is not None]
    probs = sorted(probs)

    plt.scatter(range(len(probs)), probs, color=TARGET_COLOR, marker="x",
                linewidth=2, s=100)

    plt.xlabel("Identification No.", fontproperties=FONT)
    plt.ylabel("Site Probability", fontproperties=FONT)

    ax = plt.gca()
    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=2)

    plt.annotate(f"$p_{{site}}$={threshold}",
                 (0.75 * len(probs), threshold - 0.05), fontproperties=FONT)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
