#! /usr/bin/env python3
"""
This module provides some functions useful for visualizing the peptide
identification validation results.

"""
import csv
import operator
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

from matplotlib import font_manager
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns

from pepfrag import Peptide

from .mass_spectrum import Annotation
from .peptide_spectrum_match import PSM


TARGET_COLOR = "#4472C4"
DECOY_COLOR = "#C00000"
THRESHOLD_COLOR = "green"

FONTFAMILY = "Times New Roman"
FONTSIZE = 14
FONT = font_manager.FontProperties(family=FONTFAMILY, size=FONTSIZE)

SAVE_DPI = 1200

STANDARD_ION_REGEX = re.compile(r"([a-zA-Z]+)(\d+)\[(.*)\]")


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

    Args:
        scores (list): The similarity scores.
        kde (bool, optional): Whether to plot using kernel density estimation
                              rather than as a histogram.
        save_path (str, optional): The path to which to save the plot.

    """
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


def plot_spectra(
        spectra: Sequence[np.array],
        normalize: bool = False,
        max_x: Optional[float] = None,
        save_path: Optional[str] = None
):
    """
    Plots any number of mass spectra, vertically aligned.

    Args:
        spectra (list): A list of numpy arrays (n x 2).
        normalize: If True, all spectra are normalized before plotting.
        max_x: The maximum value of m/z to be plotted.
        save_path: The path to which to save the figure.

    """
    _, axes = plt.subplots(len(spectra), 1, sharex=True)
    axes = [axes] if len(spectra) == 1 else axes

    for ax, spectrum in zip(axes, spectra):
        if normalize:
            spectrum = spectrum.normalize()
        if max_x is not None:
            spectrum = spectrum[spectrum.mz <= max_x]

        ax.stem(
            spectrum[:, 0],
            spectrum[:, 1],
            "black",
            basefmt=' ',
            markerfmt=' ',
            label=None
        )
        ax.set_ylabel(
            "Relative Intensity" if normalize else "Intensity",
            fontproperties=FONT
        )

        ax.set_ylim(bottom=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.xlabel('m/z', fontproperties=FONT, style='italic')

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)
    else:
        plt.show()


def plot_spectrum(spectrum: np.array, **kwargs):
    """
    Plots a single mass spectrum.

    Args:
        spectrum (numpy array)
    """
    plot_spectra([spectrum], **kwargs)


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


def _add_sequence(ax: Axes, ions: Sequence[str], anns: Dict[str, Annotation],
                  height: float, mzs: np.array, intensities: np.array,
                  max_int: float, max_mz: float, peptide: Peptide,
                  b_type: bool):
    """
    """
    color = "blue" if b_type else "red"
    sorted_ions = sorted(ions, key=lambda s: int(s[1:-3]))
    for ii, ion in enumerate(sorted_ions):
        ann1 = anns[ion]
        x1 = mzs[ann1.peak_num]

        # Arrow down to annotated peaks
        ax.arrow(x1, height, 0,
                 intensities[ann1.peak_num] - height,
                 alpha=0.4, ls="dashed", color=color)

        if ii + 1 >= len(sorted_ions):
            continue

        next_ion = sorted_ions[ii + 1]
        ann2 = anns[next_ion]
        if ann2.ion_pos == ann1.ion_pos + 1:
            x2 = mzs[ann2.peak_num]

            # Arrow to consecutive ion annotation
            ax.arrow(x2, height, (x1 - x2) + 0.005 * max_mz, 0,
                     length_includes_head=True,
                     head_width=0.01 * max_int, head_length=0.01 * max_mz,
                     fc=color, ec=color, clip_on=False)

            res_idx = (ann1.ion_pos if b_type
                       else len(peptide.seq) - ann2.ion_pos)
            res = peptide.seq[res_idx]
            if any(ms.site == res_idx + 1 for ms in peptide.mods):
                res += "*"

            ax.text(0.5 * (x2 + x1), 1.01 * height, res, color=color,
                    fontsize=14)


def _add_ions(ax: Axes, ions: Sequence[str], anns: Dict[str, Annotation],
              height: float, mzs: np.array, intensities: np.array,
              color: str = "black"):
    """
    Adds the selected ion annotations to the axis.

    """
    for ion in ions:
        ann = anns[ion]
        if ann.peak_num >= len(mzs):
            continue
        mz = mzs[ann.peak_num]

        # Line down to the annotated peak
        ax.arrow(mz, height, 0,
                 intensities[ann.peak_num] - height,
                 alpha=0.4, ls="dashed", color=color)

        ion_label = ""
        if ion.startswith("["):
            pattern = re.compile(r"(\[.*\])\[(.*)\]")
            match = pattern.match(ion)
            if match is not None:
                ion_label = f"{match.group(1)}$^{{{match.group(2)}}}$"
        else:
            match = STANDARD_ION_REGEX.match(ion)
            if match is not None:
                ion_label = f"{match.group(1)}$_{{{match.group(2)}}}$$^{{{match.group(3)}}}$"

        if not ion_label:
            raise ValueError(f"Ion label could not be extracted from: {ion}")

        ax.text(mz, height, ion_label, color=color, ha="center")


def plot_psm(
        psms: Union[PSM, List[PSM]],
        denoise: bool = False,
        tol: float = 0.2,
        annotation: Optional[str] = None,
        save_path: Optional[str] = None,
        rel_intensity: bool = True,
        mz_range: Optional[Tuple[float, float]] = None):
    """
    """
    all_anns = []
    
    if isinstance(psms, PSM):
        psms = [psms]

    _, axes = plt.subplots(len(psms), 1, sharex=True, figsize=(10, 5))
    axes = [axes] if len(psms) == 1 else axes
    for ax, psm in zip(axes, psms):
        if denoise:
            _, denoised_spec = psm.denoise_spectrum(tol=tol)
            spec = denoised_spec
            if psm.peptide.fragment_ions is None:
                raise RuntimeError("No fragment ions available for annotation")
            anns = denoised_spec.annotate(psm.peptide.fragment_ions, tol=tol)
        else:
            spec = psm.spectrum
            anns = psm.annotate_spectrum(tol=tol)

        max_int = max(spec[:, 1])
        if rel_intensity:
            spec[:, 1] /= max_int

        psm.peptide.clean_fragment_ions()

        if mz_range is not None:
            spec = spec[(spec[:, 0] >= mz_range[0]) &
                        (spec[:, 0] <= mz_range[1])]

        ax.stem(spec[:, 0], spec[:, 1], "black", basefmt=' ', markerfmt=' ',
                label=None, use_line_collection=True)
                
        ax.set_ylim(bottom=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_ylabel("Relative Intensity" if rel_intensity else "Intensity",
                      fontproperties=FONT)

        max_mz = max(spec[:, 0])

        # if mz_range is not None:
        #     # Filtering on the mz can invalidate the annotation indices,
        #     # so prevent annotation in this case
        #     continue
       
        if annotation == "sequence":
            def add_sequence(ions: Sequence[str], height: float, b_type: bool):
                _add_sequence(ax, ions, anns, height, spec[:, 0], spec[:, 1],
                              max_int, max_mz, psm.peptide, b_type)

            # Annotate b-ions
            add_sequence([a for a in anns.keys() if a[0] == "b" and "[+]" in a],
                         1.01 * max_int, True)

            # Annotate y-ions
            add_sequence([a for a in anns.keys() if a[0] == "y" and "[+]" in a],
                         1.07 * max_int, False)

            b_line = mlines.Line2D([], [], color="blue", marker="",
                                   label="$\\it{b}$-ions")
            y_line = mlines.Line2D([], [], color="red", marker="",
                                   label="$\\it{y}$-ions")
                                   
            plt.legend(handles=[b_line, y_line], frameon=False,
                       loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2,
                       prop=FONT)
        elif annotation == "fragments":
            def add_ions(ions: Sequence[str], height: float, color: str):
                _add_ions(ax, ions, anns, height, spec[:, 0], spec[:, 1],
                          color=color)

            def get_ions(char: str):
                return [a for a in anns.keys() if a[0] == char and ("[+]" in a)] 

            # Annotate b-ions
            add_ions(get_ions("b"), 1.01 * max_int, "blue")

            # Annotate y-ions
            add_ions(get_ions("y"), 1.04 * max_int, "red")

            # Annotate a-ions
            add_ions(get_ions("a"), 1.07 * max_int, "green")

            # Annotate precursor ions
            add_ions([a for a in anns.keys() if "[M+H]" in a], 1.10 * max_int,
                     "black")

        elif annotation is not None:
            print(f"Invalid annotation option: {annotation}")

        annotated_peaks = []
        for label, ann in anns.items():
            if ann.peak_num < len(spec[:, 0]):
                annotated_peaks.append((label, spec[:, 0][ann.peak_num],
                                        ann.mass_diff))
        annotated_peaks.sort(key=operator.itemgetter(1))
        all_anns.append(annotated_peaks)

    plt.xlabel("m/z", fontproperties=FONT, fontstyle="italic")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()
    
    return all_anns


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
