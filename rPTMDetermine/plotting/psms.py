#! /usr/bin/env python3
"""
This module provides some functions useful for visualizing the peptide
identification validation results.

"""
import collections
import operator
import re
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import numpy as np

from pepfrag import Peptide

from .. import PSM
from ..mass_spectrum import Annotation

from .constants import *


STANDARD_ION_REGEX = re.compile(r"([a-zA-Z]+)(\d+)\[(.*)\]")

ION_CHARGE_REGEX = re.compile(r'[a-zA-Z]\d+\[(\d*)\+\]')


def sort_ion_labels(ions: Sequence[str]) -> List[str]:
    """
    Sorts ion labels according to their ion number.

    In the example of b2[+], the ion number would be 2.

    Args:
        ions: Ion labels to sort.

    Returns:
        Sorted list of string ion labels.

    """
    return sorted(ions, key=lambda s: STANDARD_ION_REGEX.match(s).group(2))


def _add_sequence(
        ax: Axes,
        ions: Sequence[str],
        anns: Dict[str, Annotation],
        height: float,
        mzs: np.array,
        intensities: np.array,
        max_int: float,
        max_mz: float,
        peptide: Peptide,
        b_type: bool
):
    """
    """
    color = "blue" if b_type else "red"
    sorted_ions = sort_ion_labels(ions)
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


def _add_ions(
        ax: Axes,
        ions: Sequence[str],
        anns: Dict[str, Annotation],
        height: float,
        mzs: np.array,
        intensities: np.array,
        color: str = "black"
):
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
                ion_label = (f"{match.group(1)}$_{{{match.group(2)}}}$$^"
                             f"{{{match.group(3)}}}$")

        if not ion_label:
            raise ValueError(f"Ion label could not be extracted from: {ion}")

        ax.text(mz, height, ion_label, color=color, ha="center")


def plot_psm(
        psm: PSM,
        **kwargs
):
    """
    Plots the given `psm`, with optional annotations.

    Args:
        psm: The PSM whose spectrum should be plotted.
        kwargs: Additional arguments for plot_psms.

    """
    return plot_psms([psm], **kwargs)


def plot_psms(
        psms: Sequence[PSM],
        denoise: bool = False,
        tol: float = 0.2,
        annotation: Optional[str] = None,
        save_path: Optional[str] = None,
        rel_intensity: bool = True,
        mz_range: Optional[Tuple[float, float]] = None
):
    """
    Plots the given PSM spectra, with optional annotations.

    Args:
        psms: The PSMs whose spectra should be plotted.
        denoise: Flag indicating whether the spectrum should be denoised before
                 plotting. Defaults to False.
        tol: The tolerance for denoising using theoretical fragment ions.
        annotation: The spectrum annotation method. Options are:
            1. 'sequence': Label the peptide sequence using matched fragments.
            2. 'fragments': Label the spectrum with the fragment ion labels.
        save_path: The path to which to save the resulting plot.
        rel_intensity: Flag indicating whether the intensity scale should be
                       relative. Defaults to True.
        mz_range: An optional filter on the m/z range of the spectrum to plot.

    Returns:
        Annotations for the spectrum, including those not shown (e.g. higher
        charge state).

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
            max_int = 1.

        psm.peptide.clean_fragment_ions()

        if mz_range is not None:
            spec = spec[
                (spec[:, 0] >= mz_range[0]) & (spec[:, 0] <= mz_range[1])
            ]

        ax.stem(
            spec[:, 0],
            spec[:, 1],
            "black",
            basefmt=' ',
            markerfmt=' ',
            label=None,
            use_line_collection=True
        )

        ax.set_ylim(bottom=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_ylabel(
            "Relative Intensity" if rel_intensity else "Intensity",
            fontproperties=FONT
        )

        max_mz = max(spec[:, 0])

        # Find the charge state with the greatest number of b/y ions for
        # annotation
        top_charge: Optional[str] = None
        if annotation is not None:
            ion_counts = collections.Counter(
                [m.group(1) for a in anns.keys()
                 if (m := ION_CHARGE_REGEX.match(a)) is not None]
            )
            top_charge = ion_counts.most_common(1)[0][0]

        if annotation == "sequence":
            def add_sequence(ions: Sequence[str], height: float, b_type: bool):
                _add_sequence(
                    ax,
                    ions,
                    anns,
                    height,
                    spec[:, 0],
                    spec[:, 1],
                    max_int,
                    max_mz,
                    psm.peptide,
                    b_type
                )

            # Annotate b-ions
            add_sequence(
                [a for a in anns.keys()
                 if a[0] == "b" and f"[{top_charge}+]" in a],
                1.01 * max_int,
                True
            )

            # Annotate y-ions
            add_sequence(
                [a for a in anns.keys()
                 if a[0] == "y" and f"[{top_charge}+]" in a],
                1.07 * max_int,
                False
            )

            b_line = mlines.Line2D(
                [],
                [],
                color="blue",
                marker="",
                label="$\\it{b}$-ions"
            )
            y_line = mlines.Line2D(
                [],
                [],
                color="red",
                marker="",
                label="$\\it{y}$-ions"
            )

            plt.legend(
                handles=[b_line, y_line],
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.35),
                ncol=2,
                prop=FONT
            )
        elif annotation == "fragments":
            def add_ions(ions: Sequence[str], height: float, color: str):
                _add_ions(
                    ax,
                    ions,
                    anns,
                    height,
                    spec[:, 0],
                    spec[:, 1],
                    color=color
                )

            def get_ions(char: str):
                return [a for a in anns.keys()
                        if a[0] == char and (f"[{top_charge}+]" in a)]

            # Annotate b-ions
            add_ions(get_ions("b"), 1.01 * max_int, "blue")

            # Annotate y-ions
            add_ions(get_ions("y"), 1.04 * max_int, "red")

            # Annotate a-ions
            add_ions(get_ions("a"), 1.07 * max_int, "green")

            # Annotate precursor ions
            add_ions(
                [a for a in anns.keys() if "[M+H]" in a], 1.10 * max_int,
                "black"
            )

        elif annotation is not None:
            print(f"Invalid annotation option: {annotation}")

        annotated_peaks = []
        for label, ann in anns.items():
            if ann.peak_num < len(spec[:, 0]):
                annotated_peaks.append(
                    (label, spec[:, 0][ann.peak_num], ann.mass_diff)
                )
        annotated_peaks.sort(key=operator.itemgetter(1))
        all_anns.append(annotated_peaks)

    plt.xlabel("m/z", fontproperties=FONT, fontstyle="italic")

    if save_path is not None:
        plt.savefig(save_path, dpi=SAVE_DPI)

    plt.show()

    return all_anns
