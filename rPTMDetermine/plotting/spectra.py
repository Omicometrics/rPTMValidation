from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from rPTMDetermine import Spectrum

from .constants import *


def plot_spectra(
        spectra: Sequence[Spectrum],
        normalize: bool = False,
        mz_range: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None
):
    """
    Plots any number of mass spectra, vertically aligned.

    Args:
        spectra (list): A list of numpy arrays (n x 2).
        normalize: If True, all spectra are normalized before plotting.
        mz_range: The range of m/z values to be plotted.
        save_path: The path to which to save the figure.

    """
    _, axes = plt.subplots(len(spectra), 1, sharex=True)
    axes = [axes] if len(spectra) == 1 else axes

    for ax, spectrum in zip(axes, spectra):
        if normalize:
            spectrum = spectrum.normalize()

        if mz_range is not None:
            spectrum = spectrum[
                (spectrum[:, 0] >= mz_range[0]) &
                (spectrum[:, 0] <= mz_range[1])
            ]

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
