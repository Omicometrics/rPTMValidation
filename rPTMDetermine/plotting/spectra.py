from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .. import Spectrum

from .constants import *

def plot_spectra(
        spectra: Sequence[Spectrum],
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
