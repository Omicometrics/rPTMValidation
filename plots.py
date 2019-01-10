import matplotlib.pyplot as plt
from matplotlib import font_manager
from math import floor, ceil
import numpy as np


def getionidx(ions, iontype):
    """ get indices of ion names which belong to iontype """
    iix = set()
    for i, (mz, ion) in enumerate(ions):
        if ion:
            nx = [x.split('/')[0] for x in ion.split(',')]
            nx = [x for x in nx if not '-' in x]
            if nx:
                if any(x.startswith(iontype) for x in nx):
                    iix.add(i)
    return iix

def setlineproperty(baseline, stemlines, markerline, stemlinecolor, linewidth=1.2):
    # set line properties of the plot
    #bs = plt.setp(bsline, 'linewidth', 0.6, 'color', 'k')
    baseline.set_visible(False)
    lines = plt.setp(stemlines, 'color', stemlinecolor, linewidth=linewidth)
    mk = plt.setp(markerline, 'marker', ' ')


def partitionions(ions, modidx):
    """
    Separate annotations into different types and return indices 
    """
    kb, ky = modidx
    bix, yix, ybmix = set(), set(), set()
    for i, (mz, ion) in enumerate(ions):
        if ion:
            nx = [x.split('/')[0] for x in ion.split(',')]
            nx = [x for x in nx if not '-' in x and x[0] in 'yb']
            if nx:
                if any(x.startswith('b') for x in nx):
                    if any(int(x.split('[')[0][1:])>=kb for x in nx):
                        ybmix.add(i)
                    else:
                        bix.add(i)
                if any(x.startswith('y') for x in nx):
                    if any(int(x.split('[')[0][1:])>=ky for x in nx):
                        ybmix.add(i)
                    else:
                        yix.add(i)

    byx = bix&yix
    bix.difference_update(byx)
    yix.difference_update(byx)
    byxs = bix.union(yix)
    byxs = byxs.union(ybmix)
    pix, bydelta = set(), set()
    for i, (mz, ion) in enumerate(ions):
        if ion and i not in byxs:
            nx = [x.split('/')[0] for x in ion.split(',')]
            if any(x.startswith('p') and not '-' in x for x in nx):
                pix.add(i)
            else:
                if nx and any(x.count('-')==1 and not '-2' in x for x in nx):
                    bydelta.add(i)
                else:
                    nx2 = [x for x in nx if '-' not in x and x.startswith('a')]
                    if nx2:
                        bydelta.add(i)
    byxs = byxs.union(pix)
    byxs = byxs.union(bydelta)
    # not b and y ions
    immix = [i for i in range(len(ions)) if 'imm' in ions[i][1]]
    byxs = byxs.union(immix)
    nix = [i for i in range(len(ions)) if i not in byxs]

    return nix, bydelta, bix, yix, byx, ybmix, pix, immix 


def axadd(x, y, axis, lc=None, lw=2., label=None):
    """
    Add data to axis
    """
    if not x:
        return None, None, None
    mkline, stemlines, bsline = axis.stem(x, y, label=label)
    if lc:
        setlineproperty(bsline, stemlines, mkline, lc, lw)
    return mkline, stemlines, bsline


def plotspectrum(spectrum, ions, seq, mods, modtype, filename):
    """
    Plot tandem mass spectrum with annotations of b and y ions stored
    in ions and saved to filename.
    """
    m = max(x[1] for x in spectrum)
    spectrum = [[x[0], x[1]/m*100.] for x in spectrum]
    def getxy(idx):
        return [spectrum[i][0] for i in idx], [spectrum[i][1] for i in idx]
    # indices of b ions
    if not any(modtype in x for x in mods):
        raise ValueError('Incorrect Modification Type.')
    kx = [k for _, k, x in mods if x==modtype]
    modtag = any(m!=0 for m, _, x in mods if x==modtype)
    nr = len(seq)
    kb, ky = min(kx), min(nr-k+1 for k in kx)
    partedions = partitionions(ions, (kb, ky))

    # plots
    fig, ax = plt.subplots()
    # line colors
    colors = ('lightgray', 'slategray', 'g', 'r', 'k', 'maroon', 'mediumblue', 'indigo')
    labels = ('Unannotated', "Neutral losses", "b ions", "y ions", "y&b ions",
              "Modified ions" if modtag else "Non-modified ions (Y containing)",
              "Precursor", "Immonium ions")
    for i, ixk in enumerate(partedions):
        if labels[i] != 'Modified ions':
            _ = axadd(*(getxy(ixk)+(ax,)), lc=colors[i], label=labels[i])
        else:
            mk,stline,bsline = axadd(*(getxy(ixk)+(ax,)),
                                     lc=colors[i], label=labels[i])
            # if this is actually from modified peptide
            if mk:
                if modtag: stemlines = plt.setp(stline, 'linestyle', '--')
                mk = plt.setp(mk, 'marker', 'v', ms = 5, color = colors[i])

    # set up parameters
    ylim = ax.set_ylim(bottom=0.)#, top = 100.)
    ax.tick_params(top='off', right='off')#, labelsize = 28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
##    x0, x1 = ax.get_xlim()
##    y0, y1 = ax.get_ylim()
##    ax.xaxis.set_ticks(np.arange(0, x1, 500))
##    x0, x1 = ax.set_xlim(left=x0)
##    ax.yaxis.set_ticks(np.arange(0, y1, 10))
##    y0, y1 = ax.set_ylim(bottom=0)
    xlabel = ax.set_xlabel('m/z', fontname='Times New Roman', fontsize=20, style='italic')
    ylabel = ax.set_ylabel('Intensity', fontname='Times New Roman', fontsize=20)
    font = font_manager.FontProperties(family='times new roman', size=12)
    ax.legend(prop=font, frameon=False, bbox_to_anchor=(.75, 1.05), loc=2)
    ax.xaxis.set_label_coords(0.5, -0.06)
    fig.set_size_inches(10., 4.)
    plt.savefig(filename)#, dpi=1200)

    plt.close()


def plotspectrum2(spectrum, ions, filename):
    """
    Plot tandem mass spectrum with annotations of b and y ions stored
    in ions and saved to filename.
    """
    m = max(x[1] for x in spectrum)
    spectrum = [[x[0], x[1]/m*100.] for x in spectrum]
    def getxy(idx):
        return [spectrum[i][0] for i in idx], [spectrum[i][1] for i in idx]
    # indices of b ions
    kb, ky = 10000, 10000
    partedions = partitionions(ions, (kb, ky))

    # plots
    fig, ax = plt.subplots()
    # line colors
    colors = ('lightgray', 'slategray', 'g', 'r', 'k', 'maroon', 'mediumblue', 'indigo')
    labels = ('Unannotated', "Neutral losses", "b ions", "y ions", "y&b ions",
              "Modified ions", "Precursor", "Immonium ions")
    for i, ixk in enumerate(partedions):
        if not ixk: continue
        mk,stline,bsline = axadd(*(getxy(ixk)+(ax,)), lc=colors[i], label=labels[i])

    # set up parameters
    ylim = ax.set_ylim(bottom=0.)#, top = 100.)
    ax.tick_params(top='off', right='off')#, labelsize = 28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlabel = ax.set_xlabel('m/z', fontname='Times New Roman', fontsize=20, style='italic')
    ylabel = ax.set_ylabel('Intensity', fontname='Times New Roman', fontsize=20)
    font = font_manager.FontProperties(family='times new roman', size=16)
    ax.legend(prop=font, frameon=False, bbox_to_anchor=(.75, 1.05), loc=2)
    ax.xaxis.set_label_coords(0.5, -0.06)
    fig.set_size_inches(8., 4.)
##    plt.savefig(filename, dpi=2000)
##
    plt.show()


def bar2spectrumwithmod(sp1, ions1, sp2, ions2, seq, mods, modtype, filename):
    """
    Plot tandem mass spectra of modified peptides for comparison.
    """
    if not any(modtype in x for x in mods):
        raise ValueError('Incorrect Modification Type.')
    kx = [k for _, k, x in mods if x==modtype]
    nr = len(seq)
    kb, ky = min(kx), min(nr-k+1 for k in kx)
    # peak normalization
    m = max(x[1] for x in sp1)
    sp1 = [[x[0], x[1]/m*100.] for x in sp1]
    m = max(x[1] for x in sp2)
    sp2 = [[x[0], x[1]/m*100.*-1] for x in sp2]
    # spectrum 1
    partedions1 = partitionions(ions1, (kb, ky))
    # spectrum 2
    partedions2 = partitionions(ions2, (kb, ky))

    # plots
    fig, ax = plt.subplots()
    colors = ('lightgray', 'slategray', 'g', 'r', 'k', 'maroon', 'mediumblue', 'indigo')
    labels = ('Unannotated', "Neutral losses", "b ions", "y ions", "y&b ions",
              "Modified ions", "Precursor", "Immonium ions")
    # .. plots 1
    for i, ixk in enumerate(partedions1):
        mzk, intk = [sp1[j][0] for j in ixk], [sp1[j][1] for j in ixk]
        mk,stline,bsline = axadd(mzk, intk, ax, lc=colors[i], label=labels[i])
        if labels[i] == 'Modified ions' and mk:
            # if this is actually from modified peptide
            stemlines = plt.setp(stline, 'linestyle', '--')
            mk = plt.setp(mk, 'marker', 'v', ms = 5, color = colors[i])
    # .. plots 2
    for i, ixk in enumerate(partedions2):
        mzk, intk = [sp2[j][0] for j in ixk], [sp2[j][1] for j in ixk]
        mk,stline,bsline = axadd(mzk, intk, ax, lc=colors[i])
        if labels[i] == 'Modified ions' and mk:
            # if this is actually from modified peptide
            stemlines = plt.setp(stline, 'linestyle', '--')
            mk = plt.setp(mk, 'marker', 'v', ms = 5, color = colors[i])

    # setup parameters
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _ = ax.plot(xlim, [0, 0], lw=0.6, c='k')
    xlim, ylim = ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.tick_params(top='off', right='off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlabel = ax.set_xlabel('m/z', fontname='Times New Roman', fontsize=20, style='italic')
##    xlim = ax.set_xlim((100., 1500.))
    ylabel = ax.set_ylabel('Intensity', fontname='Times New Roman', fontsize=20)
    font = font_manager.FontProperties(family='times new roman', size=10)
    ax.legend(prop=font, frameon=False)
##    plt.savefig(filename, dpi=300)

    plt.show()


def bar2spectrumwithmod2(sp1, ions1, sp2, ions2, seq, mods, modtype, filename):
    """
    Plot tandem mass spectra of modified and non-modified
    peptides for comparison.
    """
    if not any(modtype in x for x in mods):
        raise ValueError('Incorrect Modification Type.')
    kx = [k for _, k, x in mods if x==modtype]
    nr = len(seq)
    kb, ky = min(kx), min(nr-k+1 for k in kx)
    # peak normalization
    m = max(x[1] for x in sp1)
    sp1 = [[x[0], x[1]/m*100.] for x in sp1]
    m = max(x[1] for x in sp2)
    sp2 = [[x[0], x[1]/m*100.*-1] for x in sp2]
    # spectrum 1
    partedions1 = partitionions(ions1, (kb, ky))
    # spectrum 2
    partedions2 = partitionions(ions2, (kb, ky))

    # plots
    fig, ax = plt.subplots()
    colors = ('lightgray', 'slategray', 'g', 'r', 'k', 'maroon', 'mediumblue', 'indigo')
    labels = ('Unannotated', "Neutral losses", "b ions", "y ions", "y&b ions",
              "Modified ions", "Precursor", "Immonium ions")
    # .. plots 1
    for i, ixk in enumerate(partedions1):
        mzk, intk = [sp1[j][0] for j in ixk], [sp1[j][1] for j in ixk]
        mk,stline,bsline = axadd(mzk, intk, ax, lc=colors[i], label=labels[i])
        if labels[i] == 'Modified ions' and mk:
            # if this is actually from modified peptide
            stemlines = plt.setp(stline, 'linestyle', '--')
            mk = plt.setp(mk, 'marker', 'v', ms = 5, color = colors[i])
    # .. plots 2
    for i, ixk in enumerate(partedions2):
        mzk, intk = [sp2[j][0] for j in ixk], [sp2[j][1] for j in ixk]
        _ = axadd(mzk, intk, ax, lc=colors[i])
        if labels[i] == 'Modified ions' and mk:
            mk,stline,bsline = axadd(mzk, intk, ax, lc=colors[i],
                                     label='Non-modified ions (Y containing)')
            if mk:
                mk = plt.setp(mk, 'marker', '^', ms = 5, color = colors[i])

    # setup parameters
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _ = ax.plot(xlim, [0, 0], lw=0.6, c='k')
    xlim, ylim = ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.tick_params(top='off', right='off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlabel = ax.set_xlabel('m/z', fontname='Times New Roman', fontsize=20, style='italic')
##    xlim = ax.set_xlim((100., 1500.))
    ylabel = ax.set_ylabel('Intensity', fontname='Times New Roman', fontsize=20)
    font = font_manager.FontProperties(family='times new roman', size=10)
    ax.legend(prop=font, frameon=False)
##    plt.savefig(filename, dpi=300)

    plt.show()


def bar2spectrum(sp1, ions1, sp2, ions2, filename):
    """
    Plot 2 tandem mass spectra for comparison of the 2 spectra
    with fragment ions are highlighed by different ion types
    """
    kb=ky=float('inf')
    # get ion types of spectrum 1
    m = max(x[1] for x in sp1)
    sp1 = [[x[0], x[1]/m*100.] for x in sp1]
    partedions1 = partitionions(ions1, (kb, ky))

    # get ion types of spectrum 2
    m = max(x[1] for x in sp2)
    sp2 = [[x[0], x[1]/m*100*-1.] for x in sp2]
    partedions2 = partitionions(ions2, (kb, ky))

    # plot spectra
    fig, ax = plt.subplots()
    colors = ('lightgray', 'slategray', 'g', 'r', 'k', 'maroon', 'mediumblue', 'indigo')
    labels = ('Unannotated', "Neutral losses", "b ions", "y ions", "y&b ions",
              "Modified ions", "Precursor", "Immonium ions")
    # .. plots 1
    for i, ixk in enumerate(partedions1):
        mzk, intk = [sp1[j][0] for j in ixk], [sp1[j][1] for j in ixk]
        mk,stline,bsline = axadd(mzk, intk, ax, lc=colors[i], label=labels[i])
    # .. plots 2
    for i, ixk in enumerate(partedions2):
        mzk, intk = [sp2[j][0] for j in ixk], [sp2[j][1] for j in ixk]
        mk,stline,bsline = axadd(mzk, intk, ax, lc=colors[i])

    # set up parameters
##    ylim = ax.set_ylim(bottom=0.)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _ = ax.plot(xlim, [0, 0], lw=0.6, c='k')
    xlim, ylim = ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.tick_params(top='off', right='off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlabel = ax.set_xlabel('m/z', fontname='Times New Roman', fontsize=20, style='italic')
##    xlim = ax.set_xlim((100., 1500.))
    ylabel = ax.set_ylabel('Intensity', fontname='Times New Roman', fontsize=20)
    font = font_manager.FontProperties(family='times new roman', size=10)
    ax.legend(prop=font, frameon=False)
##    plt.savefig(filename, dpi=2000)

    plt.show()


def barmixspectrum(sp, ions1, ions2, seq1, seq2, filename):
    """
    Plot 2 tandem mass spectra for comparison of the 2 spectra
    with fragment ions are highlighed by different ion types
    """
    # spectral normalization
##    m = max(x[1] for x in sp)
##    sp = [[x[0], x[1]/m*100.] for x in sp]

    # get ions using annotation 1
    bix1 = getionidx(ions1, 'b')
    print(bix1)
    yix1 = getionidx(ions1, 'y')
    aix1 = getionidx(ions1, 'a')
    xs1 = bix1.union(yix1)
    xs1 = xs1.union(aix1)
    for i, (mz, ion) in enumerate(ions1):
        if ion and i not in xs1:
            if any(x.startswith('p') and not '-' in x for x in ion.split(',')):
                xs1.add(i)
            else:
                nx = [x.split('/')[0] for x in ion.split(',')
                      if '-' in x and x[0] in 'yb']
                if nx and any(x.count('-')==1 and not '-2' in x for x in nx):
                    xs1.add(i)

    # get ions of using annotation 2
    bix2 = getionidx(ions2, 'b')
    yix2 = getionidx(ions2, 'y')
    aix2 = getionidx(ions2, 'a')
    xs2 = bix2.union(yix2)
    xs2 = xs2.union(aix2)
    # precursor ions and neutral losses
    for i, (mz, ion) in enumerate(ions2):
        if ion and i not in xs2:
            if any(x.startswith('p') and not '-' in x for x in ion.split(',')):
                xs2.add(i)
            else:
                nx = [x.split('/')[0] for x in ion.split(',')
                      if '-' in x and x[0] in 'yb']
                if nx and any(x.count('-')==1 and not '-2' in x for x in nx):
                    xs2.add(i)

    # fragments not annotated
    nix = [i for i in range(len(sp)) if i not in xs1 and i not in xs2]

    # fragments only annotated by any one
    print(xs1, xs2)
    dxs1 = xs1.difference(xs2)
    dxs2 = xs2.difference(xs1)
    xs = xs1&xs2

    # plot spectra
    fig, ax = plt.subplots()
    # , width = 0.001
    if nix:
        mz = [sp[i][0] for i in nix]
        intensity = [sp[i][1] for i in nix]
        mkline, stemlines, bsline = ax.stem(mz, intensity, label=None)
        setlineproperty(bsline, stemlines, mkline, 'lightgray')

    # .. plot b ions
    if dxs1:
        mz = [sp[i][0] for i in dxs1]
        intensity = [sp[i][1] for i in dxs1]
        mkline, stemlines, bsline = ax.stem(mz, intensity, label=seq1)
        setlineproperty(bsline, stemlines, mkline, 'g')

    # .. plot y ions
    if dxs2:
        mz = [sp[i][0] for i in dxs2]
        intensity = [sp[i][1] for i in dxs2]
        mkline, stemlines, bsline = ax.stem(mz, intensity, label=seq2)
        setlineproperty(bsline, stemlines, mkline, 'r')

    # .. plot b and y overlapped ions
    if xs:
        mz = [sp[i][0] for i in xs]
        intensity = [sp[i][1] for i in xs]
        mkline, stemlines, bsline = ax.stem(mz, intensity, label='Common')
        setlineproperty(bsline, stemlines, mkline, 'k')

    # set up parameters
    #ylim = ax.set_ylim(bottom=0.)
    xlabel = ax.set_xlabel('m/z', fontname='Times New Roman', fontsize=20, style='italic')
    ylim = ax.set_ylim(bottom=0.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ylabel = ax.set_ylabel('Intensity', fontname='Times New Roman', fontsize=20)
    font = font_manager.FontProperties(family='times new roman', size=12)
    ax.legend(prop=font, frameon=False)
    fig.set_size_inches(10., 5.)
##    plt.savefig(filename, dpi=1200)

    plt.show()
##    return fig, ax


def beeswarm(xs, ys):
    """
    Bee swarm
    """
    x2 = np.random.normal(1, 0.2, len(ys))
    x1 = np.random.normal(1, 0.2, len(xs))
    fig, ax = plt.subplots()
    _ = ax.plot(x2, ys, 'kx', mew=0.6, ms=5.)
    _ = ax.plot(x1, xs, 'ro', mfc='none', mew=0.6, ms=5.)
    ax.get_xaxis().set_visible(False)
    plt.show()
