# -*- coding: cp1252 -*-
"""
This module is to extract potential false negatives from whole dataset.
"""
import process_masspectrum as pms
import os, csv, random
from itertools import chain, combinations, permutations, combinations_with_replacement
from scorer import score, normalCDF
from collections import Counter
from shutil import copyfile
from math import sqrt, log, log10, pi, exp, copysign
from numpy import histogram
import numpy as np
import NitroDetermine as nitrodeterm
from annotation_spectrum import *
from _bisect import bisect_left, bisect_right
from scipy import stats
from operator import itemgetter
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et
import os, base64, zlib, struct


PTMDB = getmodificationdb()
RESIDUENAMES = {'Asparagine': 'N',
                'Alanine': 'A',
                'Arginine': 'R',
                'Aspartic acid': 'D',
                'Aspartate': 'D',
                'Cysteine': 'C',
                'Glutamic acid': 'E',
                'Glutamate': 'E',
                'Glutamine': 'Q',
                'Glycine': 'G',
                'Histidine': 'H',
                'Isoleucine': 'I',
                'Leucine': 'L',
                'Lysine': 'K',
                'Methionine': 'M',
                'Phenylalanine': 'F',
                'Proline': 'P',
                'Serine': 'S',
                'Threonine': 'T',
                'Tryptophan': 'W',
                'Tyrosine': 'Y',
                'Valine': 'V'}
### Get the mass of the combinations of residues with length
### of 2 to 6
RESIDUES = 'ACDEFGHIKLMNPQRSTVWY'
##MAACOMBINE = {}
##subj, mmj, maj = [], [], []
##for i in xrange(2,5):
##    for jx in combinations_with_replacement(xrange(20), i):
##        subj.append(''.join(RESIDUES[j] for j in jx))
##        mmj.append(sum(AARES[RESIDUES[j]]['mono'] for j in jx))
####        maj.append(sum(AARES[RESIDUES[j]]['avg'] for j in jx))
##MAACOMBINE['seq'] = subj
##MAACOMBINE['monosortedix'] = sorted(xrange(len(mmj)), key=lambda k: mmj[k])
##MAACOMBINE['mono'] = sorted(mmj)
####MAACOMBINE['avgsortedix'] = sorted(xrange(len(maj)), key=lambda k: maj[k])
####MAACOMBINE['avg'] = sorted(maj)

MANUALVALIDATED = [r'./I08/I08-by-1.2-9-1.05.txt', r'./I17/I17-by-1.2-9-1.05.txt',
                   r'./I18/I18-by-1.2-9-1.05.txt', r'./I19/I19-by-1.2-9-1.05.txt']
RAWSETS = ['I08','I17','I18','I19']
FILENUM = [{'1': 'F1-SAX1', '2': 'F1-SAX2', '3': 'F1-Saz1', '4': 'F1-SCX2',
            '5': 'F2-SAX1', '6': 'F2-SAX2', '7': 'F2-SCX1', '8': 'F2-SCX2',
            '9': 'F3-SAX1', '10': 'F3-SAX2', '11': 'F3-SCX1', '12': 'F3-SCX2',
            '13': 'F4-SAX1', '14': 'F4-SAX2', '15': 'F4-SCX1', '16': 'F4-SCX2',
            '17': 'F5-SAX1', '18': 'F5-SAX2', '19': 'F5-SCX1', '20': 'F5-SCX2',
            '21': 'F6-SAX1', '22': 'F6-SAX2', '23': 'F6-SCX1', '24': 'F6-SCX2',
            '25': 'F7-SAX1', '26': 'F7-SAX2', '27': 'F7-SCX1', '28': 'F7-SCX2',
            '29': 'F8-SAX1', '30': 'F8-SAX2', '31': 'F8-SCX1', '32': 'F8-SCX2'},
           {'1': 'F1-SAX', '2': 'F1-SCX', '3': 'F2-SAX', '4': 'F2-SCX',
            '5': 'F3-SAX', '6': 'F3-SCX', '7': 'F4-SAX', '8': 'F4-SCX',
            '9': 'F5-SAX', '10': 'F5-SCX', '11': 'F6-SAX', '12': 'F6-SCX',
            '13': 'F7-SAX', '14': 'F7-SCX', '15': 'F8-SAX', '16': 'F8-SCX'},
           {'1': 'f1-sax', '2': 'f1-scx', '3': 'f2-sax', '4': 'f2-scx',
            '5': 'f3-sax', '6': 'f3-scx', '7': 'f4-sax', '8': 'f4-scx',
            '9': 'f5-sax', '10': 'f5-scx', '11': 'f6-sax', '12': 'f6-scx',
            '13': 'f7-sax', '14': 'f7-scx', '15': 'f8-sax', '16': 'f8-scx'},
           {'1': 'F1-SAX1', '2': 'F1-SAX2', '3': 'F1-SCX1', '4': 'F1-SCX2',
            '5': 'F2-SAX1', '6': 'F2-SAX2', '7': 'F2-SCX1', '8': 'F2-SCX2',
            '9': 'F3-SAX1', '10': 'F3-SAX2', '11': 'F3-SCX1', '12': 'F3-SCX2',
            '13': 'F4-SAX1', '14': 'F4-SAX2', '15': 'F4-SCX1', '16': 'F4-SCX2',
            '17': 'F5-SAX1', '18': 'F5-SAX2', '19': 'F5-SCX1', '20': 'F5-SCX2',
            '21': 'F6-SAX1', '22': 'F6-SAX2', '23': 'F6-SCX1', '24': 'F6-SCX2',
            '25': 'F7-SAX1', '26': 'F7-SAX2', '27': 'F7-SCX1', '28': 'F7-SCX2',
            '29': 'F8-SAX1', '30': 'F8-SAX2', '31': 'F8-SCX1', '32': 'F8-SCX2'}]
##CONF = [94.,93.6,93.8,93.8]
CONF = [87.7, 89.5, 90.2, 90.2]
isby = lambda ionstr: ionstr[0] in 'by' and '-' not in ionstr.split('/')[0]


def parseUniprotPTM():
    """
    Read the PTMs provided by UniProt Kownledgebase
    https://www.uniprot.org/docs/ptmlist
    """
    # initialization
    uniPTM = {}
    for a in RESIDUES: uniPTM[a] = {'name': [], 'mono': []}
    # get modifications
    stag = False
    with open('ptmlist.txt', 'r') as f:
        for line in f:
            if not stag and line.startswith('_______'):
                stag=True
            if stag:
                if line.startswith('ID'):
                    modname = line.rstrip()[2:].lstrip()
                elif line.startswith('TG'):
                    if 'Undefined' in line:
                        res = None
                        continue
                    rx = line.rstrip()[2:].lstrip()
                    rname = rx.split('-')[0] if '-' in rx else rx[:-1]
                    if rname not in RESIDUENAMES:
                        res = None
                        continue
                    if ' or ' in rname:
                        res = [RESIDUENAMES[rk] for rk in rname.split(' or ')]
                    else:
                        res = RESIDUENAMES[rname]
                elif line.startswith('MM'):
                    modmass = float(line.rstrip()[2:].lstrip())
                elif line.startswith('//'):
                    if res is None: continue
                    if isinstance(res, str):
                        uniPTM[res]['name'].append(modname)
                        uniPTM[res]['mono'].append(modmass)
                    else:
                        for rk in res:
                            uniPTM[rk]['name'].append(modname)
                            uniPTM[rk]['mono'].append(modmass)
    return uniPTM


def geometricprob(n, m, k):
    """ curmulative geometric distribution """
    # as the cumulative probability when k equals to m does not
    # exist, set the possible maximum value to it.
    if k==m: return 70.
    # set up constants
    s = [log10(i+1) for i in range(n)]
    s0, s1, s2 = sum(s), sum(s[:m]), sum(s[:n-m])
    smk, sk = sum(s[:m-k]), sum(s[:k])
    # .. constants
    c = 2*s1+2*s2-s0
    # kth hypergeometric PDF
    k1, k2, k3 = k, m-k, n-m-m+k
    sj = sum(s[:k3])
    # curmulative
    gx = []
    for i in range(k+1, m+1):
        if i==m:
            gx.append(s1+s2-s0)
            break
        sk += s[i-1]
        smk -= s[m-i+1]
        sj += s[k3]
        k3 += 1
        gx.append(c-sk-sj-2*smk)

    # negative log transformed cumulative distribution
    m = max(gx)
    #print gx
    # to avoid OverflowError
    g = -m-log10(sum(10**(x-m) for x in gx))
    return g


def lognormcdf(x):
    """ negative logarithm of cumulative normal probability """
    x = abs(x)
    if 0.<=x<=5:
        c1, c2, c3, c4, c5, c6 = -0.0002658, 0.005223, -0.04586, -0.4681, -1.147, -1.001
        p = -(c1*x**5+c2*x**4+c3*x**3+c4*x**2+c5*x+c6)*log10(2.)
    else:
        p = -log10(1.-exp(-1.4*x))+(x**2)/2.*log10(exp(1.))+log10(1.135*sqrt(2*pi)*x)
    if x<0:
        p = -log10(1.-10**(-p))

    return p


def xcorr(spectrum, ions, tol=1.0005):
    """ Sequest's XCorr using fast calculation """
##    isbylocal = isby
    mz = [x[0] for x in spectrum]
    n = len(mz)
    six = sorted(list(range(n)), key=lambda k: mz[k])
    mz = np.array(sorted(mz))
    ints = np.array([spectrum[i][1] for i in six])
    # bin mz
    mz = np.round_(mz/tol)
    umz = np.unique(mz)
    # combine peak intensities
    ints2, rix, k = [], [], 0
    for x in umz:
        mk, jx = 0, None
        for j in range(k, n):
            if mz[j]>x: break
            if ints[j]>mk and mz[k]==x:
                mk = ints[j]
                jx = j
        k = j
        ints2.append(mk)
        rix.append(jx)
    # square and normalize peak intensity
    ints2 = np.sqrt(ints2)
    ints2 /= np.linalg.norm(ints2)
    n2 = len(ints2)
    # XCorr
    xcorr = 0.
    for j,i in enumerate(rix):
        if ions[i][1] and any(isby(xj) for xj in ions[i][1].split(',')):
            xcorr += ints2[j]-sum(ints2[max(0, j-50):min(j+50+1,n2)])/101.
    return xcorr


def getvaluesinbracket(string, label):
    """
    Get values in brackets.
    """
    if label[0] not in string:
        return [], string
    splitstring = string.split(label[0])
    # split string
    vs, cleanstr = [], splitstring[0]
    for x in splitstring[1:]:
        sx = x.split(label[1])
        vs.append((sx[0], len(cleanstr)))
        cleanstr += sx[1]
    return vs, cleanstr


def readms2(ms2file):
    """
    Read ms2 files
    """
    MS = []
    c, seq, mods, spk, isvalid = 0, '', '', [], False
    with open(ms2file, 'r') as f:
        for line in f:
            if line.startswith('S'):
                if spk and isvalid:
                    MS.append((seq, mods, c, scan, mz, spk))
                spk, isvalid = [], False
                sline = line.rstrip().split()
                mz = float(line.rstrip().split()[3])
                scan = int(sline[1])
            elif line.startswith('Z'):
                c = int(line.rstrip().split()[1])
            elif line.startswith('D'):
                sline = line.rstrip().split()
                if sline[1]=='seq':
                    seq = sline[2]
                else:
                    modseq = sline[3]
                    if '[' in modseq:
                        modvals, sk = getvaluesinbracket(modseq, '[]')
                        modsk = []
                        for x, k in modvals:
                            mm = float(x)
                            if mm==16.:
                                modsk.append('Oxidation(%s)@%d'%(sk[k-1], k))
                            if mm==57. or mm==57.1:
                                modsk.append('Carbamidomethyl(%s)@%d'%(sk[k-1], k))
                            if mm==45.:
                                modsk.append('Nitro(%s)@%d'%(sk[k-1], k))
                            if mm==-17.:
                                modsk.append('Gln->pyro-Glu(%s)@%d'%(sk[k-1], k))
                            if mm==42.:
                                modsk.append('Acetyl@N-term' if k==1 else 'Acetyl(%s)@%d'%(sk[k-1], k))
                            if mm==58.:
                                modsk.append('Acetyl@N-term;Oxidation(%s)@%d'%(sk[k-1], k))
                            if mm==99.1:
                                modsk.append('Acetyl@N-term;Carbamidomethyl(%s)@%d'%(sk[k-1], k))
                            if mm==1.:
                                modsk.append('Deamidated(%s)@%d'%(sk[k-1], k))
                        if modseq.count('[')==len(modsk):
                            isvalid = True
                        else:
                            print(modseq)
                        mods = ';'.join(modsk)
                    else:
                        mods, isvalid = '', True
            else:
                sline = line.rstrip().split()
                try:
                    spk.append(list(float(x) for x in sline))
                except:
                    continue
    if spk:
        MS.append((seq, mods, c, scan, mz, spk))
    
    return MS


def randionreplacement(spectrum, ions, randtheoions):
    """
    Replace the m/z of fragments in spectrum with annotations
    in "ions" by same theoretical ions predicted by decoy sequence
    """
    sp2, ions2 = list(spectrum), [(xk[0], '') for xk in spectrum]
    rms = [k for k, xk in enumerate(spectrum) for xz, xn in randtheoions
           if xn in ions[k][1]]
    kx = sorted(set(rms))
    for k in kx:
        ionsk = [xj.split('/') for xj in ions[k][1].split(',')]
        errk = [abs(float(xj[1])) for xj in ionsk]
        j = errk.index(min(errk))
        rmz = [xz for xz, xn in randtheoions if xn==ionsk[j][0]]
        if rmz:
            nmz = rmz[0]+copysign(1., float(ionsk[j][1]))*errk[j]
            sp2[k] = (nmz, spectrum[k][1])
            ions2[k] = (nmz, '/'.join(ionsk[j]))
    
    return sp2, ions2


def adaptivedenoise(spectrum, ions):
    """
    Denoise mass spectrum using the annotated ions using entropy
    methods.
    """
    mz = [x[0] for x in spectrum]
    ints = [x[1] for x in spectrum]
    npeaks = len(mz)
    # re-order mass spectrum with m/z in ascending order
    # divide spectrum into windows of 100 Da
    n = int((mz[npeaks-1]-mz[0])/100.)+1
    assignedions = [1 if xk else 0 for _, xk in ions]
    k, pix = 0, []
    # find the parameters with highest annotated ion fraction
    for i in range(n):
        mzk = mz[0]+(i+1)*100.
        for j in range(k,npeaks):
            if mz[j]>mzk:
                break
        if j==k:
            if mz[j]<=mzk and ions[k][1]:
                pix.append(j)
            continue
        # .. parameter with highest annotation fraction
        sixk = sorted(list(range(k, j)), key=lambda a: ints[a], reverse=True)
        ionsc = [assignedions[jk] for jk in sixk]
        pk = [sum(ionsc[:jk]) for jk in range(1, min(len(ionsc)+1, 9))]
        mjk = pk.index(max(pk))
        # top number of peaks with maximum fraction of annotations
        pix += sixk[:mjk+1]
        k = j
    
    pix.sort()
    return pix


def adaptivedenoise2(spectrum, ionindicators):
    """
    Denoise mass spectrum using the annotated ions using entropy
    methods.
    """
    mz = [x[0] for x in spectrum]
    ints = [x[1] for x in spectrum]
    npeaks = len(mz)
    # re-order mass spectrum with m/z in ascending order
    # divide spectrum into windows of 100 Da
    n = int((mz[npeaks-1]-mz[0])/100.)+1
    k, pix = 0, []
    # find the parameters with highest annotated ion fraction
    for i in range(n):
        mzk = mz[0]+(i+1)*100.
        for j in range(k,npeaks):
            if mz[j]>mzk:
                break
        if j==k:
            if mz[j]<=mzk and ionindicators[k]:
                pix.append(j)
            continue
        # .. parameter with highest annotation fraction
        sixk = sorted(list(range(k, j)), key=lambda a: ints[a], reverse=True)
        ionsc = [ionindicators[jk] for jk in sixk]
        pk = [sum(ionsc[:jk]) for jk in range(1, min(len(ionsc)+1, 9))]
        mjk = pk.index(max(pk))
        # top number of peaks with maximum fraction of annotations
        pix += sixk[:mjk+1]
        k = j
    
    pix.sort()
    return pix


def thresholdscore(truescore, decoyscore, FDR = 0.01):
    """
    Calculate threshold score under the set FDR.
    """
    truescore = sorted(truescore)
    decoyscore = sorted(decoyscore)
    allscore = sorted(list(truescore)+list(decoyscore), reverse=True)
    nt, nd = len(truescore), len(decoyscore)
##    cx = []
    for x in allscore:
        k1 = bisect_left(truescore, x)
        k2 = bisect_left(decoyscore, x)
        if nt-k1==0:
            cfdr = 1.
        else:
            cfdr = (nd-k2)/float(nt-k1)
##        cx.append(cfdr)
        if cfdr>=FDR:
            return x
    return x


def longestsequence(indices, dist=1):
    """
    Get the longest consecutive sequences in which the distance
    between adjacent elements is "distance".
    Return:
    maxlen: the length of longest sequence
    lgseq: the longest sequence
    
    Note:
    This only works for the sorted list
    """
    if not indices:  return 0, None
    
    maxlen, R = -1, {}
    for x in indices:
        R[x] = R.get(x-dist, 0)+1
        k = R[x]
        if k>maxlen:
            maxlen, lgend = k, x
    return maxlen, list(range(lgend-maxlen*dist+dist, lgend+dist, dist))


def nearestcentroids(X, y, Xt):
    """
    Classify the testing data according to the training set
    using nearest centroids.
    References:
    1. Tibshirani R, et al. Proc Natl Acad Sci U S A. 2002, 99(10), 6567–6572.
    2. Tibshirani R, et al. Stat Sci. 2003, 18(1), 104–117.
    """
    cls, N, C = set(y), float(len(y)), Counter(y)
    K = len(cls)
    mx = np.mean(X, axis=0)
    sx = np.zeros(X.shape[1])
    # centroids, priors and t-statistics
    mxk, mk, priors = [], [], []
    for cj in cls:
        xj = X[y==cj, :]
        mxj = np.mean(xj, axis=0)
        mxk.append(mxj)
        sx += np.sum((xj-mxj)**2, axis=0)
        mk.append(sqrt(1./C[cj]-1./N))
        priors.append([C[cj]/N for cj in c])
    sx /= N-K
    # prediction
    sk = sqrt(sx)
    s0 = pms.median(sk)
    ds = []
    for i in range(K):
        # t-statistic of class i
        mxk2 = (mxk[i]-mx)*((mk[i]*sk)/(mk[i]*sk+s0))+mx
        ds.append(np.sum((Xt-mxk2)**2/sx)-2*log(priors[i]))

    # convert scores to class probabilities and classes
    nt, _ = Xt.shape
    posts, yt = [], []
    for i in range(nt):
        dsk = [x[i] for x in ds]
        yt.append(dsk.index(max(dsk)))
        ek = []
        for j in range(K):
            ek.append(1./sum(exp((dsk[j]-xj)/2.) for xj in dsk))
        posts.append(ek)

    return yt, posts
    

def decodebinary(string, defaultArrayLength, precision=64, bzlib='z'):
    """
    Decode binary string to float points.
	If provided, should take endian order into consideration.
    """
    decoded = base64.decodestring(string)
    decoded = zlib.decompress(decoded) if bzlib == 'z' else decoded
    unpack_format = "<%dd"%defaultArrayLength if precision == 64 else \
        "<%dL"%defaultArrayLength
    return struct.unpack(unpack_format, decoded)


def convertmod(modstr):
    """
    Convert modification string in Sequest and Mascot search results
    to ProteinPilot type
    """
    if not modstr:
        return modstr
    cmod = []
    for x in modstr.replace(' ','').split(';'):
        sx = x.split('(')
        if x.startswith('N-') or x.startswith('C-'):
            cmod.append('%s@%s'%(sx[1].split(')')[0], sx[0]))
        else:
            cmod.append('%s(%s)@%s'%(sx[1].split(')')[0], sx[0][0], sx[0][1:]))
    return ';'.join(cmod)


def allions(seqms, seq, modifications, c, MZTOL=0.2, mtype='mono', TOLUNIT='Da'):
    """
    Get names of fragment ions in spectrum according to the specified
    peptide sequence "sequence", modifications and charge state
    """
    mnh3, mh2o, mco, mh = 16.998767, 18.006067, 28.0101, 1.0073
    ntm, resm, ctm = seqms
    bs, ys, pmass = generatetheoions(resm, ntermmass=ntm, ctermmass=ctm)
    baions, ynions = [], []
    _ = [baions.extend([(bk+mh, 'b%d[+]'%(i+1), i+1),
                         (bk+mh-mnh3, 'b%d-NH3[+]'%(i+1), i+1),
                         (bk+mh-mh2o, 'b%d-H2O[+]'%(i+1), i+1),
                         (bk+mh-mco, 'a%d[+]'%(i+1), i+1)])
         for i,bk in enumerate(bs)]
    _ = [ynions.extend([(yk+mh, 'y%d[+]'%(i+1), i+1),
                        (yk+mh-mnh3, 'y%d-NH3[+]'%(i+1), i+1),
                        (yk+mh-mh2o, 'y%d-H2O[+]'%(i+1), i+1)])
         for i,yk in enumerate(ys)]
    
    # other charges
    if c>=2:
        baions2 = [((bk+mh)/2., bm.replace('+', '2+'), ij) for bk, bm, ij in baions[2:]]
        ynions2 = [((yk+mh)/2., ym.replace('+', '2+'), ij) for yk, ym, ij in ynions[2:]]
        if c>=3 and len(seq)>=5:
            baions3 = [((bk+2*mh)/3., bm.replace('+', '3+'), ij)
                       for bk, bm, ij in baions[4:]]
            ynions3 = [((yk+2*mh)/3., ym.replace('+', '3+'), ij)
                       for yk, ym, ij in ynions[4:]]
            ynions += ynions3
            baions += baions3
        ynions += ynions2
        baions += baions2

    ions = ynions+baions
    # precursor ion series
    pions = []
    _ = [pions.extend([(pmass/(i+1)+mh, 'p[%d+]'%(i+1), len(seq)),
                       ((pmass-mh2o)/(i+1)+mh, 'p-H2O[%d+]'%(i+1), len(seq)),
                       ((pmass-mnh3)/(i+1)+mh, 'p-NH3[%d+]'%(i+1), len(seq))])
         for i in range(c)]
    if modifications and\
       any(nj=='iTRAQ8plex' and isinstance(sj, str) for mj, sj, nj in modifications):
            pions.append((pmass+mh-304.20536, 'p-iT8[+]', len(seq)))
    ions += pions

    # immonium ions
    immions = generateimmoniums(resm)
    kx = []
    if modifications:
        kx = [i-1 for _, i, _ in modifications if isinstance(i, int)]
    # .. add immonium ions into ion list
    ions.extend(set([(immions[i], 'imm(%s)'%('%s*'%x if i in kx else x), 0)
                     for i,x in enumerate(seq)]))
    
    return ions


def simpleannotation(theoions, spx, tol=0.2):
    """
    Simple version of annotations
    """
    mz = [x[0] for x in spx]
    l = len(spx)
    mix = [bisect_left(mz, mj) for mj, _, _ in theoions]
    # annotations
    return dict((mjname, ((k-1, mj-mz[k-1], ij)
                          if k>0 and mj-mz[k-1]<=tol
                          else (k, mj-mz[k], ij)))
         for k, (mj, mjname, ij) in zip(mix, theoions)
         if (k>0 and mj-mz[k-1]<=tol) or (k<l and mz[k]-mj<=tol))


def simpleannotation2(theoions, spx, tol=0.2):
    """
    Simple version of annotations
    """
    mz = [x[0] for x in spx]
    #theoions = sorted(theoions, key=itemgetter(0))
    theomz = [vk for vk, _, _ in theoions]
    mix = [bisect_left(mz, mj-tol) for mj in theomz]
    # annotations
    ions = []
    _ = [ions.extend([(theoions[ik][1], (k, mj-mz[k], theoions[ik][2]))
                     for k in range(i, bisect_right(mz, mj+tol, lo=i))])
         for ik, (i, mj) in enumerate(zip(mix, theomz))]
    return dict(ions)


def theoionmzall(seqms, seq, modifications, c):
    """
    Get m/z values of theoretical fragments
    """
    mnh3, mh2o, mco, mh = 16.998767, 18.006067, 28.0101, 1.0073
    ntm, resm, ctm = seqms
    bs, ys, pmass = generatetheoions(resm, ntermmass=ntm, ctermmass=ctm)
    baions, ynions, ps = [], [], []
    _ = [baions.extend([bk+mh, bk+mh-mnh3, bk+mh-mh2o, bk+mh-mco]) for bk in bs]
    _ = [ynions.extend([yk+mh, yk+mh-mnh3, yk+mh-mh2o]) for yk in ys]
    _ = [ps.extend([pmass/(cj+1)+mh, (pmass-mnh3)/(cj+1)+mh, (pmass-mh2o)/(cj+1)+mh])
         for cj in range(min(c,3))]
    # p-IT8
    if modifications:
        for mj, sj, nj in modifications:
            if nj=='iTRAQ8plex' and isinstance(sj, str):
                ps.append(pmass+mh-mj)
                break
    # other charges
    if c>=2:
        baions2 = [(bk+mh)/2. for bk in baions[2:]]
        ynions2 = [(yk+mh)/2. for yk in ynions[2:]]
        if c>=3 and len(seq)>=5:
            baions3 = [(bk+2*mh)/3. for bk in baions[4:]]
            ynions3 = [(yk+2*mh)/3. for yk in ynions[4:]]
            ynions += ynions3
            baions += baions3
        ynions += ynions2
        baions += baions2
    # immonium ions
    immions = [mj-mco+mh for mj in resm[1:]]
    
    return sorted(set(baions+ynions+immions+ps))


def theoionmzall_by(seqms, seq, modifications, c):
    """
    Get m/z values of theoretical fragments, only consider b and y ions
    """
    mnh3, mh2o, mco, mh = 16.998767, 18.006067, 28.0101, 1.0073
    ntm, resm, ctm = seqms
    bs, ys, pmass = generatetheoions(resm, ntermmass=ntm, ctermmass=ctm)
    baions = []
    _ = [baions.extend([bk+mh, bk+mh-mco]) for bk in bs]
    yions = [yk+mh for yk in ys]
    ps = [pmass/(cj+1)+mh for cj in range(min(c,3))]
    # p-IT8
    if modifications and \
       any(nj=='iTRAQ8plex' and isinstance(sj, str) for mj, sj, nj in modifications):
            ps.append(pmass+mh-304.20536)
    # other charges
    if c>=2:
        baions2 = [(bk+mh)/2. for bk in baions[4:]]
        yions2 = [(yk+mh)/2. for yk in yions[2:]]
        if c>=3 and len(seq)>=5:
            baions3 = [(bk+2*mh)/3. for bk in baions[8:]]
            yions3 = [(yk+2*mh)/3. for yk in yions[4:]]
            yions += yions3
            baions += baions3
        yions += yions2
        baions += baions2
    # immonium ions
    immions = [mj-mco+mh for mj in resm[1:]]
    
    return sorted(set(baions+yions+immions+ps))


def getmodions(ions, seq, modifications, modtype):
    """
    Get indices of modified fragments.
    """
    l = len(seq)
    kx = [k for _, k, x in modifications if x==modtype]
    kb, ky = min(kx), min(l-k+1 for k in kx)
    # modified residue
    r = [seq[k-1] for k in kx][0]
    # modified fragments
    # .. modified y and b ions
    modfrags = ['b%d'%(k+1) for k in range(l) if k+1>=kb]
    modfrags += ['a%d'%(k+1) for k in range(l) if k+1>=kb]
    modfrags += ['y%d'%(k+1) for k in range(l) if k+1>=ky]
    # .. p ions
    modfrags.append('p')
    # return types
    its = dict((ky, 2) if '%s*'%r in ky or any(xk in ky for xk in modfrags) else (ky, 1)
               for ky, val in list(ions.items()))
    return its


def ionscores(wmz, wint, c, mzerr, relint, mzsigma):
    """
    Scores of ions
    """
    g = mzsigma/2.
    #z = 2./(g * sqrt(2.*pi))
    s = 0.
    for i, x in enumerate(mzerr):
        #s += z*wmz*abs(0.5-normalCDF(x/g))+wint*relint[i]**c
        s += wmz*2*(1.-normalCDF(abs(x)/g))+wint*relint[i]**c
    return s
    

def revseq(seq, mod):
    """
    Reverse and rotate sequence to serve as null sequences.
    """
    revseqs, revmods = [], []
    l = len(seq)-1
    seqix = list(range(l+1))

    # parse modifications
    modsites = [int(x.split('@')[1])-1 for x in mod.split(';')
                if not x.split('@')[1].lower()[0] in 'nc']
    modname = [x.split('@')[0] for x in mod.split(';')
               if not x.split('@')[1].lower()[0] in 'nc']
    modleft = [x for x in mod.split(';') if x.split('@')[1].lower()[0] in 'nc']
    # .. remove the modication locates in the last amino acid
    if l in modsites:
        modleft.append('%s@%d'%(modname[modsites.index(l)], l+1))
        modname = [x for i, x in enumerate(modname) if modsites[i]!=l]
        modsites = [i for i in modsites if i!=l]
    
    # reverse and rotate the sequence
    rix = seqix[:l][::-1]
    cseq = ''.join(seq[i] for i in rix)+seq[l]
    if cseq!=seq:
        rndseqs.append(cseq)
        modx = ['%s@%d'%(modname[modsites.index(i)], j+1) for j, i in enumerate(rix)
                if i in modsites]
        rndmods.append(';'.join(modleft+modx))
    rj = rix
    for i in range(l-1):
        rj = rj[l-1:]+rj[:l-1]
        cseq = ''.join(seq[i] for i in rj)+seq[l]
        if cseq!=seq:
            revseqs.append(cseq)
            modx = ['%s@%d'%(modname[modsites.index(i)], j+1) for j, i in enumerate(rj)
                    if i in modsites]
            revmods.append(';'.join(modleft+modx))

    return revseqs, revmods


def randomseq(seq, mod, tol, mtype='mono'):
    """
    Generate random sequences for peptide with modifications but keep
    C-terminal amino acid to retain the termini of enzyme cleavage
    and 3-nitrotyrosine.
    """
    randsample = random.sample
    seq = seq.replace('L', 'I')
    # get residue mass
    aas, hmass = 'ACDEFGHIKMNPQRSTVWY', 1.0073
    aam = {}
    for aakey in aas:
        aam[aakey] = AARES[aakey][mtype]
    # combinations of residues
    mcb = MAACOMBINE[mtype]
    ncb = len(mcb)
    six = MAACOMBINE[mtype+'sortedix']
    seqcb = [MAACOMBINE['seq'][i] for i in six]
    seqcbs = [set(MAACOMBINE['seq'][i]) for i in six]
    # min and max mass of residue combinations
    m0, m2 = mcb[0], mcb[ncb-1]
    rgix, rgm, k, rgk = [], [], 0, 20.
    for i in range(1, int(m2/rgk)+1):
        for j in range(k, ncb):
            if mcb[j]>m0+i*rgk:
                break
        if j!=k:
            rgix.append(j)
            rgm.append(mcb[j])
        k = j
    if rgix[0]>0:
        rgix.insert(0, 0)
        rgm.insert(0, m0)
    
    rndseq, rndmod = [], []

    # parse modifications
    modx = parsemodifications(mod, PTMDB, mtype=mtype)
    sx = [k-1 for _, k, _ in modx if isinstance(k, int)]

    # get the residues having no 3-nitrotyrosine and C-terminus
    l, seqlist = len(seq), list(seq)
##    seqfx = [i for i,x in enumerate(seq)
##             if x=='Y' and (i not in sx or 'Nitro(Y)@%d'%(i+1) in mod)]
    seqfx = [i for i,x in enumerate(seq)
             if x in 'DE' and
             (i not in sx
              or 'Methyl(D)@%d'%(i+1) in mod
              or 'Methyl(E)@%d'%(i+1) in mod)]
    if l-1 not in seqfx: seqfx.append(l-1)
    seqrx = [i for i in range(l) if i not in seqfx]
    
    # get modifications
    modr, modrix, modnc, modvar, md = [], [], [], [], []
    for xi, k, xj in modx:
        if isinstance(k, str):
            modnc.append((xi, k, xj))
        else:
            if k-1 not in seqfx:
                modr.append((xi, xj))
                modrix.append(k-1)
            elif k == l:
                md.append((xi, k, xj))
            else:
                modvar.append((xi, len(seqfx)-seqfx.index(k-1), xj))
    seqr = ''.join(seqlist[i] for i in seqrx)

    # .. get the mass of remaining sequence part
    sxm, modrixn = [], []
    for i, a in enumerate(seqr):
        if seqrx[i] in modrix:
            j = modrix.index(seqrx[i])
            modrixn.append(i)
            sxm.append(aam[a]+modr[j][0])
        else:
            sxm.append(aam[a])

    # generate random sequences by sequence substitutions
    aams = tuple(aam.values())
    aam1, aam2 = min(aams), max(aams)
    # .. generate tags for substitution
    subseqs, stseqs, rtseqs = [], [], []
    lk, subix = len(seqrx), []
    for i in range(2, lk):
        for j in range(lk-i):
            qk = seqr[j:j+i+1]
            if qk not in stseqs:
                m, sqk = sum(sxm[j:j+i+1]), set(qk)
                ml, mu = m-tol, m+tol
                if mu>m2: continue
                # .. .. single array of combinations
                k0, k2 = -1, -1
                for k, x in enumerate(rgm):
                    if ml>x: k0 = k
                    if mu<=x: k2 = k; break
                for k in range(rgix[k0], rgix[k2]):
                    if mcb[k]>=ml and mcb[k]<=mu and len(sqk&seqcbs[k])<i-1:
                        subseqs.append(seqcb[k])
                        subix.append((j, j+i+1))
                    elif mcb[k]>mu:
                        break
                stseqs.append(qk)

##    # ===============
##    # consider the peptides with only part of the sequence is replaced
##    # but keep all other short sequence
##    rtseqs, rtmods, subix2 = [], [], []
##    for ij, (i, j) in enumerate(subix):
##        if j==len(seqrx):
##            jl, ik, jk = len(subseqs[ij]), seqrx[i], l-1
##        else:
##            jl, ik, jk = len(subseqs[ij]), seqrx[i], seqrx[j]
##        if jk<=seqfx[0]:
##            rtseqs.append(seq[:ik]+subseqs[ij]+seq[jk:])
##            modk = []
##            for xi, k, xj in modx:
##                if isinstance(k, str) or k-1<ik:
##                    modk.append((xi, k, xj))
##                elif k>=seqrx[j]:
##                    modk.append((xi, k-jk+ik+jl, xj))
##            rtmods.append(modk)
##            subix2.append(ij)
##        elif len(seqfx)>=2:
##            # .. between the two fixed residues
##            kx = [(ik, jk) for k1, k2 in zip(seqfx[:-1], seqfx[1:])
##                  if ik>k1 and jk<=k2]
##            if kx:
##                k1, k2 = kx[0]
##                rtseqs.append(seq[:k1]+subseqs[ij]+seq[k2:])
##                modk = []
##                for xi, k, xj in modx:
##                    if isinstance(k, str) or k-1<ik:
##                        modk.append((xi, k, xj))
##                    elif k>=k2:
##                        modk.append((xi, k-jk+ik+jl, xj))
##                rtmods.append(modk)
##                subix2.append(ij)
##    # also consider replacing two parts of the sequences
##    if len(subix2)>=2:
##        for ij1, ij2 in zip(subix2[:-1], subix2[1:]):
##            if subix[ij1][1]==len(seqrx):
##                i1, i2 = seqrx[subix[ij1][0]], l-1
##            else:
##                i1, i2 = seqrx[subix[ij1][0]], seqrx[subix[ij1][1]]
##            if subix[ij2][1]==len(seqrx):
##                j1, j2 = seqrx[subix[ij2][0]], l-1
##            else:
##                j1, j2 = seqrx[subix[ij2][0]], seqrx[subix[ij2][1]]
##            if i1>=j2 or j1>=i2:
##                k1, k2, k3, k4, kj1, kj2 = (j1, j2, i1, i2, ij2, ij1) if i1>=j2\
##                                           else (i1, i2, j1, j2, ij1, ij2)
##                rtseqs.append(seq[:k1]+subseqs[kj1]+seq[k2:k3]+subseqs[kj2]+seq[k4:])
##                l1, l2 = len(subseqs[kj1]), len(subseqs[kj2])
##                modk = []
##                for xi, k, xj in modx:
##                    if isinstance(k, str) or k-1<k1:
##                        modk.append((xi, k, xj))
##                    elif k2<=k<k3:
##                        modk.append((xi, k-k2+k1+l1, xj))
##                    elif k>=k3:
##                        modk.append((xi, k-k2-k3+k1+l1+l2+k3-k2, xj))
##                rtmods.append(modk)
##    return  rtseqs, rtmods
##    # ===============
    
    # if number of substitutions is larger than 2000, use 2000
    n = len(subseqs)
    if n>2000:
        idx = sorted(randsample(list(range(n)), 2000))
        nperm = 5
        subseqs = [subseqs[i] for i in idx]
        subix = [subix[i] for i in idx]
    elif n<=10:
        return None, None
    else:
        nperm = min(int(10000./n), 60)
        
    # .. generate random sequences
    seq2, qe = seqr+''.join(seq[i] for i in seqfx), seqlist[-1]
    for i, sj in enumerate(subseqs):
        # .. .. prepare the seed of random sequences
        j1, j2 = subix[i]
        seqx = sj+seq2[j2:]
        if j1>0: seqx = seq2[:j1]+seqx
        lj = len(seqx)
        qx, modj = tuple(range(lj-1)), list(modnc)
        modvar_t = []
        if modvar:
            modvar_t += [(xi, lj-k, xj) for xi, k, xj in modvar]
        for k, kj in enumerate(modrixn):
            if kj>=j2:
                modvar_t.append((modr[k][0], kj-j2+j1+len(sj), modr[k][1]))
            elif kj<j1:
                modvar_t.append((modr[k][0], kj, modr[k][1]))
        if md:
            modj.append((md[0][0], lj, md[0][2]))

        # .. .. permutations of substituted peptides
        for j in range(nperm+1):
            jx = randsample(qx, lj-1)
            modxj = list(modj)
            if modvar_t:
                modxj += [(xi, jx.index(k)+1, xj) for xi, k, xj in modvar_t]
            qi = ''.join(seqx[k] for k in jx)+qe
            if qi not in rndseq:
                rndseq.append(qi)
                rndmod.append(modxj)

    return rndseq, rndmod


def randomseq_x(seq, mod, tol, objmod, keepStructure=True, mtype='mono'):
    """
    Generate random sequences for peptide with modifications but keep
    C-terminal amino acid to retain the termini of enzyme cleavage
    and 3-nitrotyrosine.
    """
    randsample = random.sample
    #seq = seq.replace('L', 'I')
    # get residue mass
    aas, hmass = 'ACDEFGHIKMNPQRSTVWY', 1.0073
    aam = {}
    for aakey in aas:
        aam[aakey] = AARES[aakey][mtype]
    # combinations of residues
    mcb = MAACOMBINE[mtype]
    ncb = len(mcb)
    six = MAACOMBINE[mtype+'sortedix']
    seqcb = [MAACOMBINE['seq'][i] for i in six]
    seqcbs = [set(MAACOMBINE['seq'][i]) for i in six]
    # min and max mass of residue combinations
    m0, m2 = mcb[0], mcb[ncb-1]
    rgix, rgm, k, rgk = [], [], 0, 20.
    for i in range(1, int(m2/rgk)+1):
        for j in range(k, ncb):
            if mcb[j]>m0+i*rgk:
                break
        if j!=k:
            rgix.append(j)
            rgm.append(mcb[j])
        k = j
    if rgix[0]>0:
        rgix.insert(0, 0)
        rgm.insert(0, m0)
    
    rndseq, rndmod = [], []

    # get the objective residue
    objaa = objmod[-2]
    #if objaa=='L': objaa='I'

    # parse modifications
    modx = parsemodifications(mod, PTMDB, mtype=mtype)
    sx = [k-1 for _, k, _ in modx if isinstance(k, int)]

    # get the residues having no 3-nitrotyrosine and C-terminus
    l, seqlist = len(seq), list(seq)
    if keepStructure:
        seqfx = [i for i,x in enumerate(seq)
                 if x==objaa and (i not in sx or '%s@%d'%(objmod, i+1) in mod)]
    else:
        seqfx = [i for i,x in enumerate(seq)
                 if x==objaa and '%s@%d'%(objmod, i+1) in mod]
    #print seqfx, seq
    if l-1 not in seqfx: seqfx.append(l-1)
    seqrx = [i for i in range(l) if i not in seqfx]
    
    # get modifications
    modr, modrix, modnc, modvar, md = [], [], [], [], []
    for xi, k, xj in modx:
        if isinstance(k, str):
            modnc.append((xi, k, xj))
        else:
            if k-1 not in seqfx:
                modr.append((xi, xj))
                modrix.append(k-1)
            elif k == l:
                md.append((xi, k, xj))
            else:
                modvar.append((xi, len(seqfx)-seqfx.index(k-1), xj))
    seqr = ''.join(seqlist[i] for i in seqrx)

    # .. get the mass of remaining sequence part
    sxm, modrixn = [], []
    for i, a in enumerate(seqr):
        ak = 'I' if a=='L' else a
        if seqrx[i] in modrix:
            j = modrix.index(seqrx[i])
            modrixn.append(i)
            sxm.append(aam[ak]+modr[j][0])
        else:
            sxm.append(aam[ak])

    # generate random sequences by sequence substitutions
    # .. generate tags for substitution
    subseqs, stseqs, rtseqs = [], [], []
    lk, subix = len(seqrx), []
    for i in range(2, lk):
        for j in range(lk-i):
            qk = seqr[j:j+i+1]
            if qk not in stseqs:
                m, sqk = sum(sxm[j:j+i+1]), set(qk)
                ml, mu = m-tol, m+tol
                if mu>m2: continue
                # .. .. single array of combinations
                k0, k2 = -1, -1
                for k, x in enumerate(rgm):
                    if ml>x: k0 = k
                    if mu<=x: k2 = k; break
                for k in range(rgix[k0], rgix[k2]):
                    if mcb[k]>=ml and mcb[k]<=mu and len(sqk&seqcbs[k])<max(i-2, 1):
                        subseqs.append(seqcb[k])
                        subix.append((j, j+i+1))
                    elif mcb[k]>mu:
                        break
                stseqs.append(qk)
    
    # if number of substitutions is larger than 2000, use 2000
    n = len(subseqs)
    print(n)
    if n>2000:
        idx = sorted(randsample(list(range(n)), 2000))
        nperm = 5
        subseqs = [subseqs[i] for i in idx]
        subix = [subix[i] for i in idx]
    elif n<=10:
        return None, None
    else:
        nperm = min(int(10000./n), 60)
        
    # .. generate random sequences
    seq2, qe = seqr+''.join(seq[i] for i in seqfx), seqlist[-1]
    for i, sj in enumerate(subseqs):
        # .. .. prepare the seed of random sequences
        j1, j2 = subix[i]
        seqx = sj+seq2[j2:]
        if j1>0: seqx = seq2[:j1]+seqx
        lj = len(seqx)
        qx, modj = tuple(range(lj-1)), list(modnc)
        modvar_t = []
        if modvar:
            modvar_t += [(xi, lj-k, xj) for xi, k, xj in modvar]
        for k, kj in enumerate(modrixn):
            if kj>=j2:
                modvar_t.append((modr[k][0], kj-j2+j1+len(sj), modr[k][1]))
            elif kj<j1:
                modvar_t.append((modr[k][0], kj, modr[k][1]))
        if md:
            modj.append((md[0][0], lj, md[0][2]))

        # .. .. permutations of substituted peptides
        for j in range(nperm+1):
            jx = randsample(qx, lj-1)
            modxj = list(modj)
            if modvar_t:
                modxj += [(xi, jx.index(k)+1, xj) for xi, k, xj in modvar_t]
            qi = ''.join(seqx[k] for k in jx)+qe
            if qi not in rndseq:
                rndseq.append(qi)
                rndmod.append(modxj)

    return rndseq, rndmod


def randomseqwithMod(seq, mod, tol, objmod, keepStructure=True, mtype='mono'):
    """
    Generate random sequences for peptide with modifications but keep
    C-terminal amino acid to retain the termini of enzyme cleavage
    and 3-nitrotyrosine.
    """
    # get residue mass
    aas, hmass = RESIDUES, 1.0073
    aam = {}
    for aakey in aas: aam[aakey] = AARES[aakey][mtype]
    # combinations of residues
    mcb = MAACOMBINE[mtype]
    ncb = len(mcb)
    six = MAACOMBINE[mtype+'sortedix']
    seqcb = [MAACOMBINE['seq'][i] for i in six]
    # min and max mass of residue combinations
    
    rndseq, rndmod = [], []

    # get the objective residue
    objaa = objmod[-2]

    # parse modifications
    modx = parsemodifications(mod, PTMDB, mtype=mtype)
    sx = [k-1 for _, k, _ in modx if isinstance(k, int)]

    # get the residues having no 3-nitrotyrosine and C-terminus
    l, seqlist = len(seq), list(seq)
    if keepStructure:
        seqfx = [i for i,x in enumerate(seq)
                 if x==objaa and (i not in sx or '%s@%d'%(objmod, i+1) in mod)]
    else:
        seqfx = [i for i,x in enumerate(seq)
                 if x==objaa and '%s@%d'%(objmod, i+1) in mod]
    if l-1 not in seqfx: seqfx.append(l-1)
    seqrx = [i for i in range(l) if i not in seqfx]
    
    # get modifications
    modr, modrix, modnc, modvar, md = [], [], [], [], []
    for xi, k, xj in modx:
        if isinstance(k, str):
            modnc.append((xi, k, xj))
        else:
            if k-1 not in seqfx:
                modr.append((xi, xj))
                modrix.append(k-1)
            elif k == l:
                md.append((xi, k, xj))
            else:
                modvar.append((xi, len(seqfx)-seqfx.index(k-1), xj))
    seqr = ''.join(seqlist[i] for i in seqrx)

    # .. get the mass of remaining sequence part
    sxm, modrixn = [], []
    for i, a in enumerate(seqr):
        ak = 'I' if a=='L' else a
        if seqrx[i] in modrix:
            j = modrix.index(seqrx[i])
            modrixn.append(i)
            sxm.append(aam[ak]+modr[j][0])
        else:
            sxm.append(aam[ak])

    # generate random sequences by sequence substitutions
    aams = tuple(aam.values())
    aam1, aam2 = min(aams), max(aams)
    # .. generate tags for substitution
    subseqs, stseqs, rtseqs = [], [], []
    lk, subix = len(seqrx), []
    for i in range(2, lk):
        for j in range(lk-i):
            qk = seqr[j:j+i+1]
            if qk not in stseqs:
                m, sqk = sum(sxm[j:j+i+1]), set(qk)
                ml, mu = m-tol, m+tol
                if mu>m2: continue
                # .. .. single array of combinations
                k0, k2 = -1, -1
                for k, x in enumerate(rgm):
                    if ml>x: k0 = k
                    if mu<=x: k2 = k; break
                for k in range(rgix[k0], rgix[k2]):
                    if mcb[k]>=ml and mcb[k]<=mu and len(sqk&seqcbs[k])<i-1:
                        subseqs.append(seqcb[k])
                        subix.append((j, j+i+1))
                    elif mcb[k]>mu:
                        break
                stseqs.append(qk)
    
    # if number of substitutions is larger than 2000, use 2000
    n = len(subseqs)
    if n>2000:
        idx = sorted(randsample(list(range(n)), 2000))
        nperm = 5
        subseqs = [subseqs[i] for i in idx]
        subix = [subix[i] for i in idx]
    elif n<=10:
        return None, None
    else:
        nperm = min(int(10000./n), 60)
        
    # .. generate random sequences
    seq2, qe = seqr+''.join(seq[i] for i in seqfx), seqlist[-1]
    for i, sj in enumerate(subseqs):
        # .. .. prepare the seed of random sequences
        j1, j2 = subix[i]
        seqx = sj+seq2[j2:]
        if j1>0: seqx = seq2[:j1]+seqx
        lj = len(seqx)
        qx, modj = tuple(range(lj-1)), list(modnc)
        modvar_t = []
        if modvar:
            modvar_t += [(xi, lj-k, xj) for xi, k, xj in modvar]
        for k, kj in enumerate(modrixn):
            if kj>=j2:
                modvar_t.append((modr[k][0], kj-j2+j1+len(sj), modr[k][1]))
            elif kj<j1:
                modvar_t.append((modr[k][0], kj, modr[k][1]))
        if md:
            modj.append((md[0][0], lj, md[0][2]))

        # .. .. permutations of substituted peptides
        for j in range(nperm+1):
            jx = randsample(qx, lj-1)
            modxj = list(modj)
            if modvar_t:
                modxj += [(xi, jx.index(k)+1, xj) for xi, k, xj in modvar_t]
            qi = ''.join(seqx[k] for k in jx)+qe
            if qi not in rndseq:
                rndseq.append(qi)
                rndmod.append(modxj)

    return rndseq, rndmod



def generateby(sequence, mods, mtype='mono'):
    """
    Generate b and y ions
    """
    seqmass, bions, yions = [], [], []
    mh = NEUTRALS['h']
    n, mn, mc = len(sequence), MTERMINUS['nterm'][mtype], MTERMINUS['cterm'][mtype]
    modseq, cmod, nmod = [0.]*n, False, False
    if mods:
        for m, j, _ in mods:
            if isinstance(j, int):
                modseq[j-1] += m
            else:
                if j == 'cterm':
                    cmod, mc = True, m
                else:
                    nmod, mn = True, m
    for i, aa in enumerate(sequence):
        seqmass.append(AARES[aa][mtype]+modseq[i])

    # y ions
    yi = mc if cmod else mc+mh
    for residuemass in seqmass[::-1]:
        yi += residuemass
        yions.append(yi)
    yions = yions[:n-1]

    # b ions
    bi = mn if nmod else 0.
    for residuemass in seqmass:
        bi += residuemass
        bions.append(bi)
    bions = bions[:n-1]

    return yions, bions
            
            
def nitroionmatch(spectrum, sequence, mods, charge, tol, mtype='mono'):
    """
    Generate null scores for statistical inferences of normal scores
    """
    # generate b and y ions
    ys, bs = generateby(sequence, mods)
    # only consider y and b ions containing 3-nitrotyrosine
    n = len(sequence)
    kb = max(k-1 for _, k, x in mods if x=='Nitro')
    ky = max(n-k for _, k, x in mods if x=='Nitro')
    bs = [(j,x) for j,x in enumerate(bs) if j>=kb]
    ys = [(j,x) for j,x in enumerate(ys) if j>=ky]
    # print kb, ky, bs, ys
    # b and y fragment ions with different charge states
    theoions, theonames = [], []
    for i in range(charge):
        theoions += [x/float(i+1)+1.0073 for _,x in bs]
        theonames += ['b%d#%d'%(j+1, i+1) for j, _ in bs]
        theoions += [x/float(i+1)+1.0073 for _,x in ys]
        theonames += ['y%d#%d'%(j+1, i+1) for j,_ in ys]
    #print zip(theoions, theonames)

    # annotate spectrum
    matchint, matcherr, minames = [], [], []
    for x in spectrum:
        xmz, xi = x[0], x[1]
        xk = [(abs(xmz-xj), theonames[i]) for i, xj in enumerate(theoions)
              if abs(xmz-xj)<=tol]
        if xk:
            xkj = [xj for xj, _ in xk]
            xmj = min(xkj)
            mj = xk[xkj.index(xmj)][1] if len(xkj)>1 else xk[0][1]
            if mj in minames:
                # .. remove replicate annotations, retain the annotation
                # .. with higher intensity
                k = minames.index(mj)
                if matchint[k]<xi:
                    matcherr[k] = xmj
                    matchint[k] = xi
            else:
                matcherr.append(xmj)
                matchint.append(xi)
                minames.append(mj)
    return matchint, matcherr


def methylionmatch(spectrum, sequence, mods, charge, tol, mtype='mono'):
    """
    Generate null scores for statistical inferences of normal scores
    """
    # generate b and y ions
    ys, bs = generateby(sequence, mods)
    # only consider y and b ions containing 3-nitrotyrosine
    n = len(sequence)
    kb = min(k-1 for _, k, x in mods if x=='Methyl' and sequence[k-1] in 'DE')
    ky = min(n-k for _, k, x in mods if x=='Methyl' and sequence[k-1] in 'DE')
    bs = [(j,x) for j,x in enumerate(bs) if j>=kb]
    ys = [(j,x) for j,x in enumerate(ys) if j>=ky]
    # print kb, ky, bs, ys
    # b and y fragment ions with different charge states
    theoions, theonames = [], []
    for i in range(charge):
        theoions += [x/float(i+1)+1.0073 for _,x in bs]
        theonames += ['b%d#%d'%(j+1, i+1) for j, _ in bs]
        theoions += [x/float(i+1)+1.0073 for _,x in ys]
        theonames += ['y%d#%d'%(j+1, i+1) for j,_ in ys]
    #print zip(theoions, theonames)

    # annotate spectrum
    matchint, matcherr, minames = [], [], []
    for x in spectrum:
        xmz, xi = x[0], x[1]
        xk = [(abs(xmz-xj), theonames[i]) for i, xj in enumerate(theoions)
              if abs(xmz-xj)<=tol]
        if xk:
            xkj = [xj for xj, _ in xk]
            xmj = min(xkj)
            mj = xk[xkj.index(xmj)][1] if len(xkj)>1 else xk[0][1]
            if mj in minames:
                # .. remove replicate annotations, retain the annotation
                # .. with higher intensity
                k = minames.index(mj)
                if matchint[k]<xi:
                    matcherr[k] = xmj
                    matchint[k] = xi
            else:
                matcherr.append(xmj)
                matchint.append(xi)
                minames.append(mj)
    return matchint, matcherr


def modXionmatch(spectrum, sequence, mods, objmod, charge, tol, mtype='mono'):
    """
    Generate null scores for statistical inferences of normal scores
    """
    # generate b and y ions
    ys, bs = generateby(sequence, mods)
    # only consider y and b ions containing 3-nitrotyrosine
    omx, oma = objmod[:-3], objmod[-2]
    n = len(sequence)
    kb = min(k-1 for _, k, x in mods if x==omx and sequence[k-1]==oma)
    ky = min(n-k for _, k, x in mods if x==omx and sequence[k-1]==oma)
    bs = [(j,x) for j,x in enumerate(bs) if j>=kb]
    ys = [(j,x) for j,x in enumerate(ys) if j>=ky]
    # print kb, ky, bs, ys
    # b and y fragment ions with different charge states
    theoions, theonames = [], []
    for i in range(charge):
        theoions += [x/float(i+1)+1.0073 for _,x in bs]
        theonames += ['b%d#%d'%(j+1, i+1) for j, _ in bs]
        theoions += [x/float(i+1)+1.0073 for _,x in ys]
        theonames += ['y%d#%d'%(j+1, i+1) for j,_ in ys]
    #print zip(theoions, theonames)

    # annotate spectrum
    matchint, matcherr, minames = [], [], []
    for x in spectrum:
        xmz, xi = x[0], x[1]
        xk = [(abs(xmz-xj), theonames[i]) for i, xj in enumerate(theoions)
              if abs(xmz-xj)<=tol]
        if xk:
            xkj = [xj for xj, _ in xk]
            xmj = min(xkj)
            mj = xk[xkj.index(xmj)][1] if len(xkj)>1 else xk[0][1]
            if mj in minames:
                # .. remove replicate annotations, retain the annotation
                # .. with higher intensity
                k = minames.index(mj)
                if matchint[k]<xi:
                    matcherr[k] = xmj
                    matchint[k] = xi
            else:
                matcherr.append(xmj)
                matchint.append(xi)
                minames.append(mj)
    return matchint, matcherr


def modTionmatch(spectrum, sequence, mods, modtype, charge, tol, mtype='mono'):
    """
    Generate null scores for statistical inferences of normal scores
    """
    # generate b and y ions
    ys, bs = generateby(sequence, mods)
    n = len(sequence)
    kb = min(k-1 for _, k, x in mods if x==modtype)
    ky = min(n-k for _, k, x in mods if x==modtype)
    bs = [(j,x) for j,x in enumerate(bs) if j>=kb]
    ys = [(j,x) for j,x in enumerate(ys) if j>=ky]
    # print kb, ky, bs, ys
    # b and y fragment ions with different charge states
    theoions, theonames = [], []
    for i in range(charge):
        theoions += [x/float(i+1)+1.0073 for _,x in bs]
        theonames += ['b%d#%d'%(j+1, i+1) for j, _ in bs]
        theoions += [x/float(i+1)+1.0073 for _,x in ys]
        theonames += ['y%d#%d'%(j+1, i+1) for j,_ in ys]
    #print zip(theoions, theonames)

    # annotate spectrum
    matchint, matcherr, minames = [], [], []
    for x in spectrum:
        xmz, xi = x[0], x[1]
        xk = [(abs(xmz-xj), theonames[i]) for i, xj in enumerate(theoions)
              if abs(xmz-xj)<=tol]
        if xk:
            xkj = [xj for xj, _ in xk]
            xmj = min(xkj)
            mj = xk[xkj.index(xmj)][1] if len(xkj)>1 else xk[0][1]
            if mj in minames:
                # .. remove replicate annotations, retain the annotation
                # .. with higher intensity
                k = minames.index(mj)
                if matchint[k]<xi:
                    matcherr[k] = xmj
                    matchint[k] = xi
            else:
                matcherr.append(xmj)
                matchint.append(xi)
                minames.append(mj)
    return matchint, matcherr


def removetagfrag(spectrum, tol = 0.1):
    """
    Remove itraq tag fragments.
    """
    itraq_tags = [113.1, 114.1, 115.1, 116.1, 117.1, 118.1, 119.1, 121.1]
    return [x for x in spectrum if not any(t+tol>=x[0]>=t-tol for t in itraq_tags)]


def getfulltheoions(theoions):
    """
    Get full set of theoions and remove precursors
    """
    theoions = [x for x in theoions if not x[1].startswith('p')]
    # consider the y_1[2+] and so on which are not considered in
    # generating theoretical ions
    if any('2+' in x for _, x in theoions):
        tionmz = [((x+1.0073)/2., '%s[2+]'%xn.split('[')[0])
                  for x, xn in theoions if int(xn.split('[')[0][1:])<3]
        theoions += tionmz
    if any('3+' in x for _, x in theoions):
        tionmz = [((x+1.0073*2)/3., '%s[3+]'%xn.split('[')[0])
                  for x, xn in theoions if int(xn.split('[')[0][1:])<5 and '[+]' in xn]
        theoions += tionmz
    return theoions


def detectisotopes(spectrum, tol = 0.1):
    """
    Detect isotopic clusters from mass spectrum.
    """
    # maximum charge for each fragment is set to be 3
    c = [1,2,3]
    # isotopic mass difference
    h = 1.0024
    n = len(spectrum)

    isocluster_b, ax = [], set()
    mzx = [x[0] for x in spectrum]
    # find potential isotopic clusters
    for i, x in enumerate(mzx):
        clj, clbk = [[x]]*len(c), []
        cy = list(c)
        kx = [j for j, xj in enumerate(mzx) if xj>x and xj<x+10]
        for ij in range(6):
            clk, cx, lx = [], [], []
            for k, txj in enumerate(clj):
                ct = txj[-1]
                kxj = [j for j in kx if abs(mzx[j]-ct-h/cy[k])<=tol/cy[k] and mzx[j]>ct]
                for j in kxj:
                    clk.append(txj+[mzx[j]])
                    cx.append(cy[k])
                    lx.append(k)
            clbk += [(x, cy[j]) for j,x in enumerate(clj) if j not in lx]
            cy, clj = list(cx), list(clk)
        isocluster_b += clbk
        if clk:
            isocluster_b += [(x, cx[j]) for j,x in enumerate(clk)]

    # remove invalid isotopic distributions or the distributions
    # belong to other distributions
    isocluster = [(x,c) for x,c in isocluster_b if len(x)>1]
    delx = []
    setx = [set(x) for x,_ in isocluster]
    for i, x in enumerate(setx):
        if any(not x-xj and xj-x for xj in setx):
            delx.append(i)
    isocluster = [x for i,x in enumerate(isocluster) if i not in delx]

    return isocluster


def binomialprob(p, n, k):
    """
    Calculate negative decadic logarithm of cumulative binomial probability
    """
    logn = [log10(i+1) for i in range(n)]
    nk = n-k
    s = sum(logn)
    
    pbt, pbf = log10(p), log10(1.-p)
    # the initial binomial
    pbk = []
    s1, s2 = sum(logn[:k]), sum(logn[:nk])
    pbk.append(s-s1-s2+k*pbt+nk*pbf)
    # calculate the cumulative using recursive iteration
    for i in range(k, n):
        s1 += logn[i]
        s2 -= logn[nk-1]
        nk -= 1
        pbk.append(s-s1-s2+(i+1)*pbt+nk*pbf)
    m = min(pbk)

    # to avoid OverflowError
    try:
        v = -m-log10(sum(10**(x-m) for x in pbk))
    except OverflowError:
        pbk2 = [x-m for x in pbk]
        m2 = max(pbk2)
        v = -m-m2-log10(sum(10**(x-m2) for x in pbk2))
    
    return v


def probs(seq, mod, c, spectrum, tol=0.2):
    """
    Probability based scoring schemes for validation of nitrated peptides
    For considering the scoring function using intensity, refer to the
    reference: Xu H, Freitas MA. A mass accuracy sensitive probability
    based scoring algorithm for database searching of tandem mass
    spectrometry data. BMC Bioinformatics. 2007, 8, 133.
    Refs:
    [1] Karagiannidis GK, Lioumpas AS. An improved approximation for
        the Gaussian Q-function. Communications Letters, IEEE. 2007,
        11(8), 644-646.
    """

    spectrum = removetagfrag(spectrum)
    
    mz, intensity = [x[0] for x in spectrum], [x[1] for x in spectrum]
    mnmz, mxmz = min(mz), max(mz)
    # number of bins to calculate probability
    nbins = round((max(mz)-min(mz))/tol)
    m = max(intensity)
    normints = [x/m for x in intensity]
    nk = len(mz)
    nq = len(seq)
    # probability for a random match
    p = nk/float(nbins)

    # define function for identifying whether the ion is b or y
    isby = isby

    # calculate a match using binomial distribution
    # .. annote fragment ions, currently only b and y type
    # .. ions are used.
    nd = nitrodeterm.NitroDetermine(seq, mod, c)
    theoions = nd.theoreticalions()
    theoions = getfulltheoions(theoions)
    theoionmz = [tmz[0] for tmz in theoions]
    np = sum(mxmz>=tmz>=mnmz for tmz in theoionmz)

    # .. annotate peaks
    ions = nd.annotatespectrum(spectrum, tol)
    mints, nm = [], 0
    for i, (mz, ion) in enumerate(ions):
        if ion:
            if isby(ion):
##                mints.append(normints[i])
                nm += 1
##            elif any(x.startswith('imm') or x.startswith('p') for x in ion.split(',')):
##                mints.append(normints[i])

    # .. cumulative probability
    prob_match = binomialprob(p, np, nm)

##    # .. cumulative binomial probability with sub-intervals
##    # .. .. reverse the sequence as a null hypothesis
##    rev_seq = seq[::-1]
##    rev_mod = []
##    for x in mod.split(';'):
##        if not x.endswith('term'):
##            rev_mod.append('%s@%d'%(x.split('@')[0], len(seq)-int(x.split('@')[1])+1))
##        else:
##            rev_mod.append(x)
##    rev_mod = ';'.join(rev_mod)
##    rev_nd = nitrodeterm.NitroDetermine(rev_seq, rev_mod, c)
##    rev_ions = rev_nd.annotatespectrum(spectrum, tol)
##    rev_theoions = nd.theoreticalions()
##    rev_theoions = getfulltheoions(rev_theoions)
##    rev_theoionmz = [tmz[0] for tmz in rev_theoions]
##    rev_np = sum(mxmz>=tmz>=mnmz for tmz in rev_theoionmz)
##
##    # max delta cumulative probability
##    subprob, nm, rev_nm = 0., 0, 0
##    mz0 = mnmz
##    while True:
##        mzj = min(mz0+100., mxmz)
##        if mz0<mxmz:
##            dprobj = []
##            npj = len([x for x in theoionmz if mz0<=x<=mzj])
##            rev_npj = len([x for x in rev_theoionmz if mz0<=x<=mzj])
##            
##            ix = [i for i, px in enumerate(spectrum) if mzj>=px[0]>=mz0]
##            jx = sorted(ix, key=lambda k: spectrum[k][1], reverse=True)
##            for j in xrange(1, min(len(jx), 8)+1):
##                nmj, rev_nmj = 0, 0
##                for i in jx[:j]:
##                    if ions[i][1] and isby(ions[i][1]):
##                        nmj += 1
##                    if rev_ions[i][1] and isby(rev_ions[i][1]):
##                        rev_nmj += 1
##                tp1 = binomialprob(p, npj, nmj) if nmj>0 else 0.
##                tp2 = binomialprob(p, rev_npj, rev_nmj) if rev_nmj>0 else 0.
##                dprobj.append((tp1-tp2, j, nmj, rev_nmj))
##            if dprobj:
##                kx = [x[0] for x in dprobj]
##                subprob += max(kx)
##                nm += dprobj[kx.index(max(kx))][2]
##                rev_nm += dprobj[kx.index(max(kx))][3]
##        else:
##            break
##        mz0 = mzj
##    
##    prob_match2 = binomialprob(p, np, nm)-binomialprob(p, rev_np, rev_nm)
##
##    # calculate descriptive peptide scoring model
##    S = 100.*sum(mints)*(sum(mints)/sum(normints))**2*max(0., (nm-3.)/float(nm))
##    S /= sqrt(nq)
##
##    # calculate intensity match probability
##    avg_int = sum(normints)/nk
##    std_int = 1./(nk-1)*sum((x-avg_int)**2 for x in normints)
##    avg_y = nk*p*avg_int
##    std_y = nk*p*(1.-p)*avg_int**2+nk*p*std_int
##    # .. probability that the match is random
##    # .. .. transform to standard normal distribution, then use approximations
##    # .. .. calculate the logarithm of probability
##    y = (sum(mints)-avg_y)/sqrt(std_y)
##    # .. .. if 0. <= y <= 5., use quintic polynomial fits to log2(phi)
##    # .. .. else use K&L approximation (see ref [1])
##    isnegy = y<0
##    y = abs(y)
##    if 0.<=y<=5:
##        c1, c2, c3, c4, c5, c6 = -0.0002658, 0.005223, -0.04586, -0.4681, -1.147, -1.001
##        prob_intensity = -(c1*y**5+c2*y**4+c3*y**3+c4*y**2+c5*y+c6)*log10(2.)
##    else:
##        prob_intensity = -log10(1.-exp(-1.4*y))+(y**2)/2.*log10(exp(1.))+log10(1.135*sqrt(2*pi)*y)
##    if isnegy:
##        prob_intensity = -log10(1.-10**(-prob_intensity))
##    #prob_intensity = -log10(0.5+erf((sum(mints)-avg_y)/sqrt(std_y*2))/2.)
##
##    # ratio between the highest intensity and 2nd highest intensity
##    sort_int = sorted(intensity, reverse=True)
##    mr = max(sort_int[0]/sort_int[1], sort_int[1]/sort_int[2])

    return prob_match, prob_match2, S, prob_intensity, mr


def matchNonNitro(seq, mod, charge, nitrosp, nnitrosp, tol = 0.1):
    """
    Calculate similarity between spectra generated by nitrated
    and non-nitrated peptides.
    """
    nd = nitrodeterm.NitroDetermine(seq, mod, charge)
    ions = nd.annotations(nitrosp, tol)
    m = max(x[1] for i, x in enumerate(nitrosp)
            if not any(xj.startswith('p') for xj in ions[i][1].split(',')))
    normint = [x[1]/m for x in nitrosp]

    # non-nitrated peptides
    modc = ';'.join(x for x in mod.split(';') if not 'Nitro' in x)
    nd_n = nitrodeterm.NitroDetermine(seq, modc, charge)
    ions_n = nd_n.annotations(nnitrosp, tol)
    m = max(x[1] for i, x in enumerate(nnitrosp)
            if not any(xj.startswith('p') for xj in ions_n[i][1].split(',')))
    normint_n = [x[1]/m for x in nnitrosp]

    #
    np1, np2 = len(nitrosp), len(nnitrosp)

    # get matched ions
    # .. only consider the sequential a, b and y ions, and neutral
    # .. losses of y and b ions
    m1, m2, mx, allseqion = [0.]*np1, [0.]*np2, [], set()
    for i, (mz1, ion1) in enumerate(ions):
        if not ion1: continue
        nx1 = [x.split('/')[0] for x in ion1.split(',')
              if not '-2' in x and x.count('-')!=2]
        nx1 = [x for x in nx1 if not (x.startswith('a') and '-' in x)]
        if not nx1: continue
        m1[i] = normint[i]
        allseqion.update([x for x in nx1 if not '-' in x and x[0] in 'yb'])
        # .. get matched ions from non-nitrated spectrum
        for j, (mz2, ion2) in enumerate(ions_n):
            if not ion2: continue
            nx2 = [x.split('/')[0] for x in ion2.split(',')
                   if not '-2' in x and x.count('-')!=2]
            nx2 = [x for x in nx2 if not (x.startswith('a') and '-' in x)]
            if not nx2: continue
            if m2[j]==0.:
                m2[j] = normint_n[j]
            if any(x in nx1 for x in nx2):
                mx.append((i, j))
            allseqion.update([x for x in nx2 if not '-' in x and x[0] in 'yb'])

    # calculate the fraction of sequential ions matched
    mx1, mx2 = [x[0] for x in mx], [x[1] for x in mx]
##    nm, mcx = 0, set()
##    for i, j in mx:
##        mcxj = set([x.split('/')[0] for x in ions[i][1].split(',') if '-' not in x])&\
##              set([x.split('/')[0] for x in ions_n[j][1].split(',') if '-' not in x])
##        if any(x.startswith('b') or x.startswith('y') for x in mcxj):
##            nm += 1
##        mcx.update(mcxj)
##    r = float(nm)/float(len(allseqion))

    #print sorted(allseqion), sorted(mcx)

    # get matched intensities
    m = [(m1[i], m2[j]) for i, j in mx]
##    print [(ions[i], ions_n[j]) for i, j in mx]
    m += [(m1[i], 0.) for i in range(np1) if i not in mx1 and m1[i]!=0.]
    m += [(0., m2[i]) for i in range(np2) if i not in mx2 and m2[i]!=0.]
    if not mx:
        return 0., m

    # extract p-IT8Plex fragment ion
    mit8 = [xi for xi, _, xj in nd.modifications() if xj=='iTRAQ8plex']
    if mit8:
        mit8 = mit8[0]
        p = nd.precursormass()
        pn = nd_n.precursormass()
        itx = [0., 0.]
        for i, x in enumerate(nitrosp):
            if i not in mx1 and abs(p-mit8+1.0073-x[0])<=tol:
                itx[0] = normint[i]
                mx1.append(i)
                break
        for i, x in enumerate(nnitrosp):
            if i not in mx1 and abs(pn-mit8+1.0073-x[0])<=tol:
                itx[1] = normint_n[i]
                mx2.append(i)
                break
        if any(x>0. for x in itx): m.append(tuple(itx))

##    # the top 10 peaks should be matched between the two spectra
##    mnitro = [xi for xi, _, xj in nd.modifications() if xj=='Nitro'][0]
##    six = sorted(xrange(np1), key=lambda k: normint[k], reverse=True)
##    for i in six[:min(np1, 10)]:
##        if i in mx1: continue
##        mqx = []
##        for j in xrange(np2):
##            if j in mx2: continue
##            if abs(nitrosp[i][0]-nnitrosp[j][0])<=tol or\
##               abs(abs(nitrosp[i][0]-nnitrosp[j][0])-mnitro)<=tol:
##                mqx.append(j)
##        if not mqx:
##            m.append((normint[i], 0.))
##            mx1.append(i)
##        else:
##            if len(mqx)==1:
##                j = mqx[0]
##                m.append((normint[i], normint_n[j]))
##            else:
##                jx = [nnitrosp[j][1] for j in mqx]
##                j = mqx[jx.index(max(jx))]
##                m.append((normint[i], normint_n[j]))
##            mx1.append(i)
##            mx2.append(j)
##
##    six = sorted(xrange(np2), key=lambda k: normint_n[k], reverse=True)
##    for i in six[:min(np2, 10)]:
##        if i in mx2: continue
##        mqx = []
##        for j in xrange(np1):
##            if j in mx1: continue
##            if abs(nitrosp[j][0]-nnitrosp[i][0])<=tol or\
##               abs(abs(nitrosp[j][0]-nnitrosp[i][0])-mnitro)<=tol:
##                mqx.append(j)
##        if not mqx:
##            m.append((0., normint_n[i]))
##            mx2.append(i)
##        else:
##            if len(mqx)==1:
##                j = mqx[0]
##                m.append((normint[j], normint_n[i]))
##            else:
##                jx = [nitrosp[j][1] for j in mqx]
##                j = mqx[jx.index(max(jx))]
##                m.append((normint[j], normint_n[i]))
##            mx1.append(i)
##            mx2.append(j)

    return pms.similarity([x[0] for x in m], [x[1] for x in m]), m


def matchNonMethyl(seq, mod, charge, nitrosp, nnitrosp, tol = 0.1):
    """
    Calculate similarity between spectra generated by nitrated
    and non-nitrated peptides.
    """
    nd = nitrodeterm.NitroDetermine(seq, mod, charge)
    ions = nd.annotations(nitrosp, tol)
    m = max(x[1] for i, x in enumerate(nitrosp)
            if not any(xj.startswith('p') for xj in ions[i][1].split(',')))
    normint = [x[1]/m for x in nitrosp]

    # non-nitrated peptides
    modc = ';'.join(x for x in mod.split(';')
                    if not 'Methyl(D)' in x and not 'Methyl(E)' in x)
    nd_n = nitrodeterm.NitroDetermine(seq, modc, charge)
    ions_n = nd_n.annotations(nnitrosp, tol)
    m = max(x[1] for i, x in enumerate(nnitrosp)
            if not any(xj.startswith('p') for xj in ions_n[i][1].split(',')))
    normint_n = [x[1]/m for x in nnitrosp]

    #
    np1, np2 = len(nitrosp), len(nnitrosp)

    # get matched ions
    # .. only consider the sequential a, b and y ions, and neutral
    # .. losses of y and b ions
    m1, m2, mx, allseqion = [0.]*np1, [0.]*np2, [], set()
    for i, (mz1, ion1) in enumerate(ions):
        if not ion1: continue
        nx1 = [x.split('/')[0] for x in ion1.split(',')
              if not '-2' in x and x.count('-')!=2]
        nx1 = [x for x in nx1 if not (x.startswith('a') and '-' in x)]
        if not nx1: continue
        m1[i] = normint[i]
        allseqion.update([x for x in nx1 if not '-' in x and x[0] in 'yb'])
        # .. get matched ions from non-nitrated spectrum
        for j, (mz2, ion2) in enumerate(ions_n):
            if not ion2: continue
            nx2 = [x.split('/')[0] for x in ion2.split(',')
                   if not '-2' in x and x.count('-')!=2]
            nx2 = [x for x in nx2 if not (x.startswith('a') and '-' in x)]
            if not nx2: continue
            if m2[j]==0.:
                m2[j] = normint_n[j]
            if any(x in nx1 for x in nx2):
                mx.append((i, j))
            allseqion.update([x for x in nx2 if not '-' in x and x[0] in 'yb'])

    # calculate the fraction of sequential ions matched
    mx1, mx2 = [x[0] for x in mx], [x[1] for x in mx]

    # get matched intensities
    m = [(m1[i], m2[j]) for i, j in mx]
    m += [(m1[i], 0.) for i in range(np1) if i not in mx1 and m1[i]!=0.]
    m += [(0., m2[i]) for i in range(np2) if i not in mx2 and m2[i]!=0.]
    if not mx:
        return 0., m

    # extract p-IT8Plex fragment ion
    mit8 = [xi for xi, _, xj in nd.modifications() if xj=='iTRAQ8plex']
    if mit8:
        mit8 = mit8[0]
        p = nd.precursormass()
        pn = nd_n.precursormass()
        itx = [0., 0.]
        for i, x in enumerate(nitrosp):
            if i not in mx1 and abs(p-mit8+1.0073-x[0])<=tol:
                itx[0] = normint[i]
                mx1.append(i)
                break
        for i, x in enumerate(nnitrosp):
            if i not in mx1 and abs(pn-mit8+1.0073-x[0])<=tol:
                itx[1] = normint_n[i]
                mx2.append(i)
                break
        if any(x>0. for x in itx): m.append(tuple(itx))

    return pms.similarity([x[0] for x in m], [x[1] for x in m]), m


def matchNonNitro2(ions, nitrosp, ions_n, nnitrosp):
    """
    Calculate similarity between spectra generated by nitrated
    and non-nitrated peptides.
    """
    rx = [x[1] for x, (xk,_) in zip(nitrosp, ions)
          if (xk and not any(xj.startswith('p') for xj in xk)) or not xk]
    m = max(rx) if rx else max(x[1] for x in nitrosp)
    normint = [x[1]/m for x in nitrosp]

    # non-nitrated peptides
    rx = [x[1] for x, (xk,_) in zip(nnitrosp, ions_n)
          if (xk and not any(xj.startswith('p') for xj in xk)) or not xk]
    m = max(rx) if rx else max(x[1] for x in nitrosp)
    normint_n = [x[1]/m for x in nnitrosp]

    #
    np1, np2 = len(nitrosp), len(nnitrosp)
    kxn = [i for i,(_,x) in enumerate(ions_n) if x]
    nxs = [ions_n[i][0] for i in kxn]

    # get matched ions
    # .. only consider the sequential a, b and y ions, and neutral
    # .. losses of y and b ions
    m1, m2, mx = [0.]*np1, [0.]*np2, []
    for i, (nx1, ion1) in enumerate(ions):
        if not ion1: continue
        m1[i] = normint[i]
        # .. get matched ions from non-nitrated spectrum
        for ji, j in enumerate(kxn):
            if m2[j]==0.:
                m2[j] = normint_n[j]
            if set(nx1)&set(nxs[ji]):
                mx.append((i, j))

    # calculate the fraction of sequential ions matched
    mx1, mx2 = [x[0] for x in mx], [x[1] for x in mx]

    # get matched intensities
    m = [(m1[i], m2[j]) for i, j in mx]
    m += [(m1[i], 0.) for i in range(np1) if i not in mx1 and m1[i]!=0.]
    m += [(0., m2[i]) for i in range(np2) if i not in mx2 and m2[i]!=0.]

    # get matched indices
    matchix = list(mx)
    matchix += [(i, -1) for i in range(np1) if i not in mx1 and m1[i]>0.]
    matchix += [(-1, i) for i in range(np2) if i not in mx2 and m2[i]>0.]
    
    if not mx:
        return 0., m, matchix
    

    return pms.similarity([x[0] for x in m], [x[1] for x in m]), m, matchix


def matchNonMod(sp, ions, nsp, ions_n):
    """
    Calculate similarity between spectra generated by nitrated
    and non-nitrated peptides.
    """
    # get matched ion indices
    def getmx(ions, ions_n, cions):
        cx1, cx2 = [], []
        for xk in cions:
            cx1.append(ions[xk][0])
            cx2.append(ions_n[xk][0])
        return cx1, cx2

    # remove replicate indices
    def removerep(ix1, ix2, sp, nsp):
        mx = []
        if len(set(ix1))<len(ix1):
            for i in set(ix1):
                if ix1.count(i)==1:
                    j = ix1.index(i)
                    mx.append((i, ix2[j]))
                else:
                    jx = [ix2[j] for j,ji in enumerate(ix1) if ji==i]
                    ix = [nsp[j][1] for j in jx]
                    mx.append((i, jx[ix.index(max(ix))]))
        else:
            mx = list(zip(ix1, ix2))
        return mx
        
    # get the annotated names of modified peptides
    byx0 = set(xk for xk in list(ions.keys()) if xk[0] in 'ybp' and not '-' in xk)
    neutrolx0 = set(xk for xk in list(ions.keys()) if xk not in byx0)
    # get the annotated names of non-modified analogues
    byx1 = set(xk for xk in list(ions_n.keys()) if xk[0] in 'ybp' and not '-' in xk)
    neutrolx1 = set(xk for xk in list(ions_n.keys()) if xk not in byx1)

    # get matched fragments
    mby1, mby2 = getmx(ions, ions_n, byx0&byx1)
    mnl1, mnl2 = getmx(ions, ions_n, neutrolx0&neutrolx1)

    # remove replicate indices
    mby = removerep(mby1, mby2, sp, nsp)
    ax1, ax2 = set(i for i,_ in mby), set(i for _,i in mby)
    # remove matched assigned fragments as y, b and p ions
    mnl1x, mnl2x = [], []
    for i,j in zip(mnl1, mnl2):
        if i in ax1 or j in ax2: continue
        mnl1x.append(i)
        mnl2x.append(j)
    mnl = removerep(mnl1x, mnl2x, sp, nsp)
    ax1.update(i for i,_ in mnl)
    ax2.update(i for _,i in mnl)

    # get common indices
    matchixs = mby+mnl
    # if annotated fragments not matched, left as unmatched fragments
    matchixs += [(i, None) for i, _ in list(ions.values()) if i not in ax1]
    matchixs += [(None, i) for i, _ in list(ions_n.values()) if i not in ax2]
    ax1.update(i for i,_ in list(ions.values()))
    ax2.update(i for i,_ in list(ions_n.values()))

    # get matched un-annotated ions
    for i, xk in enumerate(sp):
        if i in ax1: continue
        jx = [j for j, xj in enumerate(nsp) if j not in ax2 and abs(xj[0]-xk[0])<=0.2]
        if jx:
            if len(jx)==1:
                j = jx[0]
            else:
                ix = [nsp[j][1] for j in jx]
                j = jx[ix.index(max(ix))]
            matchixs.append((i, j))
            ax2.add(j)
        else:
            matchixs.append((i, None))
        ax1.add(i)
    matchixs += [(None, i) for i in range(len(nsp)) if i not in ax2]

    return matchixs


def nitroRS(seq, mod, c, spectrum, tol=0.1):
    """
    Reimplementation of PhosphoRS for validation of nitration.
    """
    scores = []
    l = len(seq)
    mz, intensity = [x[0] for x in spectrum], [x[1] for x in spectrum]

    # probability
    p = len(mz)*tol/(max(mz)-min(mz))

    # get possible nitration sites
    yix = [i for i, x in enumerate(seq) if x=='Y']
    # all site-determining b ions and their complementaries
    bs = ['b%d'%(i+1) for i in range(l-1)]
    bsc = ['y%d'%(l-i-1) for i in range(l-1)]
    # all site-determining y ions and their complementaries
    ys = ['y%d'%(l-i) for i in range(1,l)]
    ysc = ['b%d'%i for i in range(1,l)]
    # number of nitrations
    smod = mod.split(';')
    nnitro = sum('Nitro' in x for x in smod)
    nitroname = [x.split('@')[0] for x in smod if 'Nitro' in x][0]
    modother = [x for x in smod if 'Nitro' not in x]

    #
    for sites in combinations(yix, nnitro):
        # re-organize the nitration sites
        modj = list(modother)
        for i in sites:
            modj.append('%s@%d'%(nitroname, i))
        nd = nitrodeterm.NitroDetermine(seq, ';'.join(modj), c)
        ions = nd.annotatespectrum(spectrum, tol)
        # .. get matched theoretical fragment ions
        match_theo = []
        for i, (mzj, ionj) in enumerate(ions):
            if ionj:
                nx = [x.split('/')[0] for x in ionj.split(',')
                      if x[0] in 'yb' and not '-' in x]
                for x in nx:
                    xj = x.split('[')
                    if xj[1][0]=='+':
                        match_theo.append(xj[0])
        # .. get probability
        sdions = set(['b%d'%i for i in range(min(sites)+1, l)])
        if sdions:
            sdions.update([bsc[bs.index(x)] for x in sdions])
        sdions2 = ['y%d'%i for i in range(l-max(sites), l)]
        if sdions2:
            sdions.update(sdions2)
            sdions.update([ysc[ys.index(x)] for x in sdions2])
        n = len(sdions)
        k = len(set(match_theo)&sdions)
        scores.append(binomialprob(p, n, k))

    return scores


def ascore(seq, mod, c, spectrum, tol=0.1):
    """
    Ascore of nitration
    """
    nd = nitrodeterm.NitroDetermine(seq, mod, c)
    ions = nd.annotatespectrum(spectrum, tol)
    l = len(seq)
    mz, intensity = [x[0] for x in spectrum], [x[1] for x in spectrum]

    # probability
    p = len(mz)*tol/(max(mz)-min(mz))

    # get nitration sites
    nitrosite = [int(x.split('@')[1])-1 for x in mod.split(';') if 'Nitro' in x]
    # .. all site-determining b ions and their complementaries
    bs = ['b%d'%(i+1) for i in range(l-1)]
    bsc = ['y%d'%(l-i-1) for i in range(l-1)]
    # .. all site-determining y ions and their complementaries
    ys = ['y%d'%(l-i) for i in range(1,l)]
    ysc = ['b%d'%i for i in range(1,l)]

    f = log(10.)*10.

    # .. get matched theoretical fragment ions
    match_theo = []
    for i, (mzj, ionj) in enumerate(ions):
        if ionj:
            nx = [x.split('/')[0] for x in ionj.split(',')
                  if x[0] in 'yb' and not '-' in x]
            for x in nx:
                xj = x.split('[')
                if xj[1][0]=='+':
                    match_theo.append(xj[0])
    # .. get probability
    if seq.count('Y')==1:
        j = nitrosite[0]
        sdions = set(['b%d'%i for i in range(j+1, l)])
        if sdions:
            sdions.update([bsc[bs.index(x)] for x in sdions])
        sdions2 = ['y%d'%i for i in range(l-j, l)]
        if sdions2:
            sdions.update(sdions2)
            sdions.update([ysc[ys.index(x)] for x in sdions2])
        n = len(sdions)
        k = len(set(match_theo)&sdions)
        return f*binomialprob(p, n, k)
    else:
        yix = [i for i, x in enumerate(seq) if x=='Y']
        ascores = []
        for kk in nitrosite:
            yi_b = [i-kk for i in yix if i>kk and i not in nitrosite]
            yi_y = [kk-i for i in yix if i<kk and i not in nitrosite]
            yk_y = min(yi_b) if yi_b else l
            yk_b = min(yi_y) if yi_y else l
            if yk_b<=yk_y:
                kj = max([i for i in yix if i<kk and i not in nitrosite])
                n = yk_b*2
                sdions = ['y%d'%(l-i-1) for i in range(kj, kk)]
                sdionsc = [ysc[ys.index(x)] for x in sdions]
            else:
                kj = max([i for i in yix if i>kk and i not in nitrosite])
                n = yk_y*2
                sdions = ['b%d'%(i+1) for i in range(kk, kj)]
                sdionsc = [bsc[bs.index(x)] for x in sdions]
            k = len(set(match_theo)&set(sdions+sdionsc))
            ascores.append(binomialprob(p, n, k))
        return [f*x for x in ascores]
    

def readPeptideSummary(peptidesummaryfile):
    """
    read results from ProteinPilot Peptide Summary
    """
    PPres = []
    resappend = PPres.append
    with open(peptidesummaryfile,'r') as f:
        # read ProteinPilot peptide summaries
        csvreader = csv.DictReader(f, delimiter='\t')
        for r in csvreader:
            resappend((r['Sequence'],
                       r['Modifications'],
                       int(r['Theor z']),
                       r['Spectrum'],
                       r['Time'],
                       float(r['Conf']),
                       float(r['Theor m/z']),
                       float(r['Prec m/z']),
                      r['Accessions'],
                      r['Names']))

    return PPres


def readmgf(mgffile, spid):
    """
    Read tandem mass spectra from mgf file according to the input
    spid
    """
    sp = []
    #newid = '.'.join(['1']+spid.split('.')[1:])
    read_tag = False
    with open(mgffile, 'r') as f:
        for line in f:
            if spid in line:
                read_tag = True
                tsp = []
            elif line.startswith('END IONS'):
                if read_tag:
                    sp.append(tsp)
                read_tag = False
            elif read_tag:
                if '=' in line: continue
                try:
                    tsp.append([float(x) for x in line.rstrip().split()[:2]])
                except:
                    pass
    if len(sp)==1:
        sp = sp[0]
    return sp


def extractSpectrum(setname, fileorder, PPid):
    """
    extract spectrum according to ProteinPilot ID
    """
    idinfo = PPid.split('.')
    dname = fileorder.get(idinfo[0])
    cycle = idinfo[3]
    experiment = idinfo[4]
    dtaname = '.'.join([dname, cycle, experiment, '1.dta'])
    spectrum = []
    dtafile = r'./%s/dtafiles_processed/%s'%(setname, dtaname)
    if not os.path.isfile(dtafile):
        return spectrum
    
    with open(dtafile,'r') as f:
        for line in f:
            spectrum.append(tuple([float(x) for x in line.rstrip().split('\t')]))
    return spectrum[1:]


def dtareader(dtafile):
    """
    read spectrum from dta file
    """
    sp = []
    with open(dtafile,'r') as f:
        for line in f:
            sp.append([float(x) for x in line.rstrip().split()])
    return sp[1:]


def ppid2dtaname(setname, ppid):
    """
    Convert ProteinPilot spectrum ID to dta file name
    """
    ix = RAWSETS.index(setname)
    sppid = ppid.split('.')
    filenum = FILENUM[ix]
    datafilename = filenum[sppid[0]]
    return '.'.join([datafilename, sppid[3], sppid[4], '1.dta'])


def dtaname2ppid(setname, dtaname):
    """
    Convert dta file name to ProteinPilot spectrum ID
    """
    sdx = dtaname.split('.')
    ix = RAWSETS.index(setname)
    for filenum, ppfilename in list(FILENUM[ix].items()):
        if ppfilename==sdx[0]:
            break
    return '.'.join([filenum, '1.1']+sdx[1:3])


def combinemod(seq, modification):
    """
    Integrate modifications to corresponding locations of peptide
    sequence.
    """
    if not modification:
        return seq

    # preparse modifications
    modification = modification.replace(' ','')
    mod = ';'.join(x for x in modification.split(';') if not x.startswith('No'))
    if not mod: return seq
    mod = ';'.join(x.split('ProteinTerminal')[1] if x.startswith('ProteinTerminal') else x
                   for x in mod.split(';'))
    if not mod: return seq
    
    modx = [tuple(x.split('@')) for x in mod.split(';')]
    pl, l = list(seq), len(seq)
    smod = []
    for modname, modsite in modx:
        if modsite.lower()=='n-term':
            smod.append(('[%s]'%modname,0))
        elif modsite.lower()=='c-term':
            smod.append(('[%s]'%modname,l))
        else:
            smod.append(('[%s]'%modname,int(modsite)))

    if not smod:
        return seq
    
    smod = sorted(smod, key=lambda x: x[1], reverse=True)
    for modname, i in smod:
        if i==l:
            pl.append(modname)
        else:
            pl.insert(i, modname)

    return ''.join(pl)


def combmod(seq, mod):
    """
    Combine modification to sequence
    """
    if not mod: return seq
    Nmod = False
    modsites = []
    for xk, site, _ in mod:
        if isinstance(site, str):
            Nmod = True
            Nstr = str(round(xk))
        else:
            modsites.append((site, str(round(xk))))
    
    if not modsites: return '[%s]%s'%(Nstr, seq)

    modsites = sorted(modsites, key=itemgetter(0))
    modseqstr = ['[%s]'%Nstr] if Nmod else []
    j = 0
    for site, sk in modsites:
        if site==len(seq):
            modseqstr.append('%s[%s]'%(seq[j:site], sk))
            break
        modseqstr.append('%s[%s]'%(seq[j:site], sk))
        j = site
    if site!=len(seq):
        modseqstr.append(seq[site:])
    return ''.join(modseqstr)


def processalldtafiles():
    """
    preprocess all dta files and stored in new directory
    """
    pathjoin = os.path.join
    centroidms, denoise = pms.centroidms, pms.denoise
    for rawset in RAWSETS:
        cdir = r'./%s/dtafiles'%rawset
        targetdir = r'./%s/dtafiles_processed'%rawset
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        fs = [f for f in os.listdir(cdir) if f.endswith('dta')]
        for f in fs:
            sp = []
            with open(pathjoin(cdir, f),'r') as dtaf:
                for line in dtaf:
                    sp.append([float(x) for x in line.rstrip().split()])
            spx = centroidms(sp[1:])
            spx = denoise(spx, 'median')

            if not spx or len(spx)<10:
                continue

            # set up lines
            ndtaheader = '%.4f\t%.4f\n'%(sp[0][0], sp[0][1])
            xlines = ['%.4f\t%.4f\n'%(x[0], x[1]) for x in spx]
            xlines.insert(0, ndtaheader)

            with open(pathjoin(targetdir, f),'w') as fw:
                fw.writelines(xlines)
        print(('Set %s is processed...'%rawset))


def getuniquematches(res):
    """
    Get unique match results
    """
    ures, useq = [], []
    for _, spid, seq, mod, c, nmod, df, pmz, ns, simscore, m3 in res:
        seq1 = combinemod(seq, nmod)
        if ures:
            t = False
            for i, (tseq, tc, tnmod, tdf, _, _, _, _) in enumerate(ures):
                if seq1==useq[i][0] and tc==c and tdf==df:
                    t = True
                    break
            if not t:
                ures.append((seq, c, nmod, df, pmz, ns, simscore, m3))
                useq.append((seq1, c))
        else:
            ures.append((seq, c, nmod, df, pmz, ns, simscore, m3))
            useq.append((seq1, c))
            
    return ures, useq


def getnewmatches(res, ppres):
    """
    Get new matches that are not identified by PP
    """
    ppnitros = [x for x in ppres if 'Nitro' in x[1]]
    seqs, nix = [], []
    for i, (_, _, seq, _, c, mod, _, _, _, _, _) in enumerate(res):
        cseq = combinemod(seq, mod)
        t = False
        for ppseq, ppmod, ppc, _, _, _ in ppnitros:
            seqx = combinemod(ppseq, ppmod)
            if cseq==seqx and c==ppc:
                t = True
                break
        if not t:
            seqs.append('%s#%d'%(seq, c))
            nix.append(i)
    return seqs, nix


def getmatchresults(matchfile):
    """
    Get matched results.
    """
    mlist, tlist = [], []
    with open(matchfile, 'r') as f:
        for line in f:
            if line.startswith('DATASET'):
                if tlist:
                    mlist += tlist
                tlist = []
                sline = line.rstrip().split(',')
                spid = [x.split('=')[1] for x in sline if 'MASSPECTRUMID' in x][0]
                seq = [x.split('=')[1] for x in sline if 'SEQUENCE' in x][0]
                mod = [x.split('=')[1] for x in sline if 'MODIFICATIONS' in x][0]
                nmod = [x.split('=')[1] for x in sline if 'REVISED_MODIFICATIONS' in x]
                nmod = nmod[0]
                c = int([x.split('=')[1] for x in sline if 'CHARGE' in x][0])
                rt = float([x.split('=')[1] for x in sline if 'RETENTIONTIME' in x][0])
                rset = [x.split('=')[1] for x in sline if 'DATASET' in x][0]
            else:
                if not line.startswith('Data'):
                    sline = line.rstrip().split('\t')
                    pmz = float(sline[2])
                    try:
                        simscore = float(sline[4])
                    except:
                        print(sline)
                        return
                    df = sline[1]
                    if ',' in sline[3]:
                        sc = [float(x) for x in sline[3].split(',')]
                    else:
                        sc = [float(sline[3])]
                    bt = True if sline[5]=='True' else False
                    tlist.append((rset, spid, seq, mod, c,
                                  nmod, df, pmz, sc, simscore, bt))
    return mlist
    

def getionnames(spectrum, sequence, modifications, charge, ions,
                calions=True, mtype='mono'):
    """
    get names of fragment ions in spectrum according to the specified
    peptide sequence "sequence", modifications and charge state
    """
    if calions:
        seqmass = annotesp.calpepmass(sequence, modifications, mtype)
        bs, ys, pmass = annotesp.theoions(seqmass[1],
                                          ntermmass=seqmass[0],
                                          ctermmass=seqmass[2])
        ions = annotesp.generateions(bs, ys, pmass, charge)
    
    ionnames, ionerr = annotesp.annotespectrum(spectrum, ions)
    ionnames, ionerr = annotesp.neutralassign(spectrum, ionnames, ionerr)
    ionnames = annotesp.removerepions(ionnames, ionerr)
    # only ion name with smallest error is retained if multiple
    # ion names are assigned to a fragment ion
    # before this annotation, intact sequential ions are included
    for i, name in enumerate(ionnames):
        if ',' in name:
            namex = [x.split('/')[0] for x in name.split(',')]
            errx = [float(x.split('/')[1]) for x in name.split(',')]
            kx = [k for k,nx in enumerate(namex) if nx[0] in 'ybp' and '-' not in nx]
            if kx:
                namex, errx = [namex[k] for k in kx], [errx[k] for k in kx]
            ionnames[i] = namex[errx.index(min(errx))]
        elif name:
            ionnames[i] = name.split('/')[0]
            
    return ionnames


def getshiftmz(ionname, modaddix, modmass, orignalmz):
    """
    get m/z of fragment ion of which sequence are added by modifications
    specified by indices in "modaddix"
    """
    mh = 1.0073
    ix = ionname.index('+')
    # charge of the ion
    try:
        c = int(ionname[ix-1])
    except:
        c = 1
    mion = orignalmz*c - c*mh
    # cleavage site of the fragment ion
    if ionname.startswith('p'):
        mion += sum(modmass)
    else:
        p = ionname.split('[')[0].split('-')[0]
        site = int(p[1:])
        # add the modification mass to mass of the ion
        for i, modsite in enumerate(modaddix):
            if site >= modsite:
                mion += modmass[i]

    return (mion+c*mh)/c


def getpepsumdir(setname):
    """
    get peptide summary file directory
    """
    setdir = r'./%s'%setname
    fx = [f for f in os.listdir(setdir) if 'ProteinPilot' in f][0]
    setdir = os.path.join(setdir, fx)
    fx = [f for f in os.listdir(setdir) if 'PeptideSummary' in f][0]
    return os.path.join(setdir, fx)


def getprocessedlist(setname):
    """ Get the list of spectra that have been processed """
    processedlist = []
    with open('match_raw_analogue_%s.txt'%setname,'r') as f:
        for line in f:
            if line.startswith('DATASET'):
                iditem = [s for s in line.split(',')
                          if s.lstrip().startswith('MASSPECTRUMID')][0]
                seqitem = [s for s in line.split(',')
                          if s.lstrip().startswith('SEQUENCE')][0]
                processedlist.append((iditem.split('=')[1], seqitem.split('=')[1]))
    return processedlist


def groupcandidates(candidatesets, setname):
    """
    Group tyrosine-containing peptides to faster the searching
    of nitrated analogues.
    """
    centroidms, denoise, calsim = pms.centroidms, pms.denoise, pms.calsimilarity
    filenum = FILENUM[RAWSETS.index(setname)]
    peps = []
    for seq, mods, charge, _, _, _ in candidatesets:
        peps.append('%s#%s#%d'%(seq, mods.replace(' ',''), charge))
    # get unique peptides and its accurrence
    C = Counter(peps)
    D = []
    #with open('group_peps.txt','w') as f:
    for key in list(C.keys()):
        ix = [i for i, p in enumerate(peps) if p==key]
        if len(ix)==1:
            D.append(candidatesets[ix[0]])
            continue
        # cluster peptides
        ux, uxk, pairx = [], [ix[0]], []
        ux.append(candidatesets[uxk[0]])
        while True:
            n0 = len(uxk)   # check number of unqiues in previous iteration
            for i, (_, _, _, spid, _, _) in enumerate(ux):
                rsp = extractSpectrum(setname, filenum, spid)
                #rsp = centroidms(rsp)
                #rsp = denoise(rsp, 'median')
                ixj = [j for j in ix if j not in uxk and spid not in candidatesets[j]]
                for j in ixj:
                    # ignore those matched in previous iterations
                    if (uxk[i], j) in pairx: continue
                    # get candidate spectrum and calculate similarity
                    _, _, _, spidc, _, _ = candidatesets[j]
                    csp = extractSpectrum(setname, filenum, spidc)
                    #csp = centroidms(csp)
                    #csp = denoise(csp, 'median')
                    s = calsim(rsp,csp)
                    if s<0.94:
                        ux.append(candidatesets[j])
                        uxk.append(j)
                    pairx.append((uxk[i],j))
                    #f.write('%s vs %s:\t%.4f\n'%(spid, spidc, s))
            if len(uxk)==n0:
                break
        D += ux

    return D


def checktruepositives():
    """
    Revalidation of ProteinPilot identified nitrated peptides
    """
    centroidms, denoise = pms.centroidms, pms.denoise
    calsim, matchtops = pms.calsimilarity, pms.matchtops
    NitroDetermine = nitrodeterm.NitroDetermine
    tol, mnitro = 3., 44.985078

    # get manually validated results
    with open('tpvalidation.txt','w') as f:
        f.write('Data Set\tSpectrum ID\tSequence\tModifications\t' +
                'Charge\tNitroScore\tSimilarity Score\tRight Order\t' +
                'Characteristic Peaks\tTop 3 Match\tManually Validated\n')
        for i, tpfile in enumerate(MANUALVALIDATED):
            ks = RAWSETS[i]
            dtadir = r'./%s/dtafiles_processed/'%ks
            # read ProteinPilot results
            PPres = readPeptideSummary(getpepsumdir(ks))
            nitropps = [x for x in PPres if 'Nitro(Y)' in x[1]]
            thr_conf, filenum = CONF[i], FILENUM[i]
            PPres = [x for x in PPres if x[5]>=thr_conf]
            PPres = [x for x in PPres if 'Nitro(Y)' not in x[1]]
            PPpeps, PPexps = [], []
            for ppseq, ppmod, _, ppid, _, _, _ in PPres:
                PPpeps.append(combinemod(ppseq, ppmod))
                sid = ppid.split('.')
                PPexps.append((int(sid[3]), int(sid[4])))
##
##            # read positive sequences
##            tps, kk = [], 0
##            with open(tpfile, 'r') as fx:
##                for line in fx:
##                    if 'Sequence' in line: continue
##                    linesplit = line.rstrip().split('\t')
##                    if linesplit[0]:
##                        seq = linesplit[0].replace(' ','')
##                        spectrumid = linesplit[1].replace(' ','')
##                        modification = linesplit[2].replace(' ','')
##                        t = 'Manual' in line
##                        cseq1 = combinemod(seq, modification)
##                        tx = False
##                        for j, (tseq, tmods, tc, tspid, _, _) in enumerate(nitropps):
##                            if tspid==spectrumid:
##                                cseq2 = combinemod(tseq, tmods)
##                                if cseq2==cseq1:
##                                    c, tx = tc, True
##                                    kk += 1
##                                    break
##                        if tx:
##                            tps.append((ks, spectrumid, seq, modification, c, t))
##                        else:
##                            tps.append((ks, spectrumid, seq, modification, None, t))

            # validate tps
            pids = []
            for seq, mod, c, spid, _, _, _ in nitropps:
                pids.append('%s#%s#%d'%(spid, combinemod(seq, mod), c))
            uix = [pids.index(x) for x in set(pids)]
            print((len(pids), len(uix)))
            
            for j, (seq, mod, c, spid, _, _, _) in enumerate(nitropps):
                if j not in uix: continue
##            for j, (rawset, spid, seq, mod, c, manualchecked) in enumerate(tps):
##                if not c:
##                    f.write('%s\t%s\t%s\t%s\tNone\tNone\tNone\tNone\tNone\tNone\t%s\n'%(ks,
##                                                                                        spid,
##                                                                                        seq,
##                                                                                        mod,
##                                                                                        str(manualchecked)))
##                    continue

                # setup directory and file for copying
                fs = [fj for fj in os.listdir(r'./tpspectra') if seq in fj]
                jk = 0 if not fs else len(fs)
                tf = r'./tpspectra/%s_%d'%(seq, jk)
                if not os.path.isdir(tf):
                    os.makedirs(tf)
                tdf = ppid2dtaname(ks, spid)
                copyfile(dtadir+tdf, '%s/%s'%(tf, tdf))

                # setup constants
                l = len(seq)
                smod = mod.split(';')
                ym = [int(x.split('@')[1]) for x in smod if 'Nitro' in x]
                if any(x.lower().startswith('no') for x in smod):
                    smod = [x for x in smod if not x.lower().startswith('no')]
                    mod = ';'.join(smod)
                nmod = ';'.join(x for x in smod if 'Nitro' not in x)
                cseq = combinemod(seq, nmod)
                nn = len(ym)
                
                nd1 = NitroDetermine(seq, mod, c)
                sp1 = extractSpectrum(ks, filenum, spid)
                nscore = nd1.score(sp1)
                names1 = nd1.annotatespectrum(sp1)
                m1 = max(x[1] for x in sp1)
                spk = [[sp1[k][0], sp1[k][1]/m1*100.]
                       for k,nx in enumerate(names1) if nx[1]]
                intk = [x[1] for x in spk]
                intk.sort(reverse=True)
                tsq = intk[0]/intk[1]>=0.5
                
                # criteria 4. nitrated peptide has an elution time that is
                # later than its non-modified counterpart
                sdf = spid.split('.')
                cyc, exp = int(sdf[3]), int(sdf[4])
                t4, t5, tsims, pmz = False, False, [], []
                for k, (ppcyc, ppexp) in enumerate(PPexps):
                    if PPpeps[k]!=cseq or PPres[k][2]!=c or cyc<ppcyc or (cyc==ppcyc and exp<ppexp):
                        continue
                    t4 = True
                    # criteria 5. characteristic peaks (45 Da shift corresponding
                    # to 3-nitro group) must be observed in the MS/MS spectrum
                    nd2 = NitroDetermine(seq, nmod, c)
                    sp2 = extractSpectrum(ks, filenum, PPres[k][3])
                    names2 = nd2.annotatespectrum(sp2)
                    mx = []
                    for mz1, ni1 in names1:
                        if not ni1: continue
                        nix1 = [ni.split('/')[0] for ni in ni1.split(',') if '-' not in ni]
                        if not nix1: continue
                        for mz2, ni2 in names2:
                            nix2 = [ni.split('/')[0] for ni in ni2.split(',') if '-' not in ni]
                            if not nix2: continue
                            nim = [ni for ni in nix1 if ni in nix2]
                            if nim:
                                mx.append(nim)

                    if len(mx)<5: continue
                                
                    bx = ['b%d'%ik for ik in range(l) if any(ik+1>=jk for jk in ym)]
                    yx = ['y%d'%ik for ik in range(l) if any(ik>=l-jk for jk in ym)]
                    ax = ['a%d'%ik for ik in range(l) if any(ik+1>=jk for jk in ym)]
                    bmx, ymx, amx = [], [], []
                    for mions in mx:
                        for jk, bi in enumerate(bx):
                            if any(bi in mii for mii in mions):
                                bmx.append(jk)
                        for jk, yi in enumerate(yx):
                            if any(yi in mii for mii in mions):
                                ymx.append(jk)
                        for jk, ai in enumerate(ax):
                            if any(ai in mii for mii in mions):
                                amx.append(jk)
                    nm = len(bmx)+len(ymx)+len(amx)
                    allmx = bmx+ymx+amx
                    if 0 in allmx and 1 in allmx and nm>=3 and len(mx)-nm>0:
                        t5 = True
                    if len(mx)-nm>0:
                        if len(bmx)>=3:
                            dx = [kj-ki==1 for kj, ki in zip(bmx[1:], bmx[:-1])]
                            if sum(dx)>=2:
                                t5 = True
                        if not t5 and len(ymx)>=3:
                            dx = [kj-ki==1 for kj, ki in zip(ymx[1:], ymx[:-1])]
                            if sum(dx)>=2:
                                t5 = True
                        if not t5 and len(amx)>=3:
                            dx = [kj-ki==1 for kj, ki in zip(amx[1:], amx[:-1])]
                            if sum(dx)>=2:
                                t5 = True
                                    
                    # calculate similarity scores
                    m2, spn, sps = max(x[1] for x in sp2), [], []
                    for ik, (_, name) in enumerate(names2):
                        if name:
                            sitex = [l-u+1 for u in ym] if name.startswith('y') else ym
                            mzshift = getshiftmz(names2[ik][1], sitex, [mnitro]*nn, sp2[ik][0])
                            sps.append([mzshift, sp2[ik][1]/m2*100.])
                            spn.append([mzshift, sp2[ik][1]/m2*100.])
                        else:
                            spn.append([sp2[ik][0], sp2[ik][1]/m2*100.])
                    
                    # .. get similarity score
                    intk = [x[1] for x in sps]
                    intk.sort(reverse=True)
                    tsq = intk[0]/intk[1]>=0.5
                    s = calsim(sps,spk,NORMALIZE=False,USESQUARE=tsq)
                    mt3 = matchtops(spn, sp1)

                    tdf = ppid2dtaname(ks, PPres[k][3])
                    
                    tsims.append((s, mt3, tdf))

                    # copy file
                    copyfile(dtadir+tdf, '%s/%s'%(tf, tdf))
                    
                if not tsims:
                    s, mt3 = 0., False
                else:
                    sk = [x[0] for x in tsims]
                    s, mt3, _ = tsims[sk.index(max(sk))]
                    wlines = ['%s\t%.4f\n'%(x2, x1) for x1, _, x2 in tsims]
                    with open('%s/readme.txt'%tf,'w') as fw:
                        fw.writelines(wlines)

                f.write('%s\t%s\t%s\t%s\t%d\t%s\t%.4f\t%s\t%s\t%s\n'%(ks,
                                                                          spid, seq,
                                                                          mod,
                                                                          c,
                                                                          ','.join('%.4f'%x for x in nscore),
                                                                          s,
                                                                          str(t4),
                                                                          str(t5),
                                                                          str(mt3)))
                        
##                f.write('%s\t%s\t%s\t%s\t%d\t%s\t%.4f\t%s\t%s\t%s\t%s\n'%(ks,
##                                                                          spid, seq,
##                                                                          mod,
##                                                                          c,
##                                                                          ','.join('%.4f'%x for x in nscore),
##                                                                          s,
##                                                                          str(t4),
##                                                                          str(t5),
##                                                                          str(mt3),
##                                                                          str(manualchecked)))
                print((ks, ':', j, '/', len(nitropps)))


def getallnitros():
    """
    Extract spectra of all identified PTN by ProteinPilot into a mgf
    file for Comet search for confirmation of true and false positives.
    """
    with open('ppnitros.mgf','w') as f:
        jk = 0
        for i, tpfile in enumerate(MANUALVALIDATED):
            ks = RAWSETS[i]
            dtadir = r'./%s/dtafiles/'%ks
            # read ProteinPilot results
            PPres = readPeptideSummary(getpepsumdir(ks))
            nitropps = [x for x in PPres if 'Nitro' in x[1]]
            for seq, mod, c, ppid, rt, _ in nitropps:
                dfname = ppid2dtaname(ks, ppid)
                sdf = dfname.split('.')
                f.write('BEGIN IONS\n')
                f.write('RAWFILE=%s.%s\n'%(ks, sdf[0]))
                f.write('TITLE=%s\n'%dfname)
                sp = []
                with open('%s%s'%(dtadir, dfname), 'r') as fr:
                    for line in fr:
                        sp.append(line)
                pmz = float(sp[0].rstrip().split()[0])
                sp = sp[1:]
                f.write('PEPMASS=%.6f\n'%(pmz))
                f.write('CHARGE=%d+\n'%c)
                jk += 1
                f.write('RTINSECONDS=%s\n'%(jk*60.))
                f.writelines(sp)
                f.write('END IONS\n')


def checknitropep(setname):
    """
    Check whether the identified nitrated peptides are true
    """
    NitroDetermine = nitrodeterm.NitroDetermine
    dtadir = r'./%s/dtafiles_processed/'%setname
    PPres = readPeptideSummary(getpepsumdir(setname))
    jx = RAWSETS.index(setname)
    thr_conf, filenum = CONF[jx], FILENUM[jx]
    PPres = [x for x in PPres if x[5]>=thr_conf]
    PPpeps, PPexps = [], []
    for ppseq, ppmod, _, ppid, _, _, _ in PPres:
        if 'Nitro(Y)' in ppmod: continue
        PPpeps.append(combinemod(ppseq, ppmod))
        sid = ppid.split('.')
        PPexps.append((int(sid[3]), int(sid[4])))
    print((len(PPpeps)))
    
    M = getmatchresults('match_raw_analogue_%s_g_add3x.txt'%setname)
    fxs = ['%s#%s'%(x[1], x[6]) for x in M]
    M = [M[fxs.index(x)] for x in set(fxs)]
    C, k, kt = [], 0, len(M)
    for _, spid, seq, mod, c, nmod, df, pmz, ns, simscore, m3 in M:
        if mod:
            cseq = combinemod(seq, mod)
        # criteria 1. top 3 peaks are matched
        t1 = m3
        # criteria 2. the length of peptide sequence >= 7
        l = len(seq)
        t2 = l>=7
        # criteria 3. similarity score between spectra generated by
        # nitrated and non-nitrated peptides >= 0.8
        t3 = simscore>=0.8
        # criteria 4. nitrated peptide has an elution time that is
        # later than its non-modified counterpart
        sdf = df.split('.')
        cyc, exp = int(sdf[1]), int(sdf[2])
        t4 = False
        for i, (ppcyc, ppexp) in enumerate(PPexps):
            if PPpeps[i]==cseq:
                if cyc>ppcyc or (cyc==ppcyc and exp>ppexp):
                    t4 = True
                    break
        # criteria 5. characteristic peaks (45 Da shift corresponding
        # to 3-nitro group) must be observed in the MS/MS spectrum
        nd1 = NitroDetermine(seq, mod, c)
        nd2 = NitroDetermine(seq, nmod, c)
        sp1 = extractSpectrum(setname, filenum, spid)
        sp2 = dtareader(os.path.join(dtadir, df))
        names1 = nd1.annotatespectrum(sp1)
        names2 = nd2.annotatespectrum(sp2)
        mx = []
        for mz1, ni1 in names1:
            if not ni1: continue
            nix1 = [ni.split('/')[0] for ni in ni1.split(',') if '-' not in ni]
            if not nix1: continue
            for mz2, ni2 in names2:
                nix2 = [ni.split('/')[0] for ni in ni2.split(',') if '-' not in ni]
                if not nix2: continue
                nim = [ni for ni in nix1 if ni in nix2]
                if nim: mx.append(nim)
        t5 = False
        if mx:
            ym = [int(x.split('@')[1]) for x in nmod.split(';') if 'Nitro' in x]
            bx = ['b%d'%i for i in range(l) if any(i+1>=j for j in ym)]
            yx = ['y%d'%i for i in range(l) if any(i>=l-j for j in ym)]
            ax = ['a%d'%i for i in range(l) if any(i+1>=j for j in ym)]
            bmx, ymx, amx = [], [], []
            for mions in mx:
                for j, bi in enumerate(bx):
                    if any(bi in mii for mii in mions):
                        bmx.append(j)
                for j, yi in enumerate(yx):
                    if any(yi in mii for mii in mions):
                        ymx.append(j)
                for j, ai in enumerate(ax):
                    if any(ai in mii for mii in mions):
                        amx.append(j)
            nm = len(bmx)+len(ymx)+len(amx)
            allmx = bmx+ymx+amx
            if 0 in allmx and 1 in allmx and nm>=3 and len(mx)-nm>0:
                t5 = True
            if len(mx)-nm>0:
                if len(bmx)>=3:
                    dx = [kj-ki==1 for kj, ki in zip(bmx[1:], bmx[:-1])]
                    if sum(dx)>=2:
                        t5 = True
                if not t5 and len(ymx)>=3:
                    dx = [kj-ki==1 for kj, ki in zip(ymx[1:], ymx[:-1])]
                    if sum(dx)>=2:
                        t5 = True
                if not t5 and len(amx)>=3:
                    dx = [kj-ki==1 for kj, ki in zip(amx[1:], amx[:-1])]
                    if sum(dx)>=2:
                        t5 = True
        C.append((t1, t2, t3, t4, t5))

        k += 1
        if k%100==0:
            print(('%d/%d'%(k, kt), (t1, t2, t3, t4, t5)))

    return M, C


def extractSynthesizedSpectra(pmz, c, spectrum, synfile, tol = 0.05):
    """
    extract synthesized spectra according to the input precursor
    mass to charge ratio (m/z) and corresponding spectrum
    """
    centroidms, denoise = pms.centroidms, pms.denoise
    calsim, matchtops = pms.calsimilarity, pms.matchtops
    begin_sp_read = False
    specmatch = []
    with open(synfile, 'r') as f:
        for line in f:
            if 'BEGIN IONS' in line:
                begin_sp_read = True
                cspec = []
            elif 'END IONS' in line:
                begin_sp_read = False
                if expmz>=pmz-tol and expmz<=pmz+tol and c==charge:
                    spx = centroidms(cspec)
                    spx = denoise(spx, 'median', None)
                    s = calsim(spectrum, spx)
                    p = matchtops(spectrum, spx, mztol=0.2)
                    specmatch.append((title, s, p))
            elif begin_sp_read:
                if line.startswith('TITLE'):
                    title = line.rstrip().split('=')[1]
                elif line.startswith('CHARGE'):
                    charge = int(line.rstrip().split('=')[1][0])
                elif line.startswith('PEPMASS'):
                    expmz = float(line.rstrip().split('=')[1])
                else:
                    try:
                        cspec.append([float(x) for x in line.rstrip().split()])
                    except:
                        pass
    
    return specmatch


def extractTrueMSfromWholeDataSet():
    """
    extract all tandem mass spectra from dtafiles in whole datasets
    according to the information of true positives
    """
    tol = 1.
    centroidms, denoise = pms.centroidms, pms.denoise
    calsim, matchtops = pms.calsimilarity, pms.matchtops

    # get manually validated results
    tps = []
    for setname in RAWSETS:
        res = readPeptideSummary(getpepsumdir(setname))
        tps += [(x[3], x[0], x[1], setname) for x in res if 'Nitro(Y)' in x[1]]
    
    # find potentially unidentified nitrated peptides according to
    # the true identified peptides
    ntps = len(tps)
    ppid = '%s.1.1.%s.%s'
    tpline = 'DATASET=%s, MASSPECTRUMID=%s, SEQUENCE=%s, '+\
             'MODIFICATIONS=%s, PRECURSORMZ=%s, CHARGE=%s\n'
    with open('matched_raw.txt','a') as f:
        for i, (spid, seq, mod, rawset) in enumerate(tps):
            if i<405: continue
            # extract charge state of the manually validated results
            mod, spid = mod.replace(' ',''), spid.replace(' ','')
            mod = ';'.join(x for x in mod.split(';') if not x.startswith('No'))
            # read information of the true positive from Peptide Summary
            pepsumdir = getpepsumdir(rawset)
            t = False
            with open(pepsumdir,'r') as sumf:
                for line in sumf:
                    if not t:
                        lx = line.rstrip().split('\t')
                        pmix,cix = lx.index('Prec m/z'),lx.index('Theor z')
                        t = True
                    else:
                        if spid in line and seq in line and 'Nitro(Y)' in line:
                            lx = line.rstrip().split('\t')
                            pmz, charge = lx[pmix], lx[cix]
                            break
            # write true positive information to file
            f.write(tpline%(rawset, spid, seq, mod, pmz, charge))

            # get mass spectrum of current validated peptide
            k = RAWSETS.index(rawset)
            spectrum = extractSpectrum(rawset, FILENUM[k], spid)
            spectrum = centroidms(spectrum)
            spectrum = denoise(spectrum, 'median')
            pmz = float(pmz)
            nd = nitrodeterm.NitroDetermine(seq, mod, int(charge))
            
            for k, rawsetx in enumerate(RAWSETS):
                pepsumdir = getpepsumdir(rawsetx)
                PPres = readPeptideSummary(pepsumdir)
                ppidall = [res[3] for res in PPres]
                dtadir = r'./%s/dtafiles/'%rawsetx
            
                # find spectra generated by potentially nitrated peptides
                for fileorder, fileprefix in list(FILENUM[k].items()):
                    with open(r'./%s/dtafiles/%s.summary.txt'%(rawsetx,fileprefix),'rb') as f1:
                        sumreader = csv.DictReader(f1, delimiter='\t')
                        for line in sumreader:
                            pmz_c = float(line['Precursor m/z'])
                            dtafile = os.path.join(dtadir,line['dta File Name'])
                            rt = line['Retention Time (sec)']
                        
                            if abs(float(pmz_c)-pmz)>tol:
                                continue
                            # read spectrum from dta file
                            spectrum4match = []
                            with open(dtafile,'r') as f2:
                                for line2 in f2:
                                    spectrum4match.append([float(x) for x in
                                                           line2.rstrip().split('\t')])
                            spectrum4match = spectrum4match[1:]
                            
                            # similarity score
                            if len(spectrum4match)<20: continue
                            spectrum4match = centroidms(spectrum4match)
                            if len(spectrum4match)<10: continue
                            spectrum4match = denoise(spectrum4match, 'median')
                            s = calsim(spectrum, spectrum4match)
                            # identify whether top 3 peaks are matched
                            p = matchtops(spectrum, spectrum4match)
                            if s<0.6 and not p:
                                continue
                            d = nd.score(spectrum4match)
                            # find corresponding spectrum id in ProteinPilot
                            # Peptide Summary file and get confidence
                            cycle, expm = tuple(dtafile.split('.')[1:3])
                            cppid = ppid%(fileorder,cycle,expm)
                            if cppid in ppidall:
                                j = ppidall.index(cppid)
                                conf = '%.4f'%(float(PPres[j][6]))
                            else:
                                conf = 'None'
                            # write dta file name, retention time, confidence
                            # similarity score and top peak identifier to
                            # the file
                            f.write('\t'.join([rawsetx, dtafile, rt, conf,
                                               '%.4f'%d[0], '%.4f'%s, str(p)]))
                            f.write('\n')
            print((i, 'of', ntps, 'is processed from set', rawset))


def extractUnidentifiedAnalogue(rawset):
    """
    extract all nitrated analogues using non-nitrated peptides by
    comparing unidentified mass spectra with those generated by
    non-nitrated peptides.
    """
    tol, mnitro = 3., 44.985078

    # localize global functions
    centroidms, denoise = pms.centroidms, pms.denoise
    NitroDetermine = nitrodeterm.NitroDetermine
    parsemodifications, theoions = annotesp.parsemodifications, annotesp.theoions
    calpepmass = annotesp.calpepmass
    calsim, matchtops = pms.calsimilarity, pms.matchtops
    isfile = os.path.isfile

    # set up constants
    header = 'Data Set\tSpectrum ID\tObserved m/z\tNitro Score\t' +\
                   'Similarity Score\tTop Peak Match\n'
    title = 'DATASET=%s, MASSPECTRUMID=%s, SEQUENCE=%s, MODIFICATIONS=%s,'+\
            ' REVISED_MODIFICATIONS=%s, PRECURSORMZ=%.4f, CHARGE=%d, RETENTIONTIME=%s\n'
    cands = '%s\t%s\t%.4f\t%s\t%.4f\t%s\n'

    # prepare datasets
    rj = RAWSETS.index(rawset)
    pepsumdir = getpepsumdir(rawset)
    filenum = FILENUM[rj]
    confthr = CONF[rj]
    rx = []
    # .. get all potential analogues
    analogues = readPeptideSummary(pepsumdir)
    for j, (seq, mods, _, _, _, pepconf, _) in enumerate(analogues):
        # .. only free Y-included sequences are considered
        if 'Y' not in seq or mods.count('(Y)')==seq.count('Y') or\
           'Nitro' in mods or pepconf<confthr:
            continue
        rx.append(j)
    analogues = [analogues[j] for j in rx]

##    # remove identified list
##    delist = getprocessedlist(rawset)
##    dx = 0
##    for delid, delseq in delist:
##        k = [x for x in analogues if delid in x and delseq in x]
##        for x in k:
##            analogues.remove(x)
##
##    # group all candidates to remove replicates
##    analogues = groupcandidates(analogues, rawset)
##    analogues = analogues[::-1]
    
    peps = ['%s#%s#%d'%(seq, mods.replace(' ',''), charge)
            for seq, mods, charge, _, _, _, _ in analogues]
    uniquepeps = set(peps)

    # get processed dta files then remove it from candidates
    dtas = set()
    with open('match_raw_analogue_%s_g.txt'%rawset, 'r') as f:
        for line in f:
            sline = line.split('\t')
            if len(sline)>2 and (sline[1].startswith('F') or sline[1].startswith('f')):
                dtas.add(sline[1])

    # get all candidate m/z
    dtadir = r'./%s/dtafiles_processed/'%rawset
    dtasumdir = r'./%s/dtafiles/'%rawset
    candidates, k = [], 0
    for fileprefix in list(FILENUM[rj].values()):
        with open('%s%s.summary.txt'%(dtasumdir,fileprefix),'rb') as mf:
            mfreader = csv.DictReader(mf, delimiter='\t')
            for mfrow in mfreader:
                k += 1
                mzc = float(mfrow['Precursor m/z'])
                if mfrow['dta File Name'] not in dtas:
                    dtafile = os.path.join(dtadir,mfrow['dta File Name'])
                    candidates.append((dtafile, mzc))
    print((len(dtas), len(candidates), k))

    # set up file for writing results
    writefilename = 'match_raw_analogue_%s_g_add3.txt'%rawset
    with open(writefilename,'w') as write2file:
        write2file.write(header)

        # get nitrated analogues
        kk, kkt = 0, len(analogues)
        for pepi in uniquepeps:
            kx = [ki for ki, pepx in enumerate(peps) if pepx==pepi]
            seq, mods, charge, _, _, _, _ = analogues[kx[0]]
            mods = ';'.join([m for m in mods.split(';')
                             if not m.lstrip().startswith('No')])
            try:
                modpx = parsemodifications(mods)
            except:
                continue
            
            # get unmodified Y
            yix = [j+1 for j, a in enumerate(seq) if a=='Y']
            # .. remove indices of modified tyrosine (Y)
            if modpx:
                yix = [j for j in yix if not any(j==k for _, k, _ in modpx)]

            tm = calpepmass(seq, modpx)
            bs, ys, pmass = theoions(tm[1], tm[0], tm[2])
            refions = annotesp.generateions(bs, ys, pmass, charge)
            
            # iterate through all combinations of indices of tyrosince, and
            # get candidates
            modcs, mzs, nitroscores, nsx, nnitr = [], [], [], [], []
            dtas, dtans, dtamzs, dtafiles = [], [], [], []
            for jx in chain.from_iterable(combinations(yix, k+1)
                                          for k in range(len(yix))):
                modc = mods
                for kj in jx:
                    modc += '; Nitro(Y)@%d'%kj
                nsx.append(jx)
                modc = modc.lstrip(';')
                modcs.append(modc)
                nnitr.append(len(jx))
                # get parsed modifications
                modpc = parsemodifications(modc)
                # get theoretical ions
                tm = calpepmass(seq, modpc)
                bs, ys, pmass = theoions(tm[1], tm[0], tm[2])
                ionst = annotesp.generateions(bs, ys, pmass, charge)
                # get m/z valus and scoring object
                nd = NitroDetermine(seq, modc, charge)
                mz = nd.mz()
                scorex = nd.score
                mzs.append(mz)
                # get candidates and nitro scores
                ds, jx, sps, inms, mzx, filex = [], [], [], [], [], []
                for dtafile, mzi in candidates:
                    if mzi<mz-tol or mzi>mz+tol: continue
                    if not isfile(dtafile): continue
                    dtasp = dtareader(dtafile)
                    if len(dtasp)<10: continue
                    d = scorex(dtasp)
                    #if not any(x>3. for x in d): continue
                    
                    cnames = getionnames(dtasp,_,_,_,ionst,calions=False)
                    cm = max(x[1] for x in dtasp)
                    spk = [[dtasp[k][0], dtasp[k][1]/cm*100.]
                           for k,nx in enumerate(cnames) if nx]
                    
                    ds.append(d)
                    sps.append(dtasp)
                    inms.append(spk)
                    mzx.append(mzi)
                    filex.append(os.path.basename(dtafile))
                    
                nitroscores.append(ds)
                dtas.append(sps)
                dtans.append(inms)
                dtamzs.append(mzx)
                dtafiles.append(filex)
                
            nc = len(modcs)

            # iterate all peptides and modification combinations
            for ki in kx:
                _, _, _, spid, time, _, _ = analogues[ki]
                kk += 1
                print(('%d/%d:'%(kk, kkt), spid, seq, mods.replace(' ','')))
            
                # extract tandem spectrum
                refsp = extractSpectrum(rawset, filenum, spid)
                if not refsp: continue
                ionnames = getionnames(refsp, _, _, _, refions, calions=False)
                refm = max(x[1] for x in refsp)
        
                # find spectra generated by potentially nitrated peptides
                for ii in range(nc):
                    nsxj = nsx[ii]
                    # get mass spectrum with shifted m/z
                    lmod = [mnitro]*nnitr[ii]
                    spshift, spn, h = [], [], len(seq)
                    for k, name in enumerate(ionnames):
                        if name:
                            sitex = [h-u+1 for u in nsxj] if name.startswith('y') else nsxj
                            mzshift = getshiftmz(ionnames[k], sitex, lmod, refsp[k][0])
                            spshift.append([mzshift, refsp[k][1]/refm*100.])
                            spn.append([mzshift, refsp[k][1]/refm*100.])
                        else:
                            spn.append([refsp[k][0], refsp[k][1]/refm*100.])
                    
                    candinfo = []
                    ds, csps, spcs = nitroscores[ii], dtas[ii], dtans[ii],
                    cmzs, files = dtamzs[ii], dtafiles[ii]
                    for jk, spc in enumerate(csps):
                        # .. get similarity score
                        s = calsim(spshift,spcs[jk],NORMALIZE=False)
                        t = matchtops(spn, spc)
                        if s<0.6 and not t: continue
                        if s is None: continue
                        # .. write results to file
                        candinfo.append(cands%(rawset, files[jk], cmzs[jk],
                                               ','.join(['%.4f'%x for x in ds[jk]]),
                                               s, str(t)))
                    # set line for writing to result file
                    l = title%(rawset,spid,seq,mods,modcs[ii],mzs[ii],charge,time)
                    write2file.write(l)
                    if candinfo:
                        write2file.writelines(candinfo)


def comparereps():
    """
    Compare spectra of replicate identifications
    """
    # reallocate global functions to local functions
    calsim = pms.calsimilarity
##    centroidms, denoise = pms.centroidms, pms.denoise
    isfile = os.path.isfile
    ptol1 = 1.5
    
    #sims = []
    # get all results
    allres = []
    for i, rawset in enumerate(RAWSETS):
        pepsumdir = getpepsumdir(rawset)
        ppres = readPeptideSummary(pepsumdir)
        j = len(ppres[0])-2
        confthr = CONF[i]
        allres += [(rawset, x) for x in ppres if x[j]>=confthr]

    # get all unique sequences
    peps, dfs, pmzs = [], [], []
    for rawset, (seq, mod, c, ppid, _, _, pepmz) in allres:
        pt = combinemod(seq, mod)
        peps.append('%s#%d'%(pt, c))
        dfs.append((rawset, ppid2dtaname(rawset, ppid)))
        pmzs.append(pepmz)
    upeps = set(peps)
    tk = len(upeps)
    print(('# total U seqs:', tk, '# total seqs:', len(peps)))
    return

##    # remove calculated pairs
##    processedfiles = []
##    with open('allsimscores.txt','r') as f:
##        for line in f:
##            processedfiles.append(line.rstrip().split()[0])
##    
##    jx = 0
##    for jk, pepi in enumerate(upeps):
##        ix = [j for j, x in enumerate(peps) if x==pepi]
##        if len(ix)==1: continue
##        pairs, pairfs = [], []
##        for j in ix:
##            pairfs.append('%s:%s'%dfs[j])
##        for j, k in combinations(xrange(len(ix)), 2):
##            pairs.append('%s&%s'%(pairfs[j], pairfs[k]))
##        if all(x in processedfiles for x in pairs):
##            jx = jk
##        else:
##            break
##    print jx
##    upeps = list(upeps)[11000:]
##    tk = len(upeps)

    # calculate similarities for all replicates
    dtadir = r'./%s/dtafiles_processed/%s'
    with open('allsimscores_n.txt','w') as f:
        for jk, pepi in enumerate(upeps):
            c = float(pepi[-1])
            ptol = ptol1/c
            ix = [j for j, x in enumerate(peps) if x==pepi]
            if len(ix)==1:
                if (jk+1)%100 == 0:
                    print(('%d/%d'%(jk+1, tk)))
                continue
            # get all spectra of replicates
            sps, usesqrt_tag, delx, simsc, pairfs = [], [], [], [], []
            for j in ix:
                tx = '%s:%s'%dfs[j]
                if tx in pairfs or not isfile(dtadir%dfs[j]):
                    delx.append(j)
                    continue
                
                spi = dtareader(dtadir%dfs[j])
##                spx = centroidms(spi)
##                spx = denoise(spx, 'median')
                # .. remove precursor peaks
                pmh1 = pmzs[j]*c-(c-1.)*1.0073
                spi = [x for x in spi if x[0]>pmzs[j]+ptol or x[0]<pmzs[j]-ptol]
                spi = [x for x in spi if x[0]>pmh1+ptol1 or x[0]<pmh1-ptol1]
                sps.append(spi)
                # .. whether use sqrt during similarity calculation
                stx = sorted([x[1] for x in spi], reverse=True)
                usesqrt_tag.append(stx[0]/stx[1]>=2. or stx[1]/stx[2]>=2.)
                # .. raw file and dta file names
                pairfs.append(tx)
                
            for j in delx:
                ix.remove(j)
            if len(ix)<2: continue

            # calculate similarities between replicates
            for j, k in combinations(list(range(len(ix))), 2):
                usesqrt = usesqrt_tag[j] or usesqrt_tag[k]
                s = calsim(sps[j],sps[k], USESQUARE=usesqrt, NUMTHRESHOLD=0)
                simsc.append('%s&%s\t%.4f\n'%(pairfs[j], pairfs[k], s))
            f.writelines(simsc)
            #sims.append(simsc)

            if (jk+1)%100 == 0:
                print(('%d/%d'%(jk+1, tk)))

    #return sims


def nitroScorePermTest(seq, mod, c, spx, tol=0.2):
    """
    Modified nitroscores with permutation tests
    """
    normCDF = normalCDF

    # parameters for calculating nitroscores
    cj = 0.7
    wx = [(0.9, 0.5), (2.0, 1.11), (2.0, 1.13), (2.5, 1.38), (2.72, 1.5),
          (3.0, 1.77), (3.5, 2.04), (3.5, 2.07), (4.0, 2.26), (4.0, 2.28),
          (4.0, 2.37), (4.27, 2.5), (4.5, 2.47), (4.5, 2.48), (4.5, 2.49),
          (4.5, 2.59), (4.5, 2.65), (5.0, 2.77), (5.0, 2.95)]
    # constants
    g, h = 0.2/2., log10(exp(1))

    # get nitroscores with different coefficients
    m = max(x[1] for x in spx)
    spx2 = [[x[0], x[1]/m] for x in spx]

    # get nitroscores
    qx = []
    modx = parsemodifications(mod, PTMDB)
    xi, xr = nitroionmatch(spx2, seq, modx, c, 0.2, mtype='mono')
    xr2 = sum(2*(1.-normCDF(abs(x)/g)) for x in xr)
    xi2 = sum(x**cj for x in xi)
    skx = [(w1*xr2+w2*xi2)/sqrt(len(seq)) for w1,w2 in wx]

    # random sequences
    rseq, rmods = randomseq(seq, mod, 0.05)
    if not rseq:
        return skx[10], 0.
    rxi, rxr, rl = [], [], []
    for j, sj in enumerate(rseq):
        xi, xr = nitroionmatch(spx2, sj, rmods[j], c, 0.2)
        rxr.append(sum(2*(1.-normCDF(abs(x)/g)) for x in xr))
        rxi.append(sum(x**cj for x in xi))
        rl.append(sqrt(len(sj)))

    # permutation tests
    for i, (w1,w2) in enumerate(wx): 
        sx = [(w1*x1+w2*x2)/rl[j] for j, (x1, x2) in enumerate(zip(rxr, rxi))]
        nkx, bins = histogram(sx, bins=100)
        sxm = 0.
        for j in range(5):
            if nkx[j]==0: continue
            if nkx[j+1]==0:
                sxm = bins[j+1]
                continue
            rj1 = nkx[j]/nkx[j+1]
            rj2 = nkx[j+2]/nkx[j+1]
            if rj1>=2 and rj2>=1.:
                sxm = bins[j+1]
                break
        sx = [x for x in sx if x>=sxm]
        shape, loc, scale = stats.weibull_min.fit(sx)
        qx.append(((skx[i]-loc)/scale)**shape*h)

    if not qx:
        return 0., 0.

    return skx[10], pms.median(qx)


def singleSimilarityScorewithPermTest(seq, mod, c, spx, tol=0.2):
    """
    Similarity scores between nitrated and non-nitrated spectra
    """
    centroidms, denoise = pms.centroidms, pms.denoise
    similarity = pms.similarity
    normCDF, fitweibull = normalCDF, stats.weibull_min.fit
    joinpath = os.path.join
    randsample = random.sample
    # proposed weights
    wx = [(0.9, 0.5), (2.0, 1.11), (2.0, 1.13), (2.5, 1.38), (2.72, 1.5),
          (3.0, 1.77), (3.5, 2.04), (3.5, 2.07), (4.0, 2.26), (4.0, 2.28),
          (4.0, 2.37), (4.27, 2.5), (4.5, 2.47), (4.5, 2.48), (4.5, 2.49),
          (4.5, 2.59), (4.5, 2.65), (5.0, 2.77), (5.0, 2.95)]
    cj = 0.7
    g, h = 0.2/2., log10(exp(1))

    tpep = '%s#%d'%(combinemod(seq, mod), c)

    # read synthesized data file and calculate similarity scores and nitroscores
    dtadir = r'./tpspectra_raw'
    kj, nitros, nitropeps = -1, [], []
    with open('nitro_match_comet.txt', 'r') as f:
        freader = csv.DictReader(f, delimiter='\t')
        for r in freader:
            spid, seqc, cc = r['SPECTRUMID'], r['SEQUENCE'], int(r['CHARGE'])
            modc = ';'.join(x for x in r['MODIFICATIONS'].split(';') if not x.startswith('No'))
            nitros.append((spid, seqc, modc, cc, r['VALIDATED']))
            nitropeps.append('%s#%d'%(combinemod(seqc, modc), cc))

    # unique peptide sequences
    kx = [i for i, x in enumerate(nitropeps) if x == tpep]
    # .. iterate through all replicants
    spts = []
    for k in kx:
        # .. read experimental mass spectra
        spid = nitros[k][0]
        fx = [fj for fj in os.listdir(dtadir) if seq in fj]
        for fj in fx:
            dtar = joinpath(dtadir, fj)
            dtafull = joinpath(dtar, spid)
            if os.path.isfile(dtafull):
                fs = [fk for fk in os.listdir(dtar)
                      if fk.endswith('dta') and spid not in fk]
                if fs:
                    for fk in fs:
                        spt = dtareader(joinpath(dtar, fk))
                        spt = centroidms(spt)
                        spt = denoise(spt, 'median')
                        spt = removetagfrag(spt)
                        spts.append(spt)
                break

    s, p = 0., 0.
    if spts:
        sjx, mx = [], []
        for spt in spts:
            si, mj = matchNonNitro(seq, mod, c, spx, spt, 0.2)
            sjx.append(si)
            mx.append(mj)
        s = max(sjx)
        mj = mx[sjx.index(s)]
        # .. .. .. random permutation
        m1 = [x[0] for x in mj]
        m2 = [x[1] for x in mj]
        lx, sxt = len(m1), []
        for j in range(10000):
            mxj = randsample(m1, lx)
            sxt.append(similarity(m2, mxj))
        shape, loc, scale = fitweibull(sxt)
        p = ((s-loc)/scale)**shape*h

    return s, p


def negScorewithPermTest(synsetx):
    """
    """
    centroidms, denoise = pms.centroidms, pms.denoise
    similarity = pms.similarity
    normCDF, fitweibull = normalCDF, stats.weibull_min.fit
    joinpath = os.path.join
    randsample = random.sample
    # proposed weights
    wx = [(0.9, 0.5), (2.0, 1.11), (2.0, 1.13), (2.5, 1.38), (2.72, 1.5),
          (3.0, 1.77), (3.5, 2.04), (3.5, 2.07), (4.0, 2.26), (4.0, 2.28),
          (4.0, 2.37), (4.27, 2.5), (4.5, 2.47), (4.5, 2.48), (4.5, 2.49),
          (4.5, 2.59), (4.5, 2.65), (5.0, 2.77), (5.0, 2.95)]
    cj = 0.7
    g, h = 0.2/2., log10(exp(1))

    # read synthesized data file and calculate similarity scores and nitroscores
    dtadir = r'./tpspectra_raw'
    kj, nitros, nitropeps = -1, [], []
    with open('nitro_match_comet.txt', 'r') as f:
        freader = csv.DictReader(f, delimiter='\t')
        for r in freader:
            spid, seqc, cc = r['SPECTRUMID'], r['SEQUENCE'], int(r['CHARGE'])
            modc = ';'.join(x for x in r['MODIFICATIONS'].split(';') if not x.startswith('No'))
            nitros.append((spid, seqc, modc, cc, r['VALIDATED']))
            nitropeps.append(combinemod(seqc, modc))

    #
    pepx = []
    for x in synsetx:
        if '@' in x[1]:
            pepx.append(combinemod(x[0], x[1]))
        else:
            pepx.append(combinemod(x[0], convertmod(x[1])))
    
    ns, ss = [], []
    for pj in set(pepx):
        ix = [i for i,x in enumerate(pepx) if x==pj]
        k = ix[0]
        seq, mod, c = synsetx[k][0], synsetx[k][1], synsetx[k][2]
        if '@' not in mod:
            mod = convertmod(mod)
        modn = ';'.join(xk for xk in mod.split(';') if 'Nitro' not in xk)
        nd = nitrodeterm.NitroDetermine(seq, mod, c)
        nd_n = nitrodeterm.NitroDetermine(seq, modn , c)
        modx = parsemodifications(mod, PTMDB)
        modx2 = [xk if not 'Nitro' in xk else (0., xk[1], xk[2]) for xk in modx]
        rseq, rmods = randomseq(seq, mod, 0.05)
        rmods2 = []
        for x in rmods:
            rmods2.append(tuple(list(xk if not 'Nitro' in xk else (0., xk[1], xk[2]) for xk in x)))

        # read false non-nitrated analogues
        kx = [i for i, x in enumerate(nitropeps) if x != pj]
        # .. iterate through all replicants
        spts, ionns = [], []
        for k in kx:
            # .. read experimental mass spectra
            spid = nitros[k][0]
            fx = [fj for fj in os.listdir(dtadir) if nitros[k][1] in fj]
            for fj in fx:
                dtar = joinpath(dtadir, fj)
                dtafull = joinpath(dtar, spid)
                if os.path.isfile(dtafull):
                    fs = [fk for fk in os.listdir(dtar)
                          if fk.endswith('dta') and spid not in fk]
                    if fs:
                        for fk in fs:
                            spt = dtareader(joinpath(dtar, fk))
                            spt = centroidms(spt)
                            spt = denoise(spt, 'median')
                            spt = removetagfrag(spt)
                            spts.append(spt)
                            ionns.append(nd_n.annotations(spt, 0.2))
                    break

        # get all scores
        for ik in ix:
            spx = dtareader(r'./synthetic_peptides/%s/%s'%synsetx[ik][3])
            spx = centroidms(spx)
            spx = denoise(spx, 'median')
            spx = removetagfrag(spx)
            ions = nd.annotations(spx, 0.2)
            m = max(x[1] for x in spx)
            spx2 = [[x[0], x[1]/m] for x in spx]
            rxi, rxr, rl = [], [], []
            for j, sj in enumerate(rseq):
                xi, xr = nitroionmatch(spx2, sj, rmods2[j], c, 0.2)
                rxr.append(sum(2*(1.-normCDF(abs(x)/g)) for x in xr))
                rxi.append(sum(x**cj for x in xi))
                rl.append(sqrt(len(sj)))

            # get nitroscores
            qx = []
            xi, xr = nitroionmatch(spx2, seq, modx2, c, 0.2, mtype='mono')
            xr2 = sum(2*(1.-normCDF(abs(x)/g)) for x in xr)
            xi2 = sum(x**cj for x in xi)
            skx = [(w1*xr2+w2*xi2)/sqrt(len(seq)) for w1,w2 in wx]
            for i, (w1,w2) in enumerate(wx): 
                # permutation tests
                sx = [(w1*x1+w2*x2)/rl[j] for j, (x1, x2) in enumerate(zip(rxr, rxi))]
                nkx, bins = histogram(sx, bins=100)
                sxm = 0.
                for j in range(5):
                    if nkx[j]==0: continue
                    if nkx[j+1]==0:
                        sxm = bins[j+1]
                        continue
                    rj1 = nkx[j]/nkx[j+1]
                    rj2 = nkx[j+2]/nkx[j+1]
                    if rj1>=2 and rj2>=1.:
                        sxm = bins[j+1]
                        break
                sx = [x for x in sx if x>=sxm]
                shape, loc, scale = stats.weibull_min.fit(sx)
                qx.append(((skx[i]-loc)/scale)**shape*h)
            ns.append((ik, skx[10], pms.median(qx)))

            # get similarity scores
            s, p = 0., 0.
            if spts:
                sjx, mx = [], []
                for j,spt in enumerate(spts):
                    si, mj = matchNonNitro2(nd, ions, spx, nd_n, ionns[j], spt)
                    sjx.append(si)
                    mx.append(mj)
                s = max(sjx)
                mj = mx[sjx.index(s)]
                # .. .. .. random permutation
                m1 = [x[0] for x in mj]
                m2 = [x[1] for x in mj]
                lx, sxt = len(m1), []
                for j in range(10000):
                    mxj = randsample(m1, lx)
                    sxt.append(similarity(m2, mxj))
                shape, loc, scale = fitweibull(sxt)
                p = ((s-loc)/scale)**shape*h
            ss.append((ik, s, p))
            
            print((ik, s, skx[10]))
    return ns, ss


def read_dat(datfile):
    """
    Read Mascot search result file (..dat)
    """

    res = []
    read_mass, read_pep, read_dpep, read_info = False, False, False, False
    modx, ntermmod, ctermmod, fixmod, fixmodaa = {}, '', '', [], ''
    peps, qx = [], []
    with open(datfile, 'r') as f:
        for line in f:
            if 'Content-Type' in line:
                if 'name="masses"' in line:
                    read_mass = True
                elif read_mass:
                    read_mass = False
                if 'name="peptides"' in line:
                    read_pep = True
                elif read_pep:
                    read_pep = False
                if 'name="decoy_peptides"' in line:
                    read_dpep = True
                elif read_dpep:
                    read_dpep = False
                if 'name="query' in line:
                    qxj = line.rstrip().split('query')[1].split('"')[0]
                    read_info = True
            if read_info and line.startswith('Ions1'):
                if qxj in qx:
                    j = qx.index(qxj)
                    peps[j] = (peps[j][0], peps[j][1], peps[j][2], c, rt, scan, peps[j][3])
                read_info = False
            
            # read modifications
            if read_mass and line.startswith('delta'):
                mix = line.rstrip().split('=')[0]
                tx = line.rstrip().split(',')[1].split('(')
                mname = tx[0].replace(' ','')
                maa = tx[1].split(')')[0]
                modx[mix] = {'modification': mname, 'aa': maa}
            elif read_mass and line.startswith('FixedMod') and not line.startswith('FixedModResidues'):
                tx = line.rstrip().split(',')[1].split('(')
                mname = tx[0].replace(' ','')
                maa = tx[1].split(')')[0]
                if maa == 'N-term':
                    ntermmod += '%s@%s'%(mname, maa)
                elif maa == 'C-term':
                    ctermmod = '%s@%s'%(mname, maa)
                else:
                    fixmod.append(mname)
                    fixmodaa += maa

            # read peptides
            if read_pep and '_p1=' in line and 'p1=-1' not in line:
                pepline = line.rstrip().split('=')[1].split(';')[0].split(',')
                _, _, _, _, pep, _, modstr, ionscore, _, _, _ = tuple(pepline)
                l = len(modstr)
                mod = []
                if ntermmod:
                    mod.append(ntermmod)
                if ctermmod:
                    mod.append(ctermmod)
                # get AAs that are set to have fixed modifications
                caa = set(pep)&set(fixmodaa)
                if caa:
                    for aa in caa:
                        i = fixmodaa.index(aa)
                        mod += ['%s(%s)@%d'%(fixmod[i], aa, j+1)
                                for j, aaj in enumerate(pep) if aaj==aa]
                # get variable modifications
                for i in range(1, l-1):
                    if modstr[i] != '0':
                        mod.append('%s(%s)@%d'%(modx['delta%s'%modstr[i]]['modification'],
                                                pep[i-1], i))
                
                peps.append(('N', pep, ';'.join(mod), float(ionscore)))
                qx.append(line.split('_')[0][1:])

            # read decoy peptides
            if read_dpep and '_p1=' in line and 'p1=-1' not in line:
                pepline = line.rstrip().split('=')[1].split(';')[0].split(',')
                _, _, _, _, pep, _, modstr, ionscore, _, _, _ = tuple(pepline)

                # get modification information
                mod = []
                l = len(modstr)
                if ntermmod:
                    mod.append(ntermmod)
                if ctermmod:
                    mod.append(ctermmod)
                # get AAs that are set to have fixed modifications
                caa = set(pep)&set(fixmodaa)
                if caa:
                    for aa in caa:
                        i = fixmodaa.index(aa)
                        mod += ['%s(%s)@%d'%(fixmod[i], aa, j+1)
                                for j, aaj in enumerate(pep) if aaj==aa]
                # get variable modifications
                for i in range(1, l-1):
                    if modstr[i] != '0':
                        mod.append('%s(%s)@%d'%(modx['delta%s'%modstr[i]]['modification'],
                                                pep[i-1], i))

                # replace the assignment if the score of decoy hit is
                # higher then that of normal hit
                dqx = line.split('_')[0][1:]
                if dqx in qx:
                    j = qx.index(dqx)
                    if float(ionscore)>peps[j][3]:
                        peps[j] = ('D', pep, ';'.join(mod), float(ionscore))
                else:
                    peps.append(('D', pep, ';'.join(mod), float(ionscore)))
                    qx.append(dqx)

            # get spectrum information
            
            if read_info:
                if line.startswith('rtinseconds'):
                    rt = float(line.rstrip().split('=')[1])/60.
                elif line.startswith('charge'):
                    c = int(line.rstrip().split('=')[1][0])
                elif line.startswith('scans'):
                    scan = int(line.rstrip().split('=')[1])

    return peps
            

def modifiedNitroScorewithPermTest():
    """
    Modified nitroscores with permutation tests
    """
    nitros = []
    normCDF = normalCDF

    # parameters for calculating nitroscores
    cj = 0.7
    wx = [(0.9, 0.5), (2.0, 1.11), (2.0, 1.13), (2.5, 1.38), (2.72, 1.5),
          (3.0, 1.77), (3.5, 2.04), (3.5, 2.07), (4.0, 2.26), (4.0, 2.28),
          (4.0, 2.37), (4.27, 2.5), (4.5, 2.47), (4.5, 2.48), (4.5, 2.49),
          (4.5, 2.59), (4.5, 2.65), (5.0, 2.77), (5.0, 2.95)]
    # constants
    g, h = 0.2/2., log10(exp(1))

    # MAIN
    s = []
    dtadir = r'./tpspectra_raw'
    with open('nitro_match_comet.txt', 'r') as f:
        freader = csv.DictReader(f, delimiter='\t')
        for r in freader:
            spid, seq, c = r['SPECTRUMID'], r['SEQUENCE'], int(r['CHARGE'])
            mod = ';'.join(x for x in r['MODIFICATIONS'].split(';') if not x.startswith('No'))
            nitros.append((spid, seq, mod, c, r['VALIDATED']))

            # read spectrum
            fx = [fj for fj in os.listdir(dtadir) if r['SEQUENCE'] in fj]
            for fj in fx:
                dtafull = os.path.join(os.path.join(dtadir, fj), spid)
                if os.path.isfile(dtafull):
                    spx = dtareader(dtafull)
                    break

            # get nitroscores with different coefficients
            spx = removetagfrag(spx)
            spx = pms.centroidms(spx)
            spx = pms.denoise(spx, 'median')
            m = max(x[1] for x in spx)
            spx2 = [[x[0], x[1]/m] for x in spx]
            modx = parsemodifications(mod, PTMDB)

            # random sequences
            rseq, rmods = randomseq(seq, mod, 0.05)
            rxi, rxr, rl = [], [], []
            for j, sj in enumerate(rseq):
                xi, xr = nitroionmatch(spx2, sj, rmods[j], c, 0.2)
                rxr.append(sum(2*(1.-normCDF(abs(x)/g)) for x in xr))
                rxi.append(sum(x**cj for x in xi))
                rl.append(sqrt(len(sj)))

            # get nitroscores
            qx = []
            xi, xr = nitroionmatch(spx2, seq, modx, c, 0.2, mtype='mono')
            xr2 = [2*(1.-normCDF(abs(x)/g)) for x in xr]
            xi2 = [x**cj for x in xi]
            sk = []
            for w1, w2 in wx:
                sk = sum(w1*x1+w2*x2 for x1, x2 in zip(xr2, xi2))/sqrt(len(seq))
                
                # permutation tests
                sx = [(w1*x1+w2*x2)/rl[j] for j, (x1, x2) in enumerate(zip(rxr, rxi))]
                nkx, bins = histogram(sx, bins=100)
                sxm = 0.
                for j in range(5):
                    if nkx[j]==0: continue
                    if nkx[j+1]==0:
                        sxm = bins[j+1]
                        continue
                    rj1 = nkx[j]/nkx[j+1]
                    rj2 = nkx[j+2]/nkx[j+1]
                    if rj1>=2 and rj2>=1.:
                        sxm = bins[j+1]
                        break
##                if len(s)==2: return sx
                sx = [x for x in sx if x>=sxm]
                shape, loc, scale = stats.weibull_min.fit(sx)
                qx.append((sk, ((sk-loc)/scale)**shape*h))
            
            s.append(qx)
            print((len(s), len(sx)))

    return nitros, s


def similarityScorewithPermTest():
    """
    Similarity scores between nitrated and non-nitrated spectra
    """
    randsample = random.sample
    s, kj, nitros = [], -1, []
    dtadir = r'./tpspectra_raw'
    with open('nitro_match_comet.txt', 'r') as f:
        freader = csv.DictReader(f, delimiter='\t')
        for r in freader:
            spid, seq, c = r['SPECTRUMID'], r['SEQUENCE'], int(r['CHARGE'])
            mod = ';'.join(x for x in r['MODIFICATIONS'].split(';') if not x.startswith('No'))
            nitros.append((spid, seq, mod, c))
            kj += 1

            # read spectrum
            sj, mx = [], []
            fx = [fj for fj in os.listdir(dtadir) if r['SEQUENCE'] in fj]
            for fj in fx:
                dtar = os.path.join(dtadir, fj)
                dtafull = os.path.join(dtar, spid)
                if os.path.isfile(dtafull):
                    spx = dtareader(dtafull)
                    spx = pms.centroidms(spx)
                    spx = pms.denoise(spx, 'median')
                    spx = removetagfrag(spx)
                    fs = [fk for fk in os.listdir(dtar)
                          if fk.endswith('dta') and spid not in fk]
                    if fs:
                        for fk in fs:
                            spt = dtareader(os.path.join(dtar, fk))
                            spt = pms.centroidms(spt)
                            spt = pms.denoise(spt, 'median')
                            spt = removetagfrag(spt)
                            si, mj = matchNonNitro(seq, mod, c, spx, spt, 0.2)
                            sj.append(si)
                            mx.append(mj)
                    break
            if sj:
                smj = max(sj)
                mj = mx[sj.index(smj)]
                # random permutation
                m1 = [x[0] for x in mj]
                m2 = [x[1] for x in mj]
                lx, sx = len(m1), []
                for k in range(10000):
                    mxj = randsample(m1, lx)
                    sx.append(pms.similarity(m2, mxj))
                shape, loc, scale = stats.weibull_min.fit(sx)
                ej = ((smj-loc)/scale)**shape*log10(exp(1))
                s.append((kj, smj, ej))
                print((kj, smj, ej, r['SPECTRUMID']))

    return nitros, s


def testUsingSynthesized():
    """
    Test the scores using synthesized data
    """
    centroidms, denoise, calsim = pms.centroidms, pms.denoise, pms.calsimilarity
    similarity = pms.similarity
    normCDF, fitweibull = normalCDF, stats.weibull_min.fit
    joinpath = os.path.join
    randsample = random.sample
    # proposed weights
    wx = [(0.9, 0.5), (2.0, 1.11), (2.0, 1.13), (2.5, 1.38), (2.72, 1.5),
          (3.0, 1.77), (3.5, 2.04), (3.5, 2.07), (4.0, 2.26), (4.0, 2.28),
          (4.0, 2.37), (4.27, 2.5), (4.5, 2.47), (4.5, 2.48), (4.5, 2.49),
          (4.5, 2.59), (4.5, 2.65), (5.0, 2.77), (5.0, 2.95)]
    cj = 0.7
    g, h = 0.2/2., log10(exp(1))

    # synthesized data file
    synf1 = r'./synthetic_peptides/QQ_iNi-1'
    synf2 = r'./synthetic_peptides/QQ_iNi-2'

    # write to file
    uk = 6
    # .. but firstly to check whether the file is open, if so, close it
    file_handles = []
    fx = 'syn_test_res_xp_%d.txt'%uk
    try:
        file_handles.append(open(fx))
    except:
        pass

    finally:
        for fh in file_handles:
            fh.close()
    fwrite = open(fx, 'w')

    # read synthesized data file and calculate similarity scores and nitroscores
    dtadir = r'./tpspectra_raw'
    kj, nitros, nitropeps = -1, [], []
    with open('nitro_match_comet.txt', 'r') as f:
        freader = csv.DictReader(f, delimiter='\t')
        for r in freader:
            spid, seq, c = r['SPECTRUMID'], r['SEQUENCE'], int(r['CHARGE'])
            mod = ';'.join(x for x in r['MODIFICATIONS'].split(';') if not x.startswith('No'))
            nitros.append((spid, seq, mod, c, r['VALIDATED']))
            nitropeps.append('%s#%d'%(combinemod(seq, mod), c))

    # unique peptide sequences
    upx = set(nitropeps)
    print((len(upx), uk))
    for jk, pk in enumerate(upx):
        if jk>=30*uk or jk<30*(uk-1): continue
        kx = [i for i, x in enumerate(nitropeps) if x == pk]
        k = kx[0]
        seq, mod, c = nitros[k][1], nitros[k][2], nitros[k][3]
        nd = nitrodeterm.NitroDetermine(seq, mod, c)
        ljs = sqrt(len(seq))
        mz = nd.mz()
        modx = parsemodifications(mod, PTMDB)

        # .. random sequences and modifications
        rseq, rmods = randomseq(seq, mod, 0.05)

        # .. read spectrum ID of synthesized spectra
        synx1 = []
        with open(joinpath(synf1, 'readme.txt'), 'r') as fh:
            for line in fh:
                sline = line.rstrip().split('\t')
                if abs(float(sline[1])-mz) <= 0.5:
                    synx1.append((joinpath(synf1, '%s.dta'%sline[0]), float(sline[1]), int(sline[2])))
        
        synx2 = []
        with open(joinpath(synf2, 'readme.txt'), 'r') as fh:
            for line in fh:
                sline = line.rstrip().split('\t')
                if abs(float(sline[1])-mz) <= 0.5:
                    synx2.append((joinpath(synf2, '%s.dta'%sline[0]), float(sline[1]), int(sline[2])))
        synx = synx1+synx2
        nj = len(synx)
        if nj==0:
            for k in kx: print((k, seq))
            continue

        # .. iterate through all replicants
        synsp_matched = []
        for k in kx:
            # .. read experimental mass spectra
            spid, spts = nitros[k][0], []
            fx = [fj for fj in os.listdir(dtadir) if seq in fj]
            for fj in fx:
                dtar = joinpath(dtadir, fj)
                dtafull = joinpath(dtar, spid)
                if os.path.isfile(dtafull):
                    spx = dtareader(dtafull)
                    spx = centroidms(spx)
                    spx = denoise(spx, 'median')
                    spx = removetagfrag(spx)
                    fs = [fk for fk in os.listdir(dtar)
                          if fk.endswith('dta') and spid not in fk]
                    if fs:
                        for fk in fs:
                            spt = dtareader(joinpath(dtar, fk))
                            spt = centroidms(spt)
                            spt = denoise(spt, 'median')
                            spt = removetagfrag(spt)
                            spts.append(spt)
                    break

            # .. read each synthesized spectrum and calculate the scores and p values
            snx = []
            for kk, (spdta, _, c) in enumerate(synx):
                spsx = dtareader(spdta)
                spsx = centroidms(spsx)
                spsx = denoise(spsx, 'median')
                spsx = removetagfrag(spsx)
                # .. .. similarity between synthesized and experimental spectra
                s1 = calsim(spsx, spx)
##                if s1<0.5:
##                    continue

                # .. .. similarity between synthesized spectrum and non-nitrated analog
                s3, s3p = None, None
                if spts:
                    sjx, mx = [], []
                    for spt in spts:
                        si, mj = matchNonNitro(seq, mod, c, spsx, spt, 0.2)
                        sjx.append(si)
                        mx.append(mj)
                    s3 = max(sjx)
                    mj = mx[sjx.index(s3)]
                    # .. .. .. random permutation
                    m1 = [x[0] for x in mj]
                    m2 = [x[1] for x in mj]
                    lx, sxt = len(m1), []
                    for j in range(10000):
                        mxj = randsample(m1, lx)
                        sxt.append(similarity(m2, mxj))
                    shape, loc, scale = fitweibull(sxt)
                    s3p = ((s3-loc)/scale)**shape*h

                # .. .. if this synthesized mass spectrum have been processed
                # .. .. for this peptide, will not process it again
                if spdta in synsp_matched:
                    tx = list([x[2]==spdta for x in snx][0])
                    fwrite.write('SPECTRUM: %s|%s|%s|%s|%d\n'%(spid, spdta, seq, mod, c))
                    fwrite.write('%.6f\n'%tx[2])
                    fwrite.write('|'.join('%.6f'%x for x in tx[3]))
                    fwrite.write('\n')
                    fwrite.write('|'.join('%.6f'%x for x in tx[4]))
                    fwrite.write('\n')
                    if s3 is not None:
                        fwrite.write('%.6f\n'%s3)
                        fwrite.write('%.6f\n'%s3p)
                    else:
                        fwrite.write('None\n')
                        fwrite.write('None\n')
                        s3, s3p = 0., 0.
                    print((k, '%d/%d'%(kk+1, nj), spid, spdta, '%.4f||%.4f||%.4f'%(s1, max(s2), s3)))
                    continue

                # .. .. NitroScore with permutation test at synthesized spectra
                m = max(x[1] for x in spsx)
                spx2 = [[x[0], x[1]/m] for x in spsx]
                xi, xr = nitroionmatch(spx2, seq, modx, c, 0.2, mtype='mono')
                xr2 = sum([2*(1.-normCDF(abs(x)/g)) for x in xr])
                xi2 = sum(x**cj for x in xi)
                s2 = [(w1*xr2+w2*xi2)/ljs for w1, w2 in wx]
                
                # .. .. .. permutation test
                rxi, rxr, rl = [], [], []
                s2p = []
                for j, sj in enumerate(rseq):
                    tj = sqrt(len(sj))
                    xi, xr = nitroionmatch(spx2, sj, rmods[j], c, 0.2)
                    rxr.append(sum(2*(1.-normCDF(abs(x)/g)) for x in xr))
                    rxi.append(sum(x**cj for x in xi))
                    rl.append(tj)
                for i, (w1, w2) in enumerate(wx):
                    sw = [(w1*x1+w2*x2)/rl[j] for j, (x1, x2) in enumerate(zip(rxr, rxi))]
                    nkx, bins = histogram(sw, bins=100)
                    sxm = 0.
                    for j in range(5):
                        if nkx[j]==0: continue
                        if nkx[j+1]==0:
                            sxm = bins[j+1]
                            continue
                        rj1 = nkx[j]/float(nkx[j+1])
                        rj2 = nkx[j+2]/float(nkx[j+1])
                        if rj1>1 and rj2>1.:
                            sxm = bins[j+1]
                            break
                    #swm = min(sw)
                    sw = [x for x in sw if x>=sxm]
                    shape, loc, scale = fitweibull(sw)
                    
                    s2p.append(((s2[i]-loc)/scale)**shape*h)
                    
                snx.append((spid, spdta, s1, s2, s2p, s3, s3p))

                # .. .. write results to file
                fwrite.write('SPECTRUM: %s|%s|%s|%s|%d\n'%(spid, spdta, seq, mod, c))
                fwrite.write('%.6f\n'%s1)
                fwrite.write('|'.join('%.6f'%x for x in s2))
                fwrite.write('\n')
                fwrite.write('|'.join('%.6f'%x for x in s2p))
                fwrite.write('\n')
                if s3 is not None:
                    fwrite.write('%.6f\n'%s3)
                    fwrite.write('%.6f\n'%s3p)
                else:
                    fwrite.write('None\n')
                    fwrite.write('None\n')
                    s3, s3p = 0., 0.
                print((k, '%d/%d'%(kk+1, nj), spid, spdta, '%.4f||%.4f||%.4f'%(s1, max(s2), s3)))

    fwrite.close()


def extractMS1(mzmlfile):
    """
    Extract extracted ion chromatogram (XIC)
    """
    MS1 = []
    # iterate to all children in element and its children until
    # the value of itemname is reached
    def getdeepchild(element, itemname):
        for child in element:
            items = list(child.items())
            if items:
                items = dict(items)
                if 'name' in list(items.keys()) and items['name']==itemname:
                    return items['value']
            if getdeepchild(child, itemname):
                return getdeepchild(child, itemname)

    # read from xml data
    for event, element in et.iterparse(mzmlfile):
        msj = {}
        if event == 'end' and element.tag.endswith('spectrum'):
            # cycle and expriment of current mass spectrum
            spectruminfo = dict(list(element.items()))
            defaultArrayLength = int(spectruminfo['defaultArrayLength'])

            # retention time
            time = float(getdeepchild(element, 'scan start time'))
            
            # get tandem mass spectrum
            ms, mslevel = {}, 1
            for child in element:
                if child.tag.endswith('precursorList'):
                    mslevel = 2
                    break
                elif child.tag.endswith('binaryDataArrayList'):
                    for bdarray in child:
                        if bdarray.tag.endswith('binaryDataArray'):
                            for binaryInfo in bdarray:
                                if binaryInfo.tag.endswith('cvParam'):
                                    binfo = dict(list(binaryInfo.items()))
                                    if binfo['name'] == 'm/z array':
                                        mskey = 'mz'
                                        break
                                    elif binfo['name'] == 'intensity array':
                                        mskey = 'intensity'
                                        break
                            for binaryInfo in bdarray:
                                if binaryInfo.tag.endswith('binary'):
                                    ms[mskey] = decodebinary(binaryInfo.text,
                                                              defaultArrayLength)
            element.clear()

            # ignore ms level >= 2
            if mslevel == 2: continue

            # remove intensity of 0
            mz, intensity = [], []
            for i in range(defaultArrayLength):
                if ms['intensity'][i]>0:
                    mz.append(ms['mz'][i])
                    intensity.append(ms['intensity'][i])
            msj['mz'], msj['intensity'], msj['rt'] = tuple(mz), tuple(intensity), time
            msj['info'] = spectruminfo
            MS1.append(msj)

    return MS1


def decoyMatchNonNitro(seq, mod, charge, nitrosp, nnitrosp, tol=0.2):
    """
    Matching decoys for evaluating the p value of matches between
    mass spectrum of nitrated peptide and its non-nitrated analogues
    """
    nd = nitrodeterm.NitroDetermine(seq, mod, charge)
    ions = nd.annotations(nitrosp, tol)
    mk1 = max(x[1] for i, x in enumerate(nitrosp)
            if not any(xj.startswith('p') for xj in ions[i][1].split(',')))
    spx = [[x[0], x[1]/mk1] for i,x in enumerate(nitrosp) if ions[i][1]]

    # non-nitrated peptides
    modc = ';'.join(x for x in mod.split(';') if not 'Nitro' in x)
    nd_n = nitrodeterm.NitroDetermine(seq, modc, charge)
    ions_n = nd_n.annotations(nnitrosp, tol)
    mk2 = max(x[1] for i, x in enumerate(nnitrosp)
            if not any(xj.startswith('p') for xj in ions_n[i][1].split(',')))
    spt = [[x[0], x[1]/mk2] for i,x in enumerate(nnitrosp) if ions_n[i][1]]

    # get matched fragments
    m1, m2 = [], []
    for i, (mz1, ion1) in enumerate(ions):
        if not ion1: continue
        nx1 = set([x.split('/')[0] for x in ion1.split(',')])
        for j, (mz2, ion2) in enumerate(ions_n):
            if not ion2 or j in m2: continue
            nx2 = [x.split('/')[0] for x in ion2.split(',')]
            if nx1&set(nx2):
                m2.append(j)
                m1.append(i)
                break

    # retain all annotated fragments and corrected m/z of matched fragment
    spt, ions_nx = [], []
    for i, x in enumerate(nnitrosp):
        if not ions_n[i][1]: continue
        if i in m2:
            j = m1[m2.index(i)]
            spt.append([nitrosp[j][0], x[1]/mk2])
            nx = [x for x in ions_n[i][1].split(',')
                  if any(x.split('/')[0] in xj for xj in ions[j][1].split(','))]
            ions_nx.append(','.join(nx))
        else:
            spt.append([x[0], x[1]/mk2])
            ions_nx.append(ions_n[i][1])

    # shuffle m/z of fragments according to decoy sequences
    rseq, rmods = randomseq(seq, mod, 0.05)
    sr = []
    for i, sj in enumerate(rseq):
        sqms = calpepmass(sj, rmods[i])
        ionsr = allions(sqms, sj, rmods[i], charge)
        spr, ax = [], []
        for mzr, ionr in ionsr:
            for j, x in enumerate(ions_nx):
                if ionr in x and j not in ax:
                    ax.append(j)
                    ek = min(float(xj.split('/')[1]) for xj in x.split(',') if ionr in xj)
                    spr.append((mzr-ek, spt[j][1]))
                    break
        mr1, mr2 = [], []
        for j, x in enumerate(spr):
            for k, xj in enumerate(spx):
                if abs(x[0]-xj[0])<=tol:
                    mr1.append(j)
                    mr2.append(k)
                    break
        vx = []
        for j, x in enumerate(spr):
            if j in mr1:
                vx.append((spr[j][1], spx[mr2[mr1.index(j)]][1]))
            else:
                vx.append((spr[j][1], 0.))
        for j,x in enumerate(spx):
            if j not in mr2:
                vx.append((0., spx[j][1]))
        sr.append(sum(x1*x2 for x1, x2 in vx)/sqrt(sum(x1**2 for x1, _ in vx)*sum(x2**2 for _, x2 in vx)))

    return sr


def mgf2dta(mgffile, outdir, dtaprefix=None):
    """
    convert mgf peak list to dta peaks
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

##    if not dtaprefix:
##        dtaprefix = '.'.join(mgffile.split('.')[:-1])

    # read spectra from mgf file and store them into dta files
    read_spectrum, summaries = False, []
    with open(mgffile,'r') as f:
        for line in f:
            if 'PEPMASS' in line:
                pepmass = line.rstrip().split('=')[1]
            if 'RTINSECONDS' in line:
                rt = line.rstrip().split('=')[1]
            if 'TITLE' in line:
                spstr = []
                dtaname = '%s.dta'%(line.rstrip().split()[0].split(':')[1])
                read_spectrum = True
            elif read_spectrum and not 'END IONS' in line:
                if not '=' in line:
                    intk = float(line.rstrip().split()[1])
                    if intk>0:
                        spstr.append('%s\n'%('\t'.join(line.rstrip().split()[:2])))
            elif read_spectrum and 'END IONS' in line:
                spstr.insert(0,'%s\t1\n'%pepmass)
                with open(r'%s/%s'%(outdir, dtaname),'w') as fx:
                    fx.writelines(spstr)
                read_spectrum = False
                summaries.append('%s\t%s\t%s\n'%(dtaname, pepmass, rt))

    # write summaris to summary file
    summaries.insert(0,'%s\t%s\t%s\n'%('dta File Name', 'Precursor m/z', 'Retention Time (sec)'))
    with open(r'%s/%s.summary.txt'%(outdir, dtaprefix),'w') as f:
        f.writelines(summaries)


def getvariables(seq, mods, charge, ions, spectrum, modtype, tol):
    """
    Generate variables for input mass spectrum according to the
    assigned sequence
    """
    # intensity of base peak
    m = max(x[1] for x in spectrum)
    ix20 = set([i for i,x in enumerate(spectrum) if x[1]>=m*0.2 and x[0]>=300])
    # mz and intensities
    spint = [x[1] for x in spectrum]
    mz = [x[0] for x in spectrum]
    # length of peptide
    l = len(seq)
    #
    if isinstance(modtype, str):
        modionk = {'b': min(k for _, k, x in mods if x==modtype),
                   'y': min(l-k+1 for _, k, x in mods if x==modtype)}
    else:
        modionk = {'b': min(k for _, k, x in mods if x in modtype),
                   'y': min(l-k+1 for _, k, x in mods if x in modtype)}
    # number of peaks
    npeaks = float(len(spectrum))
    # ions
    seqions = [ionj for ionj in list(ions.keys()) if ionj[0] in 'yb' and not '-' in ionj]
    # ..
    iix = set([vj[0] for vj in list(ions.values())])
    ionfrac = len(iix)/npeaks
    ionfracint = sum(spint[k] for k in iix)/sum(spint)
    # .. fraction of peaks higher than 20%
    bix = set([ions[ky][0] for ky in seqions if ions[ky][0] in ix20])
    sifrac20 = len(bix)/float(len(ix20)) if ix20 else 0
    # .. mod ion intensity
    modsum = sum(spint[ions[ky][0]] for ky in list(ions.keys())
                 if (ky[0]=='y' and not '-' in ky and ions[ky][1]>=modionk['y'])
                 or (ky[0]=='b' and not '-' in ky and ions[ky][1]>=modionk['b']))

    # sequence coverage
    lgseq, nseq, nmodseq, ionx = 0, 0, 0, {'b':-1, 'y':-1}
    for cj in range(charge):
        cstr = '[+]' if cj==0 else '[%d+]'%(cj+1)
        nj, nmk = 0, 0
        for iontype in 'yb':
            kx = sorted([int(x.split('[')[0][1:])
                         for x in seqions if x[0]==iontype and cstr in x])
            nk = len(kx)
            nmk += nk-bisect_left(kx, modionk[iontype])
            nj += nk
            if nk>ionx[iontype]: ionx[iontype]=nk
            r, _ = longestsequence(kx)
            if r>lgseq: lgseq = r
        if nj>nseq: nseq = nj
        if nmk>nmodseq: nmodseq=nmk
    sc = lgseq/float(l)
    fb = ionx['b']/float(l)
    fy = ionx['y']/float(l)

    # ion score
    nbins = round((max(mz)-min(mz))/tol)
    mp, mp2 = 0., 0.
    if nbins>0:
        p = npeaks/float(nbins)
        mp = binomialprob(p, 2*(l-1), nseq)
        mp2 = binomialprob(p, l-1, nmodseq)
    mp /= sqrt(l)
    mp2 /= sqrt(l)

    return [ionfrac, ionfracint, sifrac20, sc, fb, fy, ionx['b'], ionx['y'], mp, mp2, modsum]


def getvariablesfromarray(seq, mods, charge, ions, spectrum, modtype, tol):
    """
    Generate variables for input mass spectrum according to the
    assigned sequence
    """
    # intensity of base peak
    m = np.amax(spectrum[:,1])
    ix20, = np.where((spectrum[:,1]>=m*0.2)&(spectrum[:,0]>=300))
    # mz and intensities
    spint, mz = spectrum[:,1], spectrum[:,0]
    # length of peptide
    l = len(seq)
    #
    modionk = {'b': min(k for _, k, x in mods if x==modtype),
               'y': min(l-k+1 for _, k, x in mods if x==modtype)}
    # number of peaks
    npeaks = float(len(spectrum))
    # ions
    seqions = [ionj for ionj in list(ions.keys()) if ionj[0] in 'yb' and not '-' in ionj]
    # ..
    iix = set([vj[0] for vj in list(ions.values())])
    ionfrac = len(iix)/npeaks
    ionfracint = np.sum(spint[iix])/sum(spint)
    # .. fraction of peaks higher than 20%
    bix = set([ions[ky][0] for ky in seqions if ions[ky][0] in ix20])
    sifrac20 = len(bix)/float(len(ix20)) if ix20 else 0
    # .. mod ion intensity
    modsum = sum(spint[ions[ky][0]] for ky in list(ions.keys())
                 if (ky[0]=='y' and not '-' in ky and ions[ky][1]>=modionk['y'])
                 or (ky[0]=='b' and not '-' in ky and ions[ky][1]>=modionk['b']))

    # sequence coverage
    lgseq, nseq, nmodseq, ionx = 0, 0, 0, {'b':-1, 'y':-1}
    for cj in range(charge):
        cstr = '[+]' if cj==0 else '[%d+]'%(cj+1)
        nj, nmk = 0, 0
        for iontype in 'yb':
            kx = sorted([int(x.split('[')[0][1:])
                         for x in seqions if x[0]==iontype and cstr in x])
            nk = len(kx)
            nmk += nk-bisect_left(kx, modionk[iontype])
            nj += nk
            if nk>ionx[iontype]: ionx[iontype]=nk
            r, _ = longestsequence(kx)
            if r>lgseq: lgseq = r
        if nj>nseq: nseq = nj
        if nmk>nmodseq: nmodseq=nmk
    sc = lgseq/float(l)
    fb = ionx['b']/float(l)
    fy = ionx['y']/float(l)

    # ion score
    nbins = round((max(mz)-min(mz))/tol)
    mp, mp2 = 0., 0.
    if nbins>0:
        p = npeaks/float(nbins)
        mp = binomialprob(p, 2*(l-1), nseq)
        mp2 = binomialprob(p, l-1, nmodseq)
    mp /= sqrt(l)
    mp2 /= sqrt(l)

    return [ionfrac, ionfracint, sifrac20, sc, fb, fy, ionx['b'], ionx['y'], mp, mp2, modsum]


def getionscore(seq, mods, charge, ions, spectrum, tol):
    """
    Generate variables for input mass spectrum according to the
    assigned sequence
    """
    # intensity of base peak
    mz = [x[0] for x in spectrum]
    npeaks = float(len(spectrum))
    # length of peptide
    l = len(seq)
    # ions
    seqions = [ionj for ionj in list(ions.keys()) if ionj[0] in 'yb' and not '-' in ionj]

    # sequence coverage
    nseq = 0
    for cj in range(charge):
        cstr = '[+]' if cj==0 else '[%d+]'%(cj+1)
        nj = 0
        for iontype in 'yb':
            kx = sorted([int(x.split('[')[0][1:])
                         for x in seqions if x[0]==iontype and cstr in x])
            nk = len(kx)
            nj += nk
        if nj>nseq: nseq = nj

    # ion score
    nbins = round((max(mz)-min(mz))/tol)
    mp = 0.
    if nbins>0:
        p = npeaks/float(nbins)
        mp = binomialprob(p, 2*(l-1), nseq)
    mp /= sqrt(l)

    return mp


def getionscore2(seq, charge, ions, spectrum, tol):
    """
    Generate variables for input mass spectrum according to the
    assigned sequence
    """
    if not ions: return 0.
    npeaks, mzrange = len(spectrum), spectrum[-1][0]-spectrum[0][0]
    # length of peptide
    l = len(seq)

    # sequence coverage
    cx = ['[+]' if cj==0 else '[%d+]'%(cj+1) for cj in range(charge)]
    nseq = max(sum(v[0] in 'yb' and ck in v for _, v, _ in ions) for ck in cx)

    # ion score
    nbins = round(mzrange/tol)
    mp = 0.
    if nbins>0:
        p = npeaks/float(nbins)
        mp = binomialprob(p, 2*(l-1), nseq)
    mp /= sqrt(l)

    return mp


def getvariables_add(seq, mod, charge, ions, spectrum, mods, modtype, tol):
    """
    Generate variables for input mass spectrum according to the
    assigned sequence
    """
##    nd = nitrodeterm.NitroDetermine(seq, mod, charge)
##    ions = nd.annotations(spectrum, tol)
    # intensity of base peak
    m = max(x[1] for x in spectrum)
    # length of peptide
    l = len(seq)
    modionk = {'b': min(k-1 for _, k, x in mods if x==modtype),
               'y': min(l-k for _, k, x in mods if x==modtype)}
    # ions
    seqions = []
    for i, (_, ion) in enumerate(ions):
        if ion:
            nx = []
            for x in ion.split(','):
                if x[0] in 'yb' and not '-' in x:
                    nx.append(x.split('/')[0])
            # .. with intensity higher than 20% base peak
            if nx: seqions += nx

    nmodseq = 0
    for cj in range(charge):
        cstr = '[+]' if cj==0 else '[%d+]'%(cj+1)
        nj, nmk = 0, 0
        for iontype in 'yb':
            kx = [(x[0], int(x.split('[')[0][1:]))
                  for x in seqions if x[0]==iontype and cstr in x]
            nmk += sum(nk>=modionk[iontype] for tj, nk in kx)
        if nmk>nmodseq: nmodseq=nmk

    # ion score
    mz = [x[0] for x in spectrum]
    nbins = round((max(mz)-min(mz))/tol)
    mp2 = 0.
    if nbins>0:
        p = len(spectrum)/float(nbins)
        mp2 = binomialprob(p, l-1, nmodseq)
    mp2 /= sqrt(len(seq))

    return mp2


##def spectrumComparisonVariables(seq, mod, c, ions, modms, ionsn, nonmodms, mObj, tol=0.2):
##    """ Get variables for spectral comparison """
##    # local function for getting maximum values and corresponding index
##    def maxima(ix, ms):
##        intx = ms[ix][:,1]
##        mint = max(intx)
##        return ix[intx==mint][0], mint
##
##    # similarity scores only considering annotated fragments 
##    s1, _, mix = matchNonNitro2(ions, modms, ionsn, nonmodms)
##    # similarity score with all fragments
##    n1, n2 = len(modms), len(nonmodms)
##    m1x, m2x = [x[0] for x in mix], [x[1] for x in mix]
##    rm1x = [i for i in xrange(n1) if i not in m1x]
##    rm2x = [i for i in xrange(n2) if i not in m2x]
##    # .. get objective modification mass
##    modmb = [np.round(mObj/(i+1)/tol) for i in xrange(c)]
##    # process spectrum
##    rmodms = np.array([modms[i] for i in rm1x])
##    rnmodms = np.array([nonmodms[i] for i in rm2x])
##    mix2, mix1, dx1 = [], [], []
##    if rmodms.size>0 and rnmodms.size>0:
##        mz1, mz2 = np.round(rmodms[:,0]/tol), np.round(rnmodms[:,0]/tol)
##        # get matched peaks
##        for i, mzj in enumerate(mz1):
##            if i in dx1: continue
##            ix1, = np.where(mz1==mzj)
##            ixk, mt1 = maxima(ix1, rmodms)
##            mix1.append(ixk)
##            if len(ix1)>1:
##                dx1 += list(j for j in ix1 if j!=ixk)
##            # find peak from mass spectrum of non-modified analogue
##            ix, = np.where(mz2==mzj)
##            if len(ix)>0:
##                itx = rnmodms[ix][:,1]
##                ixj, mt = maxima(ix, rnmodms)
##                m1x.append(rm1x[ixk])
##                m2x.append(rm2x[ixj])
##                mix2.append(ixj)
##            else:           # match potential modified peaks
##                ix = []
##                for cj in xrange(c):
##                    ixj, = np.where(mz2==mzj-modmb[cj])
##                    ix += list(ixj)
##                ix = np.array(list(set(ix)-set(mix2)))
##                if len(ix)>0:
##                    ixj, mt = maxima(ix, rnmodms)
##                    m1x.append(rm1x[ixk])
##                    m2x.append(rm2x[ixj])
##                    mix2.append(ixj)
##                else:
##                    m1x.append(rm1x[ixk])
##                    m2x.append(-1)
##    for i,x in enumerate(rnmodms):
##        if i not in mix2:
##            m1x.append(-1)
##            m2x.append(rm2x[i])
##    for i,x in enumerate(rmodms):
##        if i not in dx1 and i not in mix1:
##            m2x.append(-1)
##            m1x.append(rm1x[i])
####    s2 = pms.similarity([sqrt(modms[i][1]) if i>=0 else 0. for i in m1x],
####                        [sqrt(nonmodms[i][1]) if i>=0 else 0. for i in m2x])
##    # match top 20 peaks
##    nsel = min(20, min(n1, n2))
##    six1 = sorted(xrange(n1), key=lambda k: modms[k][1], reverse=True)[:nsel]
##    six2 = sorted(xrange(n2), key=lambda k: nonmodms[k][1], reverse=True)[:nsel]
##    m1t, m2t = [], []
##    for i, j in zip(m1x, m2x):
##        if i in six1 and j in six2:
##            m1t.append(sqrt(modms[i][1]))
##            m2t.append(sqrt(nonmodms[j][1]))
##        elif i in six1 and j not in six2:
##            m1t.append(sqrt(modms[i][1]))
##            m2t.append(0.)
##        else:
##            m1t.append(0.)
##            m2t.append(sqrt(nonmodms[j][1]))
##    s3 = pms.similarity(m1t, m2t)
##
##    # sequence coverage
####    nseq1, nseq2 = 0, 0
####    for cj in xrange(c):
####        cstr = '[+]' if cj==0 else '[%d+]'%(cj+1)
####        nj1, nj2 = 0, 0
####        for iontype in 'yb':
####            nj1 += sum(any(xj[0]==iontype and not '-' in xj and cstr in xj
####                      for xj in x.split(',')) for _, x in ions if x)
####            nj2 += sum(any(xj[0]==iontype and not '-' in xj and cstr in xj
####                      for xj in x.split(',')) for _, x in ionsn if x)
####        if nj1>nseq1: nseq1 = nj1
####        if nj2>nseq2: nseq2 = nj2
##
##    # ion score
####    mz = [x[1] for x in modms]
####    nbins1 = round((max(mz)-min(mz))/tol)
####    mz = [x[1] for x in nonmodms]
####    nbins2 = round((max(mz)-min(mz))/tol)
####    p1 = n1/float(nbins1)
####    p2 = n2/float(nbins2)
####    mp1 = binomialprob(p1, 2*(len(seq)-1), nseq1)
####    mp2 = binomialprob(p2, 2*(len(seq)-1), nseq2)
####    s4 = abs(mp1-mp2)/max(mp1, mp2)
##    # sequence coverage
####    isby2 = isby
####    lgseq1, lgseq2 = 0, 0
####    for cj in xrange(c):
####        cstr = '[+]' if cj==0 else '[%d+]'%(cj+1)
####        for iontype in 'yb':
####            kx1 = []
####            for _, ionj in ions1:
####                if not ionj: continue
####                kx1 += [int(x.split('[')[0][1:]) for x in ionj.split(',')
####                         if x[0]==iontype and cstr in x and isby2(x)]
####            r = len(kx1)
####            if r>lgseq1: lgseq1 = r
####
####            kx2 = []
####            for _, ionj in ions2:
####                if not ionj: continue
####                kx2 += [int(x.split('[')[0][1:]) for x in ionj.split(',')
####                         if x[0]==iontype and cstr in x and isby2(x)]
####            r = len(kx2)
####            if r>lgseq2: lgseq2 = r
####    sc = min(lgseq1, lgseq2)/float(len(seq))
##    
##    return [s1, s3]


def checkspectrum(ms, normalize=True):
    """
    check the spectrum and normalize to the base peak
    return n, sorted spectrum, and sort indices
    """
    n = len(ms)
    six = sorted(list(range(n)), key=lambda k: ms[k][0])
    m = max(x[1] for x in ms)
    return n, [[ms[i][0], ms[i][1]/m] for i in six], six


def checkmatchtype(matchedion, modix):
    """
    Check the type of matches and identify whether the matched
    ions are modified
    type of matches:
    's': sequential ions
    'p': precursor ions
    'n': neutral losses
    'i': immonium ions
    't': internal fragments
    """
    if matchedion.startswith('p'):
        return 'p', True
    if matchedion.startswith('imm'):
        if '*' in matchedion:
            return 'i', True
        return 'i', False
    if not '[' in matchedion:
        return 't', False
    # y, b, a and neutrals
    kb, ky = modix['b'], modix['y']
    st = '-' if '-' in matchedion else '['
    iix = int(matchedion.split(st)[0][1:])
    if (matchedion[0] in 'ba' and iix>=kb) or (matchedion[0]=='y' and iix>=ky):
        if '-' in matchedion:
            return 'n', True
        return 's', True
    if '-' in matchedion:
        return 'n', False
    return 's', False


def kendalltau(x, y):
    """
    Kendall rank correlation coefficient
    """
    n = len(x)
    if n<=1: return 0.
    six = sorted(list(range(n)), key=lambda k: x[k])
    x = sorted(x)
    y = [y[i] for i in six]
    nc, nd, nxtie, nytie = 0, 0, 0, 0
    # number of x ties
    nxtie = sum(x[i]==x[i+1] for i in range(n-1))
    # number of concordants and disconcordants
    for y1, y2 in combinations(y, 2):
        if y2>y1:
            nc += 1
        elif y2<y1:
            nd += 1
        else:
            nytie += 1
    n0 = n*(n-1)/2.
    n1 = 0 if nxtie<2 else nxtie*(nxtie-1)/2.
    n2 = 0 if nytie<2 else nytie*(nytie-1)/2.
    
    try:
        tau = min(1, (nc-nd)/sqrt((n0-n1)*(n0-n2)))
    except:
        tau = 0.
    
    return tau

def relativeSimScore(ions, modms, ionsn, nonmodms, seq, mods, modtype, tol=0.2):
    """
    Calculate similairty score using relative intensities, then consider
    matches of modified and non-modified fragments separately.
    """
    # mass of target modification
    mm = [x for x, _, mt in mods if mt==modtype][0]
    # check mass spectrum to make m/z in ascending order
##    npeaks1, msp, six = checkspectrum(modms)
##    mions = [ions[i] for i in six]
##    npeaks2, nsp, six = checkspectrum(nonmodms)
##    nions = [ionsn[i] for i in six]
    # get indices of matched fragments
    msp, nsp, mions, nions = modms, nonmodms, ions, ionsn
    mix = matchNonMod(msp, mions, nsp, nions)
    # normalize the peaks
##    m1 = max(x[1] for i,x in enumerate(msp) if mions[i][1])
##    m2 = max(x[1] for i,x in enumerate(nsp) if nions[i][1])
##    mint = [x[1]/m1 for x in msp]
##    nint = [x[1]/m2 for x in nsp]
    mint, nint = [x[1] for x in msp], [x[1] for x in nsp]
    # get the indices of fragments that are modified
    nr = len(seq)
    kb = min(k for _, k, x in mods if x==modtype)
    ky = min(nr-k+1 for _, k, x in mods if x==modtype)
    # calculate similarity scores by setting weights to matches
    # belonging to modified parts and non-modified parts
    mmod, mnod = [], []
    for i,j in mix:
        # .. relative intensities
        mcint = max(mint[i], nint[j])
        rint = abs(mint[i]-nint[j])/mcint
        # .. absolute m/z differences
        dm = abs(msp[i][0]-nsp[j][0])
        # .. is modified?
        nxs1 = set([xj.split('/')[0] for xj in mions[i][1].split(',')])
        nxs2 = set([xj.split('/')[0] for xj in nions[j][1].split(',')])
        cx = []
        for ionj in nxs1&nxs2:
            if not '+' in ionj or '[+' in ionj:
                cx.append(1.)
            else:
                cx.append(float(ionj.split('[')[1][0]) )
        # .. if t is True, yes
##        t = False
##        rintx = (1.-rint)/2 if rint>0.3 or mcint<0.2 else 1.-rint
##        rintx = (1.-rint)/2 if rint>0.3 else 1.-rint
        if dm>min(abs(dm-mm/cj) for cj in set(cx)):
            # .. .. reconstructed intensity
            mmod.append(rint)
##            t = True
        else:
            mnod.append(rint)
##        minfo.append((nsp[j][0], nxs1&nxs2, rint, t))

    return mmod, mnod


def spectrumComparisonVariables(ions, modms, ionsn, nonmodms, tol=0.2):
    """
    Get variables between mass spectra of modifications and non-modifications
    """
    # reorder the mass spectra according to the m/z values in ascending order
    npeaks1, msp, six = checkspectrum(modms)
    mions = [ions[i] for i in six]
    npeaks2, nsp, six = checkspectrum(nonmodms)
    nions = [ionsn[i] for i in six]
    # get matched fragment ion indices
    mix = matchNonMod(msp, mions, nsp, nions)
    npeaks = npeaks1 if npeaks1>npeaks2 else npeaks2
    nmatches = float(len(mix))
    if nmatches<=1:
        return 0, 0, 0, 0, 0, 0

    # ranks of sorted intensities
    rk1 = sorted(list(range(npeaks1)), key=lambda k: msp[k][1], reverse=True)
    rk2 = sorted(list(range(npeaks2)), key=lambda k: nsp[k][1], reverse=True)
    # get ranks of matched fragments
    rk1m = [rk1.index(i)+1 for i,_ in mix]
    rk2m = [rk2.index(i)+1 for _,i in mix]
    # .. stats of ranks
    srk1, srk2 = sum(rk1m), sum(rk2m)
    srk1s, srk2s = sum(x**2 for x in rk1m), sum(x**2 for x in rk2m)
    srkm = sum(x*y for x,y in zip(rk1m, rk2m))

    # get the intensities of matched fragments
    mints = [(msp[i][1], nsp[j][1]) for i,j in mix]
    # .. get unmatched peaks
    u1x = sorted(set(range(npeaks1))-set([i for i,_ in mix]))
    u2x = sorted(set(range(npeaks2))-set([i for _,i in mix]))

    # Kendall rank correlation coefficient
    krtau = kendalltau([x[0] for x in mints], [x[1] for x in mints])
    krtau = max(krtau, 0)
    if len(mints)>=6:
        sigma2 = 2*(2*nmatches+5)/(9.*nmatches*(nmatches-1.))
        krprob = lognormcdf(abs(krtau)/sqrt(sigma2))
    else:
        krprob = 0.

    # spearman's rank coefficients
    var1 = (srk1s-srk1**2/nmatches)/nmatches
    var2 = (srk2s-srk2**2/nmatches)/nmatches
    cov = (srkm-(srk1*srk2)/nmatches)/nmatches
    if var1==0 or var2==0:
        rs=0.
    else:
        rs = cov/sqrt(var1*var2)
        rs = max(rs, 0.)

    # adjusted dot product penalized by intensities of unmatched fragments
    mdp = sum(x*y for x,y in mints)
    udp = sum(msp[i][1]**2 for i in u1x)+sum(nsp[i][1]**2 for i in u2x)
    adjdp = mdp/(mdp+udp)

    # dot product of ranks
    npeaks += 1
    h = nmatches*npeaks**2
    rkdp = (h-npeaks*(srk1+srk2)+srkm)/\
           sqrt((h-2*npeaks*srk1+srk1s)*(h-2*npeaks*srk2+srk2s))

    # hypergeometric distribution
    nbins = int((msp[-1][0]-msp[0][0])/(2*tol))
    #print nbins, npeaks2, nmatches
    gprob = geometricprob(nbins, npeaks2, int(nmatches))

    return krtau, krprob, rs, adjdp, rkdp, gprob


def similarityscoreAdj(ions, modms, ionsn, nonmodms, tol=0.2):
    """
    Get variables between mass spectra of modifications and non-modifications
    """
    # reorder the mass spectra according to the m/z values in ascending order
    npeaks1 = len(modms)
    npeaks2 = len(nonmodms)
    # get matched fragment ion indices
    mix = matchNonMod(modms, ions, nonmodms, ionsn)
    if not mix: return 0.

    # get the intensities of matched fragments
    mints = [(modms[i][1], nonmodms[j][1]) for i,j in mix]
    # .. get unmatched peaks
    u1x = set(range(npeaks1))-set([i for i,_ in mix])
    u2x = set(range(npeaks2))-set([i for _,i in mix])

    # adjusted dot product penalized by intensities of unmatched fragments
    mdp = sum(x*y for x,y in mints)
    udp = sum(modms[i][1]**2 for i in u1x)+sum(nonmodms[i][1]**2 for i in u2x)
    adjdp = mdp/(mdp+udp)

    return adjdp


def ldacv(X, y, n, seltype='f'):
    """
    Train LDA using cross validation
    """
    lda, normpdf = LDA().fit, stats.norm().pdf
    cx = set(y)
    nx, nk = [sum(y==cj) for cj in cx], len(cx)
    nl = max(int(n/2.),1) if seltype=='n' else max(int(max(nx)/float(n))+1,1)
    if nl==1: seltype='n'
    nt = sum(nx)
    sl = np.array([True]*nt)
    err = 0
    s, prob = np.zeros(nt), dict((int(cj), np.zeros(nt)) for cj in cx)
    bprob = np.zeros(nt)
    if seltype=='n':
        # leave one out cross-validation
        nf = int(nt/n)
        for i in range(nf):
            sl[i*nl:(i+1)*nl] = False
            Xtr, ytr = X[sl].copy(), y[sl].copy()
            Xte = X[i].reshape(1,-1)
            yte = lda(Xtr, ytr).predict(Xte)
            err += sum(y[i*nl:(i+1)*nl]!=yte)
            sl[i*nl:(i+1)*nl] = True
    else:
        # n fold cross-validation
        if nl*(n-1)>min(nx): nl -=1
        for i in range(n):
            kx = []
            if i==n-1:
                _ = [kx.extend(sum(nx[:j])+k for k in range(i*nl, nx[j]))
                     for j in range(nk)]
            else:
                _ = [kx.extend(sum(nx[:j])+k for k in range(i*nl, min((i+1)*nl, nx[j])))
                     for j in range(nk)]
            kx = np.array(kx)
            sl[kx] = False
            model = lda(X[sl], y[sl])
            yte = model.predict(X[kx])
            err += sum(y[kx]!=yte)
            sl[kx] = True
            # .. calculate scores and probabilities
            sc = model.decision_function(X)
            s[kx] = sc[kx]
            yc = model.predict(X)

            # .. .. calculate calibrated probability
            statx = [(np.mean(sc[yc==cj]), np.std(sc[yc==cj])) for cj in cx]
            pbt = np.zeros(len(kx))
            for j, cj in enumerate(cx):
                scj = sc[yc==cj]
                prob[cj][kx] = normpdf((sc[kx]-statx[j][0])/statx[j][1])/\
                         sum(normpdf((sc[kx]-mk)/dk) for mk, dk in statx)
                pbt += prob[cj][kx]
            bprob[kx] = prob[1][kx]/pbt
                
    return err/float(nt), s, prob, bprob
    

def adaptiveClassification(X, varidx=0, F=0.1, p=0.3, top=False):
    """ """
    # preallocation functions
    norm, concatenate = np.linalg.norm, np.concatenate
    niters = 100
    # scaling by median
    mx = np.median(X, axis=0)
    Xm = X/mx
    X = Xm
    n = len(X)
    # K means clustering using sequential ion fraction above 20% base peak,
    # ion coverage and modification scores
    varix = list(range(Xm.shape[1]))
    y_pred = KMeans(n_clusters=2, random_state=1).fit_predict(Xm[:,varix])
    center1 = np.mean(Xm[y_pred==0], axis=0)
    center2 = np.mean(Xm[y_pred==1], axis=0)
    
    # select potentially true and false identifications, and make sure
    # Xm1 and y1 are positive sets
    targetvarix = varidx
    fls, tr = (1,0) if center1[targetvarix]>center2[targetvarix] else (0,1)
    X1, X2 = X[y_pred==tr], X[y_pred==fls]
    center0 = np.mean(Xm[y_pred==tr], axis=0)
    Xmn = Xm[y_pred==fls]

    # initialization of positives and negatives for classification
    nsel2 = nsel1 = int(X.shape[0]*F)
    dn = np.sum((Xmn[:,varix]-center0[varix])**2, axis=1)
    negsix = np.argsort(dn)[::-1]
    nn = min(nsel2, negsix.shape[0])
    Xn, yn = X2[negsix[:nn]], np.zeros(nn)
    posix = np.argsort(X1[:,targetvarix])[::-1]
    pn = min(nsel1, posix.shape[0])
    Xp, yp = X1[posix[:pn]], np.ones(pn)
    # select the number of samples for testing in cross-validation
    # and type of cross-validation
    if min(nsel1, nsel2) <= 20:
        nf, scv = 1, 'n'
    elif min(nsel1, nsel2) <= 30:
        nf, scv = 5, 'f' # five fold
    else:
        nf, scv = 10, 'f' # ten fold
    # Linear Discriminant Analysis
    R, nyp, S, M, se, ex = [], [], [], [], [], []
    ny, c0, cx, itx = [], np.zeros(7), [], []
    for k in range(niters):
        Xx, yx = concatenate((Xp, Xn)), concatenate((yp,yn))
        cfk = LDA().fit(Xx, yx)
        yprob = cfk.predict_proba(X)[:,1]
        c1 = cfk.coef_[0]
        yscore = cfk.decision_function(X)
        #se = ldacv(Xx, yx, nf, seltype=scv)
##        if isinstance(cx, np.ndarray):
##            cx = np.vstack((cx, cfk.coef_[0]))
##        else:
##            cx = cfk.coef_[0]
##        itx.append(cfk.intercept_[0])
##        c1 = cx if cx.ndim==1 else np.mean(cx, axis=0)
##        itj = sum(itx)/len(itx)
##        yscore = np.dot(X, c1)+itj
##        yprob = yscore*-1
##        np.exp(yprob, yprob)
##        yprob += 1
##        np.reciprocal(yprob, yprob)
        e = abs(norm(c1)-norm(c0))/norm(c1)
        ex.append(e)
        if e<=0.01:
                M = cfk
                break
        if top:
            six = np.argsort(yscore)
            Xn, Xp = X[six[:nsel2]], X[six[::-1][:nsel1]]
            yp, yn = np.ones(nsel1), np.zeros(nsel2)
        else:
            sixn = [j for j,x in enumerate(yprob) if x<0.01 and x>0.0001]
            sixp = [j for j,x in enumerate(yprob) if x>=p]
            if len(sixn)<n*0.001 or len(sixp)<n*0.001:
                six = np.argsort(yscore)
                Xn, Xp = X[six[:nsel2]], X[six[::-1][:nsel1]]
            else:
                sixn = sorted(sixn, key=lambda i: yprob[i], reverse=True)
                Xn = np.copy(X[sixn[:min(len(sixn), nsel2)]])
                sixp = sorted(sixp, key=lambda i: yprob[i])
                Xp = np.copy(X[sixp[:min(len(sixp), nsel1)]])
            yn, yp = np.zeros(len(Xn)), np.ones(len(Xp))
            if k==niters-1:
                Xx, yx = concatenate((Xp, Xn)), concatenate((yp,yn))
                cfk = LDA().fit(Xx, yx)
                Xt = np.copy(X)
                yprob = cfk.predict_proba(Xt)[:,1]
                #se = ldacv(Xx, yx, nf, seltype=scv)            
        c0 = c1

    # output final model with maximum
    se = ldacv(Xx, yx, nf, seltype=scv)
    # k = nyp.index(max(nyp))
    R = {'coef': c1,
         'model': cfk,
         'probs': yprob,
         'scores': yscore,
         'normalizer': mx,
         'errs': ex}
##    R = {'model': cfk,
##         'normalizer': mx,
##         'probabilities': yprob,
##         'accuracy': 1.-se,
##         'scores': yscore}
         #'trac': ny}
    
    return R
