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
RESIDUES = 'ACDEFGHIKLMNPQRSTVWY'
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
CONF = {'I01': 92.9, 'I02': 91.6, 'I03': 88.8, 'I04': 88.7, 'I05': 86.2,
        'I06': 89.5, 'I07': 88.2, 'I08': 87.7, 'I09': 89.6, 'I10': 90.3,
        'I11': 93.3, 'I12': 92.5, 'I13': 89.4, 'I14': 89.1, 'I15': 93.1,
        'I16': 92.9, 'I17': 89.5, 'I18': 90.2, 'I19': 90.2, 'N01': 92.7,
        'N02': 92.5, 'N03': 92.7, 'N04': 95.7}
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
    six = sorted(range(n), key=lambda k: mz[k])
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
        sixk = sorted(range(k, j), key=lambda a: ints[a], reverse=True)
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
        sixk = sorted(range(k, j), key=lambda a: ints[a], reverse=True)
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
    for x in allscore:
        k1 = bisect_left(truescore, x)
        k2 = bisect_left(decoyscore, x)
        if nt-k1==0:
            cfdr = 1.
        else:
            cfdr = (nd-k2)/float(nt-k1)
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
               for ky, val in ions.items())
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
                nm += 1

    # .. cumulative probability
    prob_match = binomialprob(p, np, nm)

    return prob_match, prob_match2, S, prob_intensity, mr


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
    byx0 = set(xk for xk in ions.keys() if xk[0] in 'ybp' and not '-' in xk)
    neutrolx0 = set(xk for xk in ions.keys() if xk not in byx0)
    # get the annotated names of non-modified analogues
    byx1 = set(xk for xk in ions_n.keys() if xk[0] in 'ybp' and not '-' in xk)
    neutrolx1 = set(xk for xk in ions_n.keys() if xk not in byx1)

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
    matchixs += [(i, None) for i, _ in ions.values() if i not in ax1]
    matchixs += [(None, i) for i, _ in ions_n.values() if i not in ax2]
    ax1.update(i for i,_ in ions.values())
    ax2.update(i for i,_ in ions_n.values())

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
    for filenum, ppfilename in FILENUM[ix].items():
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
                for fileorder, fileprefix in FILENUM[k].items():
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
            print(i, 'of', ntps, 'is processed from set', rawset)


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
    seqions = [ionj for ionj in ions.keys() if ionj[0] in 'yb' and not '-' in ionj]
    # ..
    iix = set([vj[0] for vj in ions.values()])
    ionfrac = len(iix)/npeaks
    ionfracint = sum(spint[k] for k in iix)/sum(spint)
    # .. fraction of peaks higher than 20%
    bix = set([ions[ky][0] for ky in seqions if ions[ky][0] in ix20])
    sifrac20 = len(bix)/float(len(ix20)) if ix20 else 0
    # .. mod ion intensity
    modsum = sum(spint[ions[ky][0]] for ky in ions.keys()
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
    seqions = [ionj for ionj in ions.keys() if ionj[0] in 'yb' and not '-' in ionj]

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


def checkspectrum(ms, normalize=True):
    """
    check the spectrum and normalize to the base peak
    return n, sorted spectrum, and sort indices
    """
    n = len(ms)
    six = sorted(range(n), key=lambda k: ms[k][0])
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
    six = sorted(range(n), key=lambda k: x[k])
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
    # get indices of matched fragments
    msp, nsp, mions, nions = modms, nonmodms, ions, ionsn
    mix = matchNonMod(msp, mions, nsp, nions)
    # normalize the peaks
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
        if dm>min(abs(dm-mm/cj) for cj in set(cx)):
            # .. .. reconstructed intensity
            mmod.append(rint)
        else:
            mnod.append(rint)

    return mmod, mnod


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
