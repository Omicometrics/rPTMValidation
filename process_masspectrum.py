from math import sqrt, floor, copysign
from collections import Counter

from operator import mul, itemgetter


isotable = []
with open('isodist.txt','r') as __f:
    for __line in __f:
        __sline = [list(float(__xj) for __xj in __x.split('\t'))
                   for __x in __line.rstrip().split(';') if __x]
        isotable.append(__sline)
        

def similarity(vec1, vec2):
    """
    Calculate cosine similarity of vector 1 "vec1" and vector 2
    "vec2".
    """
    if len(vec1) != len(vec2):
        raise ValueError('The lengths of input vectors must be the same.')
    
    def dotproduct(x1, x2):
        return sum(map(mul, x1, x2))
    
    return dotproduct(vec1, vec2)/(sqrt(dotproduct(vec1, vec1)*dotproduct(vec2, vec2)))


def median(x):
    """
    Median value of the list
    """
    sortx, n = sorted(x), len(x)
    if n<=1: return x
    idx = (n - 1) // 2
    return sortx[idx] if (n%2) else (sortx[idx]+sortx[idx+1])/2.


def detectisotopes(spectrum, tol):
    """
    Detect isotopic clusters from mass spectrum.
    """
    # maximum charge for each fragment is set to be 3
    c = [1,2,3]
    # isotopic mass difference
    h = 1.0024
    n = len(spectrum)
    # sort the mass spectrum
    mzx = sorted([x[0] for x in spectrum])
    
    # find potential isotopic clusters
    isocluster_b = []
    for i, x in enumerate(mzx):
        if i>0:
            jx = [j for j,xj in enumerate(mzx[:i]) if xj<x-10.]
            pk = max(jx) if jx else 0
            
        clj, clix, clbkx = [[x]]*len(c), [[i]]*len(c), []
        cy = list(c)
        kx = [j for j, xj in enumerate(mzx) if xj>x and xj<x+10]
        for ij in range(7):
            clk, cx, lx, pix = [], [], [], []
            for k, txj in enumerate(clj):
                ct = txj[-1]
                kxj = [j for j in kx if abs(mzx[j]-ct-h/cy[k])<=tol/cy[k] and mzx[j]>ct]
                for j in kxj:
                    clk.append(txj+[mzx[j]])
                    cx.append(cy[k])
                    lx.append(k)
                    pix.append(clix[k]+[j])
            clbkx += [(set(xj), cy[j]) for j,xj in enumerate(clix) if j not in lx]
            cy, clj, clix = list(cx), list(clk), list(pix)
        if pix: clbkx += [(set(xj), cx[j]) for j,xj in enumerate(pix)]
        
        # remove replicate sets or subsets
        delx = set()
        for j, xj in enumerate(clbkx):
            if j in delx: continue
            for k, xk in enumerate(clbkx):
                if k in delx or k==j: continue
                if xj[0].issubset(xk[0]) and xj[1]==xk[1]:
                    delx.add(j)
        # .. check previous clusters
        if isocluster_b:
            jx = [j for j,xj in enumerate(isocluster_b) if min(xj[0])<pk]
            jk = max(jx) if jx else 0
            delx2 = set()
            for j,xj in enumerate(clbkx):
                if j in delx: continue
                for k, xk in enumerate(isocluster_b[jk:]):
                    if xj[1]==xk[1]:
                        if xj[0].issubset(xk[0]):
                            delx.add(j)
                        elif xk[0].issubset(xj[0]):
                            delx2.add(k+jk)
            if delx2:
                isocluster_b = [xk for k,xk in enumerate(isocluster_b) if not k in delx2]
        isocluster_b += [xj for j,xj in enumerate(clbkx) if j not in delx]

    return isocluster_b


def deisotope(spectrum, tol, iso_table=isotable):
    """
    Remove isotopic distributed fragments
    """
    sign = lambda x: copysign(1, x)
    refmz = [x[0][0] for x in iso_table]
    # sort the mass spectrum with m/z values in ascending order
    mzs = [x[0] for x in spectrum]
    six = sorted(range(len(mzs)), key=lambda k: mzs[k])
    spectrum = [spectrum[i] for i in six]
    # get all potential isotopic clusters
    isoclusters = detectisotopes(spectrum, tol)
    # m/z and intensities
    mzs = sorted(mzs)
    ints = [x[1] for x in spectrum]
    # candidate isotopic distributions
    isocandidates = []
    for x, cj in isoclusters:
        lx, xs = len(x), sorted(list(x))
        mx = []
        for ii, i in enumerate(xs[:lx-1]):
            dmz = [abs(xj-mzs[i]) for xj in refmz]
            j = dmz.index(min(dmz))
            isox = [xj[1] for xj in iso_table[j]]
            if lx>len(isox):
                isox += [0.]*(lx-len(isox))
            for ij, j in enumerate(xs[ii+1:]):
                if sign(ints[j]-ints[xs[ii+ij]])==sign(isox[1]-isox[0]) and \
                   similarity([ints[k] for k in xs[ii:ii+ij+2]], isox[:ij+2])>=0.95:
                    mx.append(set(xs[ii:ii+ij+2]))
                else:
                    break
        if mx:
            delx1, delx2 = [], []
            for j, xj in enumerate(mx):
                if j in delx1: continue
                for k, xk in enumerate(mx):
                    if k in delx1 or j==k: continue
                    if xj.issubset(xk):
                        delx1.append(j)
                for k, xk in enumerate(isocandidates):
                    if k in delx2: continue
                    if xj.issubset(xk):
                        delx1.append(j)
                    elif xk.issubset(xj):
                        delx2.append(k)
            isocandidates = [xk for k,xk in enumerate(isocandidates) if k not in delx2]
            isocandidates += [xj for j,xj in enumerate(mx) if j not in delx1]

    # retain the identified isotopic distribution with highest
    # intensities
    delx, isotemp = set(), list(isocandidates)
    isos = set()
    for x in isotemp: isos.update(x)
    isos2 = sorted(isos, key=lambda k: ints[k], reverse=True)
    delisos = []
    for j in isos2:
        if j in delx: continue
        kx = [k for k, xk in enumerate(isotemp) if j in xk]
        if not kx: continue
        if len(kx)>=2:
            lx = [len(isotemp[k]) for k in kx]
            k = kx[lx.index(max(lx))]
        else:
            k = kx[0]
        delx.update(sorted(list(isotemp[k]))[1:])
    return sorted(list(delx))
    

def denoise(spectrum, METHOD, param):
    """
    Remove noise of mass spectrum by counting the occurrence of
    each intensity and then removing the intensities less than the
    intensity with highest number
    """
    npeak = len(spectrum)
    six = sorted(range(npeak), key=lambda k: spectrum[k][0])
    spectrum = [spectrum[i] for i in six]
    intensity = [v[1] for v in spectrum]
    sortx, npeak = sorted(intensity), len(intensity)

    if METHOD == 'hist':
        # count the number of intensities in specified bins
        if not isinstance(param, float) and not isinstance(param, int):
            raise Exception('Incorrect Parameter for this method, '+\
                            'should be integer or float number!')
        n, m0, int0, j, n0 = param, min(intensity), 0, 0, 0
        b = (max(intensity)-min(intensity))/float(n)
        for i in range(n):
            h = m0+b*(i+1)
            for k in range(j, npeak):
                if sortx[k]>=h:
                    break
            if n0<=k-j:
                n0, int0 = k-j, h
            j = k
        delx = [k for k in range(npeak) if intensity[k]<=int0]
        
    elif METHOD == 'median':
        # median value of intensities lower than median of
        # overall intensity median
        n = npeak//2
        int0 = median(sortx[:n])
        delx = [k for k in range(npeak) if intensity[k]<=int0]
        
    elif METHOD == 'relativebk':
        # relative intensity to base peak
        if not isinstance(param, float) and not isinstance(param, int):
            raise Exception('Incorrect Parameter for this method, '+\
                            'must be float number lower than 1.!')
        if param>=1:
            raise Exception('The fraction must be less than 1.')
        int0 = max(intensity)*param
        delx = [k for k in range(npeak) if intensity[k]<=int0]
        
    elif METHOD == 'ztopx':
        # select top x peaks in -z and +z region of high intensity
        # peak
        if not isinstance(param, tuple) and not isinstance(param, list):
            raise Exception('Incorrect Parameter for this method.')
        z, n = tuple(param)
        six = sorted(range(npeak), key = lambda k: intensity[k], reverse=True)
        mz = [x[0] for x in spectrum]
        delx, sxk = [], list(six)
        for i in six:
            if not sxk: break
            mz0 = mz[sxk[0]]
            kx = [j for j in sxk if abs(mz[j]-mz0)<=z]
            if len(kx)>n:
                sortxj = sorted(kx, key=lambda k: intensity[k], reverse=True)
                delx += sortxj[n:]
            sxk = [j for j in sxk if j not in kx]

    elif METHOD == 'wtopx':
        # select top x peaks in each of w regions
        if not isinstance(param, tuple) and not isinstance(param, list):
            raise Exception('Incorrect Parameter for this method.')
        w, n = tuple(param)
        mz = [x[0] for x in spectrum]
        b, j = (mz[npeak-1]-mz[0])/float(w), 0
        delx = []
        for i in range(w):
            h = mz[0]+b*(i+1)
            for k in range(j, npeak):
                if mz[k]>h:
                    break
            if k-j>n:
                sortxj = sorted(range(j,k), key=lambda jk: intensity[jk], reverse=True)
                delx += sortxj[n:]
            j = k

    elif METHOD == 'xinz':
        # select top x peaks in the windows with length of Z Da
        if not isinstance(param, tuple) and not isinstance(param, list):
            raise Exception('Incorrect Parameter for this method.')
        z, n = tuple(param)
        mz = [x[0] for x in spectrum]
        delx, j, nz, mzk = [], 0, int((mz[-1]-mz[0])/z)+1, mz[0]
        for i in range(nz):
            mzk += z
            for k in range(j, npeak):
                if mz[k]>mzk: break
            if k-j>n:
                sortxj = sorted(range(j,k), key=lambda jk: intensity[jk], reverse=True)
                delx += sortxj[n:]
            j = k

    elif METHOD == 'dnl':
        # dynamic noise level
        delx = []
        snr = param
        six = sorted(range(npeak), key=lambda k: intensity[k])
        if sortx[1]/sortx[0]<snr:
            si2, si = float(1+2**2), float(1+2)
            for k in range(2,npeak):
                ay = [sum((i+1)*sortx[i] for i in range(k)), sum(sortx[:k])]
                a = (ay[0]*k-ay[1]*si)/(k*si2-si**2)
                b = (ay[1]*si2-ay[0]*si)/(k*si2-si**2)
                #b = sum(sortx[i]*(k-(i+1)*si) for i in xrange(k))/(k*si2-si**2)
                h = a*(k+1)+b
                if sortx[k]/h>snr:
                    delx = six[:k]
                    break
                si2 += (k+1)**2
                si += k+1
        else:
            delx.append(six[0])

    return [x for i,x in enumerate(spectrum) if i not in delx]


def centroidms(spectrum):
    """
    Centroid tandem mass spectra according to the m/z difference.
    Currently, all fragment ions with adjacent m/z difference less
    than 0.1Da are centroided into the ion with highest intensity
    """
    if len(spectrum)<=1:
        return spectrum

    centroidmz, centroidint = [], []
    
    # sort the m/z values
    spectrum = sorted(spectrum, key=itemgetter(0))
    mz, intensity = [v[0] for v in spectrum], [v[1] for v in spectrum]
    # get m/z difference
    diffmz = [j-i for i,j in zip(mz[:-1],mz[1:])]
    b = [i<=0.1 for i in diffmz]

    n, i = len(diffmz), 0
    if n==1:
        if b[0]:
            j = intensity.index(max(intensity))
            return [spectrum[j]]
        else:
            return spectrum

    # centroid many peaks
    while True:
        j = i-1
        while True:
            j += 1
            if j==n-1 or not b[j]:
                break
        if j>i:
            intens_cluster = intensity[i:j+1]
            if len(set(intens_cluster))==1:
                centroidmz.append(sum(mz[i:j+1])/float(j-i+1))
                centroidint.append(intens_cluster[0])
            else:
                ix = intens_cluster.index(max(intens_cluster))
                centroidmz.append(mz[i+ix])
                centroidint.append(intensity[i+ix])
        else:
            centroidmz.append(mz[i])
            centroidint.append(intensity[i])

        i = j+1
        if i>=n:
            if not b[j]:
                centroidmz.append(mz[i])
                centroidint.append(intensity[i])
            break

    return list(zip(centroidmz, centroidint))


def binmz(mz, binwidth=1.0005, binoffset=0.4):
    """
    bin m/z values into 1.0005 with bin offset to be 0.4
    the binned mass is calculated according to:
    
    binned mass = floor((original mass / bin-width) + 1.0 - bin-offset)
    which is from:
    (http://cruxtoolkit.sourceforge.net/crux-faq.html; and
    http://comet-ms.sourceforge.net/parameters/parameters_201701/fragment_bin_tol.php)
    """

    return [floor(v/binwidth + 1. - binoffset) for v in mz]


def calsimilarity(target, matcher, wbin=1.0005, offset=0.4,
                  normalize=True, usesquare=False, numthr=5):
    """
    calculate the similarity between two spectra using dot product
    """
    #target = denoise(target)
    #matcher = denoise(matcher)
    # minimum number of peaks considered for similarity scoring
    n1, n2 = len(target), len(matcher)
    if len(target)<numthr or len(matcher)<numthr:
        return None
    n = max(n1,n2)

    targetmz, targetint = [v[0] for v in target], [v[1] for v in target]
    matchmz, matchint = [v[0] for v in matcher], [v[1] for v in matcher]

    # binned m/z
    targetmz = binmz(targetmz, binwidth=wbin, binoffset=offset)
    matchmz = binmz(matchmz, binwidth=wbin, binoffset=offset)

    # square root of intensity to remove the dominance of
    # very high intensities in similarity comparison
    if usesquare:
        targetint = [sqrt(v) for v in targetint]
        matchint = [sqrt(v) for v in matchint]

    # normalize intensity to highest intensity with maximum
    # intensity to be 100
    if normalize:
        m = float(max(targetint))
        targetint = [v/m*100. for v in targetint]
        m = float(max(matchint))
        matchint = [v/m*100. for v in matchint]

    # find matched intensities
    intensity_t, intensity_m = [], []
    mt, mm = [False]*len(targetmz), [False]*len(matchmz)
    for i, mzi in enumerate(matchmz):
        # find the peak with m/z in same bin and the target peak
        # not being assigned previously
        ix = [j for j, mzj in enumerate(targetmz) if mzj==mzi and not mt[j]]
        if ix:
            if len(ix)>1:
                matched_intdiff = [abs(targetint[j]-matchint[i]) for j in ix]
                ix = ix[matched_intdiff.index(min(matched_intdiff))]
            else:
                ix = ix[0]
            intensity_t.append(targetint[ix])
            intensity_m.append(matchint[i])
            mt[ix], mm[i] = True, True

    # once all peaks in same bin are matched between target and match
    # spectrum, find peaks in adjacent bins by shift bin in match spectrum
    # +1 and -1 then find shifted bin in target spectrum
    for i, mzi in enumerate(matchmz):
        if mm[i]: continue
        # find the peak with m/z in mzi+1 and mzi-1 in target spectrum
        # also the spectrum must not be assigned previously
        ix = [j for j, mzj in enumerate(targetmz)
              if (mzj+1==mzi or mzj-1==mzi) and not mt[j]]
        if ix:
            if len(ix)>1:
                matched_intdiff = [abs(targetint[j]-matchint[i]) for j in ix]
                ix = ix[matched_intdiff.index(min(matched_intdiff))]
            else:
                ix = ix[0]
            intensity_t.append(targetint[ix])
            intensity_m.append(matchint[i])
            mt[ix], mm[i] = True, True

    # calculate similarity score
    # .. to complect the two spectra, if any peak is not matched, 0
    for i in range(n):
        if i>=n1:
            if not mm[i]:
                intensity_t.append(0.)
                intensity_m.append(matchint[i])
        elif i>=n2:
            if not mt[i]:
                intensity_t.append(targetint[i])
                intensity_m.append(0.)
        else:
            if not mt[i]:
                intensity_m.append(0.)
                intensity_t.append(targetint[i])
            if not mm[i]:
                intensity_t.append(0.)
                intensity_m.append(matchint[i])
        
    return similarity(intensity_t,intensity_m)


def matchtops(target, matcher, num=3, mztol=1.):
    """
    identify whether top num peaks are matched between target and
    match spectrum
    """
    targetmz, targetint = [v[0] for v in target], [v[1] for v in target]
    matchmz, matchint = [v[0] for v in matcher], [v[1] for v in matcher]
    # get indices of sorted intensity in descending order
    sortx_target = sorted(list(range(len(targetint))), key=lambda k: targetint[k])[::-1]
    sortx_match = sorted(list(range(len(matchint))), key=lambda k: matchint[k])[::-1]

    # find matched m/z values
    mx, matchmzx, targetmzx = [], sortx_match[:num], sortx_target[:num]
    for i in matchmzx:
        mzdiff = [abs(matchmz[i]-targetmz[j]) for j in targetmzx]
        if any(d<=mztol for d in mzdiff):
            mx.append(mzdiff.index(min(mzdiff)))
        else:
            return False

    if len(set(mx))<num:
        return False

    return True


def clusterpeaks(spectra):
    """
    cluster spectra
    """
    pf = int(len(spectra)*0.6)
    
    cmz, cint = [], []
    # combine all spectra
    spa = []
    for spi in spectra:
        spa += spi
    mz, intensity = [v[0] for v in spa], [v[1] for v in spa]
    # sort the m/z values
    six = sorted(range(len(mz)), key=lambda k: mz[k])
    mz = sorted(mz)
    intensity = [intensity[i] for i in six]
    # get m/z difference
    diffmz = [j-i for i,j in zip(mz[:-1],mz[1:])]
    b = [i<=0.1 for i in diffmz]

    n, i = len(diffmz), 0
    if n==1:
        return spa

    # centroid many peaks
    while True:
        j = i-1
        while True:
            j += 1
            if j==n-1 or not b[j]:
                break
        if j>i:
            intens_cluster = intensity[i:j+1]
            nj = len(intens_cluster)
            if nj>=pf:
                cmz.append(sum(mz[i:j+1])/nj)
                cint.append(median(intens_cluster))

        i = j+1
        if i>=n:
            break
        
    return list(zip(cmz, cint))
    
