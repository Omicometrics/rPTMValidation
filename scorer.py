from math import sqrt, pi, exp


def normalPDF(sigma):
    """
    Probability density function (PDF) of normal distribution
    in this specific situation, the following is used, thus is not
    real PDF
    """
    return 1. / (sqrt(2*pi) * sigma)

def normalCDF(deltamass):
    """
    Approximation of cumulative density function (CDF) of normal
    distribution using Chebyshev fitting to an inspired guess
    (Numerical Recipes in Fortran 77, p214)
    if one use Python 2.7 or above, there exists an attribute in math
    module called 'erf', thus normal CDF can be calculated by
    (erf( deltamass / sqrt(2.) ) + 1.)/2.
    """
    z = abs(deltamass/sqrt(2.))
    t = 1./(1.+0.5*z)
    c0 = -1.26551223
    c1 = 1.00002368
    c2 = 0.37409196
    c3 = 0.09678418
    c4 = -0.18628806
    c5 = 0.27886807
    c6 = -1.13520398
    c7 = 1.48851587
    c8 = -0.82215223
    c9 = 0.17087277
    tau = t*exp(-z**2 + c0  + c1*t+ c2*(t**2) + c3*(t**3) + c4*(t**4) +
                c5*(t**5) + c6*(t**6) + c7*(t**7) + c8*(t**8) + c9*(t**9))

    if deltamass<0:
        return (tau-1.+1.)/2.
    else:
        return (1.-tau+1.)/2.

def modcost(deltamz, intvalue, mztol, wmz, wint, expc):
    """
    Cost of peak
    """
    c = wmz*(2*normalPDF(mztol/2.)*deltamz -
             2*abs(0.5-normalCDF(deltamz/(mztol/2.))))+\
             wint*pow(1.-intvalue, expc)
    return c

def score(spectrum, seqions, nitrosite,
          MATCHWEIGHT=0.75, MATCHTOL=1.0, INTMODE='RELRANK',
          INTWEIGHT=1.25, INTEXP=1.00):
    """
    Score the modified peptide using spectrum assigned to it according
    to the annotations of each peak in the spectrum
    """
    costs = []
    default_cost = 1.05
    #default_cost = 1.73
    # print mass
    maxintens = max(peak[1] for peak in spectrum)
    npeaks = len(spectrum)

    # the intensities used in calculating the scores
    if INTMODE=='RELRANK':
        spx = sorted(spectrum, key = lambda p: p[1], reverse = True)
        mz = [v[0] for v in spx]
        #relintens = [1.-(i+1.)/npeaks for i in xrange(npeaks)]
        relintens = [1.-i/float(npeaks-1) for i in range(npeaks)]
    else:
        mz = [v[0] for v in spectrum]
        relintens = [v[0]/maxintens for v in spectrum]

    # get cost list
    for ionmz, _ in seqions:
        cost = default_cost
        for i, mzj in enumerate(mz):
            dmz = abs(mzj - ionmz)
            if dmz < MATCHTOL:
                cost_t = modcost(dmz, relintens[i],
                                 MATCHTOL, MATCHWEIGHT, INTWEIGHT, INTEXP)
                if cost_t < cost:
                    cost = cost_t
        costs.append(cost)

    # get final scores
    l, s, s_clean = len(seqions), 0., 0.
    for k in range(nitrosite, nitrosite+3):
        s += costs[k-1] if k<=l else default_cost
##        if k <= l and costs[k-1]!=default_cost:
##            s_clean += 1./costs[k-1]
            
    return s#, s_clean


def score_all(spectrum, seqions, nitrosite,
          MATCHWEIGHT=0.75, MATCHTOL=1.0, INTMODE='RELRANK',
          INTWEIGHT=1.25, INTEXP=1.00):
    """
    Score the modified peptide using spectrum assigned to it according
    to the annotations of each peak in the spectrum
    """
    costs = []
    #default_cost = 1.05
    # print mass
    maxintens = max(peak[1] for peak in spectrum)
    npeaks = len(spectrum)

    # the intensities used in calculating the scores
    if INTMODE=='RELRANK':
        spx = sorted(spectrum, key = lambda p: p[1], reverse = True)
        mz = [v[0] for v in spx]
        relintens = [1.-(i+1.)/npeaks for i in range(npeaks)]
    else:
        mz = [v[0] for v in spectrum]
        relintens = [v[0]/maxintens for v in spectrum]

    # get cost list
    for ionmz, _ in seqions:
        #cost = default_cost
        for i, mzj in enumerate(mz):
            dmz = abs(mzj - ionmz)
            if dmz < MATCHTOL:
                cost_t = modcost(dmz, relintens[i],
                                 MATCHTOL, MATCHWEIGHT, INTWEIGHT, INTEXP)
                costs.append(cost_t)

    if costs:
        return sum(costs)/float(len(costs))
    else:
        # return max value of the cost in interval [0, 1]
        return 6.4782

##    # get final scores
##    l, s = len(seqions), 0
##    for k in xrange(nitrosite, nitrosite+3):
##        s += costs[k-1] if k<=l-1 else default_cost
##        
##    return s
    
