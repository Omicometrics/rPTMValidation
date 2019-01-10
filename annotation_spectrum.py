"""
Module for analysis of peptide sequence and calculation of sequential
ion mass for annotating tandem mass spectra. All functions defined
in this module will be integrated into a class involving peptide
sequence and tandem mass spectrum analysis.
"""
import re, csv
from collections import Counter
from _bisect import bisect_left


AARES = {'G': {'mono': 57.02146, 'avg': 57.052},
         'A': {'mono': 71.03711, 'avg': 71.078},
         'S': {'mono': 87.03203, 'avg': 87.078},
         'P': {'mono': 97.05276, 'avg': 97.117},
         'V': {'mono': 99.06841, 'avg': 99.133},
         'T': {'mono': 101.04768, 'avg': 101.105},
         'C': {'mono': 103.00918, 'avg': 103.144},
         'I': {'mono': 113.08406, 'avg': 113.160},
         'L': {'mono': 113.08406, 'avg': 113.160},
         'N': {'mono': 114.04292, 'avg': 114.104},
         'D': {'mono': 115.02693, 'avg': 115.089},
         'Q': {'mono': 128.05857, 'avg': 128.131},
         'K': {'mono': 128.09495, 'avg': 128.174},
         'E': {'mono': 129.04258, 'avg': 129.116},
         'M': {'mono': 131.04048, 'avg': 131.198},
         'H': {'mono': 137.05891, 'avg': 137.142},
         'F': {'mono': 147.06841, 'avg': 147.177},
         'R': {'mono': 156.10110, 'avg': 156.188},
         'Y': {'mono': 163.06332, 'avg': 163.170},
         'W': {'mono': 186.07931, 'avg': 186.213}
         }

NEUTRALS = {'w': 18.006067,   # mass of water
            'a': 16.998767,    # mass of ammonia
            'c': 28.0101,   # mass of carbon mono-oxide, CO
            'h': 1.0073     # mass of hydrogen
            }

MTERMINUS = {'cterm': {'mono': 16.998767, 'avg': 17.008},
             'nterm': {'mono': 1.0073, 'avg': 1.0079}}

MELEMENT = {'H':{'mono': 1.0073, 'avg':1.0079},
            '2H':{'mono': 2.0141, 'avg': 2.0141},
            'Li':{'mono': 6.941, 'avg': 2.0141},
            'C': {'avg': 12.0107, 'mono': 12},
            '13C': {'avg': 13.0034, 'mono': 13.0033},
            'N': {'avg': 14.0067, 'mono': 14.0031},
            '15N': {'avg': 15.0001, 'mono': 15.0001},
            'O': {'avg': 15.9994, 'mono': 15.9949},
            '18O': {'avg': 17.99916, 'mono': 17.99916},
            'F': {'avg': 18.99840, 'mono': 18.99840},
            'Na': {'avg': 22.98977, 'mono': 22.98977},
            'P': {'avg': 30.97376, 'mono': 30.97376},
            'S': {'avg': 32.065, 'mono': 31.97207},
            'Cl': {'avg': 35.453, 'mono': 34.96885},
            'K': {'avg': 39.0983, 'mono': 38.96371},
            'Ca': {'avg': 40.078, 'mono': 39.96259},
            'Fe': {'avg': 55.845, 'mono': 55.93494},
            'Ni': {'avg': 58.6934, 'mono': 57.93535},
            'Zn': {'avg': 65.409, 'mono': 63.92914},
            'Se': {'avg': 78.96, 'mono': 79.91652},
            'Br': {'avg': 79.904, 'mono': 78.91834},
            'Ag': {'avg': 107.8682, 'mono': 106.90509},
            'Hg': {'avg': 200.59, 'mono': 201.97062},
            'Au': {'avg': 196.96655, 'mono': 196.96654},
            'I': {'avg': 126.90447, 'mono': 126.90447},
            'Mo': {'avg': 95.94, 'mono': 97.9054073},
            'Cu': {'avg': 63.546, 'mono': 62.9295989},
            'e': {'avg': 0.000549, 'mono': 0.000549},
            'B': {'avg': 10.811, 'mono': 11.0093055},
            'As': {'avg': 74.9215942, 'mono': 74.9215942},
            'Cd': {'avg': 112.411, 'mono': 113.903357},
            'Cr': {'avg': 51.9961, 'mono': 51.9405098},
            'Co': {'avg': 58.933195, 'mono': 58.9331976},
            'Mn': {'avg': 54.938045, 'mono': 54.9380471},
            'Mg': {'avg': 24.305, 'mono': 23.9850423},
            'Pd': {'avg': 106.42, 'mono': 105.903478},
            'Al': {'avg': 26.9815386, 'mono': 26.9815386}
            }
                


def getmodificationdb():
    """
    Read modifications from UNIMOD (http://www.unimod.org)
    """
    modifications = {}
    with open('unimod.txt', 'r') as f:
        csvreader = csv.DictReader(f,delimiter='\t')
        keys = csvreader.fieldnames
        for key in keys:
            modifications[key] = []
        for row in csvreader:
            for key in keys:
                value = float(row[key]) if 'mass' in key else row[key]
                modifications[key].append(value)
    return modifications


def parsemodformula(string, mtype):
    """
    Parse modification fomula which represents the ambiguous name,
    empirical formula.
    """
    m = 0.

    # parse string to get 
    def parstring(substring, elemass):
        mx = 0.
        if not substring or not substring.startswith('('):
            mx += elemass
        else:
            mx += elemass*int(substring.split(')')[0][1:])
        return mx
    
    for key in MELEMENT.keys():
        # firstly consider the element whose name uses more
        # than 2 vectors, so no ambiguous element assignment
        # can occur
        if len(key)>1:
            if key in string:
                sstr = string.split(key)
                for s in sstr[1:]:
                    mt = parstring(s, MELEMENT[key][mtype])
                    m += mt
    for key in MELEMENT.keys():
        if len(key)==1:
            if key in string:
                sstr = string.split(key)
                for s in sstr[1:]:
                    mt = parstring(s, MELEMENT[key][mtype])
                    m += mt

    return m


def parsemodifications_temp(modinfo, mtype='mono'):
    """ temp """
    modifdb = {'iTRAQ8plex': 304.205360,
               'Carbamidomethyl': 57.021464,
               'Nitro': 44.985078,
               'Oxidation': 15.994915,
               'Deamidated': 0.984016,
               'Gln->pyro-Glu': -17.026549,
               'Acetyl': 42.010565}

    modpx = []
    if not modinfo:
        return None
    
    mods = modinfo.split(';')
    for modi in mods:
        mx = modi.strip().split('@')
        # get modification mass
        tx = mx[0].split('(')[0]
        m = modifdb[tx]

        # get modification site
        try:
            site = int(mx[1])
        except:
            mxl = mx[1].lower()
            if mxl.startswith('c') and 'term' in mxl:
                site = 'cterm'
            elif mxl.startswith('n') and 'term' in mxl:
                site='nterm'
        modpx.append((m, site, tx))
        
    return modpx


def parsemodifications(modinfo, PTMDB, mtype='mono'):
    """
    Parse modifications from input modinfo
    """
    if not modinfo:
        return None
    
    modpx = []
    masskey = 'Monoisotopic mass' if mtype=='mono' else 'Average mass'
    def parsemodstring(modstring, mtype=mtype, PTMDB=PTMDB):
        mx = modi.strip().split('@')
        # get modification mass
        tx = mx[0]
        if '(' in tx:
            tj = max(i for i,x in enumerate(tx) if x=='(')
            tx = tx[:tj]
            
        if tx in PTMDB['PSI-MS Name']:
            i = PTMDB['PSI-MS Name'].index(tx)
            m = PTMDB[masskey][i]
        elif tx in PTMDB['Interim name']:
            i = PTMDB['Interim name'].index(tx)
            m = PTMDB[masskey][i]
        else:
            tx = tx.replace(' ','').lower()
            if tx.startswith('delta'):
                m = parsemodformula(tx, mtype)
            else:
                t = False
                for i, s in enumerate(PTMDB['Description']):
                    if s.replace(' ','').lower() == tx:
                        m = PTMDB[masskey][i]
                        t = True
                        break
                if not t:
                    raise NameError('Unrecognizable modification: %s.'%tx)

        # get modification site
        try:
            site = int(mx[1])
        except:
            mxl = mx[1].lower()
            if mxl.startswith('c') and 'term' in mxl:
                site = 'cterm'
            elif mxl.startswith('n') and 'term' in mxl:
                site='nterm'
            else:
                raise ValueError('Invalid modification site specified.')
        return m, site, tx
    
    if isinstance(modinfo, str):
        if '@' in modinfo:
            mods = modinfo.split(';')
            for modi in mods:
                modpx.append(parsemodstring(modi))
        else:
            sx, naa = modinfo.split(']'), 0
            for k, kx in enumerate(sx[:-1]):
                subpx, mx = tuple(kx.split('['))
                try:
                    modm = float(mx)
                except:
                    raise NameError('Unrecognized modification in peptide sequence.')
                if subpx=='n':
                    site = 'nterm'
                elif subpx.endswith('c'):
                    site = 'cterm'
                else:
                    naa += len(subpx)
                    site = naa
                    modm -= AARES[subpx[-1]][mtype]
                # try to find the modification name
                md = [abs(mi-modm) for mi in PTMDB[masskey]]
                i = md.index(min(md))
                modpx.append((modm, site, PTMDB['Interim name'][i]))
    elif isinstance(modinfo, list):
        for modi in modinfo:
            if isinstance(modi, str):
                if '@' not in modi:
                    raise NameError('Incorrect modification expression.')
                modpx.append(parsemodstring(modi))
            else:
                if len(modi) != 2:
                    raise NameError('Incorrect modification expression.')
                if not isinstance(modi[0], float):
                    raise NameError('Incorrect modification expression.')
                if not isinstance(modi[1], int) and not isinstance(modi[1], float):
                    mx = modi[1].lower()
                    if mx.startswith('c') and 'term' in mx:
                        site = 'cterm'
                    elif mx.startswith('n') and 'term' in mx:
                        site = 'nterm'
                    else:
                        raise NameError('Incorrect modification expression.')
                if int(modi[1]) != modi[1]:
                    raise NameError('Incorrect modification expression.')
                site = int(modi[1])
                md = [abs(mi-modi[0]) for mi in PTMDB[masskey]]
                i = md.index(min(md))
                modpx.append(tuple([modi[0], site, PTMDB['Interim name'][i]]))

    return modpx


def calpepmass(sequence, modifications, mtype='mono'):
    """
    Calculate peptide mass according to the sequence and modification
    specified.
    The modifications must be firstly sparsed by function "parsemodifications".
    """
    
    seqmass = []
    n, mn, mc = len(sequence), None, None
    modseq = [0.]*n
    if modifications:
        for m, j, _ in modifications:
            if isinstance(j, int):
                modseq[j-1] += m
            else:
                if j == 'cterm':
                    mc = m
                else:
                    mn = m

    for i, aa in enumerate(sequence):
        seqmass.append(AARES[aa][mtype]+modseq[i])

    return [mn, seqmass, mc]


def generatetheoions(seqmass, ntermmass=None, ctermmass=None):
    """
    Theoretical ions
    """
    mh, mh2o = NEUTRALS['h'], NEUTRALS['w']
    l = len(seqmass)

    # y ions
    if not ctermmass:
        yi = mh2o
    else:
        yi = ctermmass+mh
    seqmass4y = seqmass[::-1]
    yions = [yi+sum(seqmass4y[:i]) for i in range(1,l)]

    # b ions
    bi = 0. if not ntermmass else ntermmass
    bions = [bi+sum(seqmass[:i]) for i in range(1,l)]

    pmass = sum(seqmass)+mh2o
    if ntermmass: pmass += ntermmass        

    return bions, yions, pmass


def generateinternals(seqmass):
    """
    Generate internal fragment ions
    """
    h = NEUTRALS['h']
    l = len(seqmass)
    return [((i, j+1), sum(seqmass[i:j+1])+h)
            for i in range(1, l-3) for j in range(i+1, l-1)]


def generateimmoniums(seqmass, NEUCONST=NEUTRALS):
    """
    m/z of immonium ions
    """
    c, h = NEUCONST['c'], NEUCONST['h']
    return tuple([m-c+h for m in seqmass])


def generateions(bions, yions, precursormass, charge):
    """
    Generate ion m/z and their names
    """
    ions = []
    mw, ma, mco, mh = NEUTRALS['w'], NEUTRALS['a'], NEUTRALS['c'], NEUTRALS['h']

    # get ions and names with different charges, neutral losses
    def getions(ionmass, prefix, charge):
        iontemps = []
        for i, m in enumerate(ionmass):
            scions = []
            scions.append((m+mh, '%s%d[+]'%(prefix, i+1)))
            scions.append((m+mh-mw, '%s%d-H2O[+]'%(prefix, i+1)))
            scions.append((m+mh-ma, '%s%d-NH3[+]'%(prefix, i+1)))
            #if i >= 1:  # two neutral losses
##                scions.append((m+mh-mw-mw, '%s%d-2H2O[+]'%(prefix, i+1)))
##                scions.append((m+mh-ma-ma, '%s%d-2NH3[+]'%(prefix, i+1)))
##                scions.append((m+mh-ma-mw, '%s%d-NH3-H2O[+]'%(prefix, i+1)))
            iontemps += list(scions)

            if charge>=2 and i>=2:
                # only ions with number of residues>=3 are considered
                # to be doubly charged
                for ionmass, name in scions:
                    ionmass += mh
                    ionmass /= 2.
                    jx = name.index('+')
                    name = name[:jx]+'2'+name[jx:]
                    iontemps.append((ionmass, name))

            if charge>=3 and i>=4:
                # only ions with number of residues>=5 are considered
                # to be triply charged
                for ionmass, name in scions:
                    ionmass += 2*mh
                    ionmass /= 3.
                    jx = name.index('+')
                    name = name[:jx]+'3'+name[jx:]
                    iontemps.append((ionmass, name))
                    
        return iontemps

    # generate ions
    # .. b ions
    ions += getions(bions, 'b', charge)
    # .. y ions
    ions += getions(yions, 'y', charge)
    # .. a ions
    ions += getions([m-mco for m in bions], 'a', charge)
    
    # precursor ion series
    pions = []
    for i in range(charge):
        c = i+1
        pions.append(((precursormass+c*mh)/c, 'p[%d+]'%c))
        pions.append(((precursormass+c*mh-mw)/c, 'p-H2O[%d+]'%c))
        pions.append(((precursormass+c*mh-ma)/c, 'p-NH3[%d+]'%c))
##        pions.append(((precursormass+c*mh-2*ma)/c, 'p-2NH3[%d+]'%c))
##        pions.append(((precursormass+c*mh-2*mw)/c, 'p-2H2O[%d+]'%c))
##        pions.append(((precursormass+c*mh-ma-mw)/c, 'p-NH3-H2O[%d+]'%c))
    ions += pions

    return ions
    

def annotatespectrum(spectrum, theoions, mztol=1., mztolunit='Da'):
    """
    Annotate spectrum according to the sequence input
    """
    f, g = (0, mztol) if mztolunit=='Da' else (mztol*1e-6, 0)

    mz = [x[0] for x in spectrum]
    ionsmatch = [[(im-mzk, imn) for im, imn in theoions if abs(im-mzk)<=mzk*f+g] for mzk in mz]
    ionnames = [[xj[1] for xj in x] if x else [] for x in ionsmatch]
    ionerr = [[xj[0] for xj in x] if x else [] for x in ionsmatch]

    return ionnames, ionerr


def neutralassign(spectrum, ionnames, ionerr):
    """
    Assign conflict neutral losses to peaks, especially H2O and NH3
    """
    mz, intensity = [v[0] for v in spectrum], [v[1] for v in spectrum]
    n = len(ionnames)
    
    def matchix(namelist,mz,intensity,i,conflict_name,errup,errlow):
        # get matched ix
        nions = len(mz)
        if any(namei==conflict_name for namei in namelist):
            ix, ixint = [], []
            for j in range(nions):
                if mz[j]>=mz[i]-errup and mz[j]<=mz[i]-errlow and j!=i:
                    ix.append(j)
                    ixint.append(intensity[j])
            if ix:
                return ix[ixint.index(max(ixint))]
        return -1
    
    for i in range(n):
        if len(ionnames[i])<=1: continue
        
        name, errx = list(ionnames[i]), list(ionerr[i])
        nh3info = []
        for j, namei in enumerate(name):
            if 'NH3' in namei:
                nh2o = namei.count('H2O')
                # charge state of the ion
                try:
                    c = int(namei[namei.index('+')-1])
                except:
                    c = 1
                # number of NH3
                try:
                    nnh3 = int(namei[namei.index('NH3')-1])
                except:
                    nnh3 = 1
                # ion series name, like b, or y that generate
                # the neutral loss
                px = namei.split('-')[0]
                nh3info.append((px, c, nh2o, nnh3, namei, errx[j]))
                
        if not nh3info: continue

        delname = set()
        for px, c, n1, n2, cname, cerr in nh3info:
            cc = '[+]' if c==1 else '[%d+]'%c
            # find the neutral loss name if multiple neutral losses
            # assigned to the same peak
            cftname = cname.replace('NH3', 'H2O')
            if cftname.count('-H2O')==2:
                cftname = '%s-2H2O%s'%(px,cc)
            jx = [j for j, namej in enumerate(name) if namej==cftname]
            errthr = 0.8 if c==1 else 0.6
            if jx and abs(cerr)>=errthr:
                delname.add(cname)
                continue
            elif jx and abs(errx[jx[0]])>=errthr:
                delname.add(cftname)
                continue

            # set -H2O and -NH3 name counterparts
            if n2 == 1:
                nh2o = '-H2O' if n1==0 else '-%dH2O'%(n1+1)
                cftname = '%s%s%s'%(px,nh2o,cc)
            else:
                cftname = '%s-NH3-H2O%s'%(px,cc)

            # find the conflict ion name and assign to other peak
            # .. backward search
            j = matchix(name,mz,intensity,i,cftname,1./c+0.5,1./c-0.5)
            if j>=0:
                ionnames[j].append(cname)
                ionerr[j].append(cerr)
                delname.add(cftname)
            # .. forward search
            j = matchix(name,mz,intensity,i,cftname,-1./c+0.5,-1./c-0.5)
            if j>=0:
                ionnames[j].append(cname)
                ionerr[j].append(cerr)
                delname.add(cname)

            # find peaks with higher m/z being -H2O whereas lower
            # m/z being -NH3
            if n2>=1:
                for j in range(i+1, n):
                    if cftname in ionnames[j]:
                        delname.add(cname)
                        break

        # remove the ion names
        ionerr[i] = [ej for j,ej in enumerate(errx) if name[j] not in delname]
        ionnames[i] = [namej for namej in name if namej not in delname]

    return ionnames, ionerr


def removerepions(ionnames, ionerr):
    """
    Remove the ion name assigned to multiple fragment ions.
    """
    # remove replicate ion name for single fragment ion
    for i, name in enumerate(ionnames):
        if len(name)<=1: continue
        errx, nameset = ionerr[i], set(name)
        if len(nameset)!=len(name):
            nameix = [name.index(nj) for nj in nameset]
            ionnames[i] = list(nameset)
            ionerr[i] = [errx[j] for j in nameix]
            
    # get all ion names
    allnames = []
    [allnames.extend(x) for x in ionnames]
    nameixs = [i for i,x in enumerate(ionnames) if x]
    # get unique names and number of each unique name
    namecounter = Counter(allnames)
    # remove those replicate ion names according to the errors
    # between observed and theoretical ion m/z
    for key in namecounter.keys():
        if namecounter[key]>1:
            err, cix = [], []
            for k in nameixs:
                sname = list(ionnames[k])
                kx = [j for j, x in enumerate(sname) if x==key]
                if kx:
                    cix.append(k)
                    err.append(abs(ionerr[k][kx[0]]))
            # retain the name with the smallest error
            k = cix[err.index(min(err))]
            kx = set([kj for kj in cix if kj!=k])
            for kj in kx:
                if len(ionnames[kj])==1:
                    ionnames[kj] = []
                    ionerr[kj] = []
                    nameixs.remove(kj)
                else:
                    namet, errt = ionnames[kj], ionerr[kj]
                    ionerr[kj] = [errt[j] for j,x in enumerate(namet) if x!=key]
                    ionnames[kj] = [x for x in namet if x!=key]

    for i in range(len(ionnames)):
        if isinstance(ionnames[i], list):
            if not ionnames[i]:
                ionnames[i] = ''
            else:
                ionnames[i]=','.join('%s/%.4f'%(x,e) for x,e in zip(ionnames[i],ionerr[i]))
            
    return ionnames


def checknitro(mod_info):
    """
    Check whether nitration exists in peptide. If exist, return the
    sites.
    """
    mods = parsemodifications_temp(mod_info)
    sites = []
    for _, i, name in mods:
        if name == 'Nitro':
            sites.append(i)
    if not sites:
        raise ValueError('No nitrated modification exists in peptide.')
    return sites
