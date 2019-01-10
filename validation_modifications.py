import os, re, csv, sys
import process_rawdata as prd
import process_masspectrum as pms
import annotation_spectrum as annotates
import numpy as np
from _bisect import bisect_left
from itertools import combinations
from operator import itemgetter
import tempfuns


def adaptivedenoise(seq, mod, c, spectrum):
    """
    Adaptive denoising
    """
    if isinstance(mod, str):
        mod = ';'.join(xk for xk in mod.replace(' ','').split(';') if not xk.startswith('No'))
        mod = ';'.join(xk.split('ProteinTerminal')[1] if xk.startswith('ProteinTerminal') else xk for xk in mod.split(';'))
        try:
            modx = prd.parsemodifications(mod, prd.PTMDB)
        except:
            return None
    else:
        modx = mod
    # spx
    qmx = annotates.calpepmass(seq, modx)
    theoionsx = prd.allions(qmx, seq, modx, c)
    ions = prd.simpleannotation(theoionsx, spectrum)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spectrum)]
    slx = sorted(prd.adaptivedenoise(spectrum, ionsx))
    ions = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    spectrum = [spectrum[j] for j in slx]
    return spectrum, ions


def normalizespectrum(spectrum):
    """
    Normalize mass spectrum using base peak
    """
    m = max(spectrum, key=itemgetter(1))[1]
    return sorted([(mk[0], mk[1]/m) for mk in spectrum], key=itemgetter(0))


def getsimilarityscores(spx, spt, seq, mod, modn, c):
    """
    Get similarity scores
    """
    sqrt = np.sqrt
    spxk, ions1 = adaptivedenoise(seq, mod, c, spx)
    sptk, ions2 = adaptivedenoise(seq, modn, c, spt)
    # similarity score
    mixs = prd.matchNonMod(spxk, ions1, sptk, ions2)
    n1 = sum(sqrt(spxk[i][1])*sqrt(sptk[j][1]) for i,j in mixs if i is not None and j is not None)
    m1 = sqrt(sum(spxk[i][1] for i,_ in mixs if i is not None))
    m2 = sqrt(sum(sptk[i][1] for _,i in mixs if i is not None))
    return n1/(m1*m2)


def getvariables(seq, mod, charge, spectrum, targetmod):
    """
    Get variables
    """
    spxk, ions = adaptivedenoise(seq, mod, charge, spectrum)
    spxk = normalizespectrum(spxk)
    vx = prd.getvariables(seq, mod, charge, ions, spxk, targetmod, 0.2)
    tmc = sum(xk in 'RK' and seq[j+1]!='P' for j, xk in enumerate(seq[:-1]))
    vx.append(tmc)
    return vx


def getvalidationvars(targetmod, targetresidues):
    """
    Get data set for peptide validation according to the specified modification type
    """
    confs = {'I01': 92.9, 'I02': 91.6, 'I03': 88.8, 'I04': 88.7, 'I05': 86.2,
             'I06': 89.5, 'I07': 88.2, 'I08': 87.7, 'I09': 89.6, 'I10': 90.3,
             'I11': 93.3, 'I12': 92.5, 'I13': 89.4, 'I14': 89.1, 'I15': 93.1,
             'I16': 92.9, 'I17': 89.5, 'I18': 90.2, 'I19': 90.2, 'N01': 92.7,
             'N02': 92.5, 'N03': 92.7, 'N04': 95.7}
    
    resdir = r'D:\Data\Network_Analysis\monkey brain MS data\proteinpilot outputs_backup'
    resfiles = [f for f in os.listdir(resdir) if 'PeptideSummary' in f and f.endswith('txt')]
    print('number of data files: %d'%len(resfiles))
    ddir = r'D:\Data\PTN\MGF_complete'
    # set up constant parameters
    aares = prd.AARES
    uniptm = prd.parseUniprotPTM()
    mtag, mh2o, mh, mC = 304.20536, 18.006067, 1.0073, 57.021464
    residues = set('ACDEFGHIKLMNPQRSTVWY')
    # ....
    idents, ppres = [], {}
    # get target modifications at residue "resj"
    for fj in resfiles:
        res = prd.readPeptideSummary(os.path.join(resdir, fj))
        setj = fj.split('_')[1]
        try:
            pk = confs[setj]
        except:
            continue
        if not setj.startswith('I'): continue
        ppres[setj] = {}
        res = [xj for xj in res if xj[5]>=pk]
        for xj in res:
            mod = xj[1].replace(' ','')
            mod = ';'.join(xk for xk in mod.split(';') if not xk.startswith('No'))
            mod = ';'.join(xk.split('ProteinTerminal')[1] if xk.startswith('ProteinTerminal') else xk for xk in mod.split(';'))
            try:
                _ = prd.parsemodifications(mod, prd.PTMDB)
            except:
                continue
            if xj[5]>=pk and any('%s(%s)'%(targetmod, resj) in xj[1] for resj in targetresidues):
                idents.append((setj, xj[3], xj[0], mod, xj[2]))
            try:
                ppres[setj][xj[3]].append((xj[0], mod, xj[2], xj[5]))
            except:
                ppres[setj][xj[3]] = [(xj[0], xj[1], mod, xj[5])]
        print(fj)

    # get modification mass
    mod = idents[0][3]
    modx = prd.parsemodifications(mod, prd.PTMDB)
    for m1, _, sk in modx:
        if sk==targetmod:
            modmass = m1
            break

    # read tandem mass spectra
    print('load and process tandem mass spectra of identifications ...')
    identinfos = {}
    for x in idents:
        dtafile = os.path.join(ddir, '%s_MGFPeaklist\%s.dta'%x[:2])
        if not os.path.isfile(dtafile): continue
        modx = prd.parsemodifications(x[3], prd.PTMDB)
        if not any(mk==targetmod for _,_,mk in modx): continue
        spx = prd.dtareader(dtafile)
        spx = pms.centroidms(spx)
        spx = prd.removetagfrag(spx)
        # .. parse to a dictionary
        identinfos[x] = {}
        identinfos[x]['spectrum'] = list(spx)
        identinfos[x]['parsedmod'] = modx
        # .. get variables
        vk = getvariables(x[2], modx, x[4], spx, targetmod)
        identinfos[x]['variables'] = tuple(vk)
    print('total %d identifications ...'%len(identinfos))
        
    # ================================================================
    idents = list(identinfos.keys())
    peps = [prd.combinemod(x[2],x[3]) for x in identinfos.keys()]
    pepset = set(peps)
    print('read decoy peptides and generate variables ...')
    for resj in targetresidues:
        print('target modification "%s" at residue "%s"'%(targetmod, resj))
        fixaa2 = '%sKC'%resj
        targetmodstr = '%s(%s)'%(targetmod, resj)
        # PTMs
        ptmcomb = {}
        for ky in uniptm.keys():
            if ky in fixaa2: continue
            ptmcomb[ky] = list(set(xk for xk in uniptm[ky]['mono'] if abs(xk)<=100))
        mmx1, mmi1 = max(max(ptmcomb[ky]) for ky in ptmcomb.keys()), min(min(ptmcomb[ky]) for ky in ptmcomb.keys())
        # random sequences
        dbseqs = []
        with open(r'D:\Data\PTN\ProteinDigestionSimulator_Installer\macaca fascicularis_NCBI_20140113_reversed_digested_Mass400to6000.txt', 'r') as f:
            frd = csv.DictReader(f, delimiter='\t')
            for r in frd:
                if resj in r['Sequence']:
                    dbseqs.append(r['Sequence'])
        dbseqs = [x for x in dbseqs if len(x)>=7]
        dbseqs = [x for x in dbseqs if residues.issuperset(x)]
        dbseqs = list(set(dbseqs))
        yixs = [[i for i,xj in enumerate(x) if xj in resj] for x in dbseqs]
        dbkx, dbmods, dbseqmass = [], [], []
        for i,x in enumerate(dbseqs):
            mk = sum(aares[xk]['mono'] for xk in x)+mh2o
            modc = [(mtag, 'nterm', 'iTRAQ8plex')]
            fixmc = mtag
            if 'K' in x:
                modc += [(mtag, k+1, None) for k,xk in enumerate(x) if xk=='K']
                fixmc += mtag*x.count('K')
            if 'C' in x:
                modc += [(mC, k+1, None) for k,xk in enumerate(x) if xk=='C']
                fixmc += x.count('C')*mC
            yix = yixs[i]
            mk += fixmc
            if len(yix)==1:
                modc.append((modmass, yix[0]+1, targetmod))
                dbkx.append(i)
                dbseqmass.append(mk+modmass)
                dbmods.append(modc)
            else:
                for j in range(min(len(yix),3)):
                    for yk in combinations(yix, j+1):
                        dbkx.append(i)
                        dbseqmass.append(mk+modmass*len(yk))
                        dbmods.append(modc+[(modmass, k+1, targetmod) for k in yk])
        six = sorted(range(len(dbseqmass)), key=lambda k: dbseqmass[k])
        dbseqmass = sorted(dbseqmass)
        dbkx = [dbkx[i] for i in six]
        dbmods = [dbmods[i] for i in six]
        dbseqmass = np.array(dbseqmass)
        rgs = (dbseqmass[-1]-dbseqmass[0])/800
        rgixs, rgbds = [], []
        for i in range(801):
            zm = rgs*i+dbseqmass[0]
            j = bisect_left(dbseqmass, zm)
        if j==0:
            rgixs.append(0)
            rgbds.append(dbseqmass[0])
        elif j<len(dbseqmass):
            rgixs.append(j-1)
            rgbds.append(dbseqmass[j-1])
        else:
            rgixs.append(j-1)
            rgbds.append(dbseqmass[-1])

        print('total %d random sequences for target residue %s ...'%(len(dbseqs), resj))
        # get variables
        print('get variables and decoy identifications ...')
        for jj, pj in enumerate(pepset):
            kx = [i for i,xk in enumerate(peps) if xk==pj]
            tmx = idents[kx[0]]
            modx = identinfos[tmx]['parsedmod']
            cx = set([idents[i][4] for i in kx])
            for cj in cx:
                kjx = [i for i in kx if idents[i][4]==cj]
                tmx = idents[kjx[0]]
                pmz = (sum(aares[xk]['mono'] for xk in tmx[2])+sum(xk for xk, _, _ in modx)+mh2o)/cj
                randcc = []
                for crj in range(2, 5):
                    pmk = pmz*crj
                    rtol = 0.01*crj
                    rend = min([rgk for rgmk, rgk in zip(rgbds, rgixs) if rgmk>pmk+1]+[rgixs[-1]])
                    rstart = max([rgk for rgmk, rgk in zip(rgbds, rgixs) if rgmk<pmk-1]+[0])
                    rx, = np.where((dbseqmass[rstart:rend]>=pmk-rtol) & (dbseqmass[rstart:rend]<=pmk+rtol))
                    rx += rstart
                    randcc += [(dbseqs[dbkx[i]], dbmods[i], crj) for i in rx]
                    rstart = rgixs[max(bisect_left(rgbds, pmk-mmx1-1)-1, 0)]
                    rend = rgixs[min(bisect_left(rgbds, pmk-mmi1+1), len(rgixs)-1)]
                    dbmodsj = dbmods[rstart:rend]
                    dbseqsj = [dbseqs[i] for i in dbkx[rstart:rend]]
                    vmodixsj = [vmodixs[i] for i in dbkx[rstart:rend]]
                    dbseqmassj = dbseqmass[rstart:rend]
                    for ky in ptmcomb.keys():
                        _ = [randcc.extend((dbseqsj[i], dbmodsj[i]+[(xj, j+1, None)], crj) for j in vmodixsj[i] if dbseqsj[i][j]==ky)
                             for xj in ptmcomb[ky] for i in np.where((dbseqmassj>=pmk-rtol-xj)&(dbseqmassj<=pmk+rtol-xj))[0]]
                if len(randcc)<1000:
                    randcc = []
                    for crj in range(2, 5):
                        pmk = pmz*crj
                        rtol = 0.1*crj
                        rend = min([rgk for rgmk, rgk in zip(rgbds, rgixs) if rgmk>pmk+1]+[rgixs[-1]])
                        rstart = max([rgk for rgmk, rgk in zip(rgbds, rgixs) if rgmk<pmk-1]+[0])
                        rx, = np.where((dbseqmass[rstart:rend]>=pmk-rtol) & (dbseqmass[rstart:rend]<=pmk+rtol))
                        rx += rstart
                        randcc += [(dbseqs[dbkx[i]], dbmods[i], crj) for i in rx]
                        rstart = rgixs[max(bisect_left(rgbds, pmk-mmx1-1)-1, 0)]
                        rend = rgixs[min(bisect_left(rgbds, pmk-mmi1+1), len(rgixs)-1)]
                        dbmodsj = dbmods[rstart:rend]
                        dbseqsj = [dbseqs[i] for i in dbkx[rstart:rend]]
                        vmodixsj = [vmodixs[i] for i in dbkx[rstart:rend]]
                        dbseqmassj = dbseqmass[rstart:rend]
                        for ky in ptmcomb.keys():
                            _ = [randcc.extend((dbseqsj[i], dbmodsj[i]+[(xj, j+1, None)], crj) for j in vmodixsj[i] if dbseqsj[i][j]==ky)
                                 for xj in ptmcomb[ky] for i in np.where((dbseqmassj>=pmk-rtol-xj)&(dbseqmassj<=pmk+rtol-xj))[0]]
                if len(randcc)==0: continue
                # .. .. get spectra
                spxs, mxs, xxs = [], [], []
                for i in kjx:
                    tmx = idents[i]
                    spx = identinfos[tmx]['spectrum']
                    spxs.append(spx)
                    mxs.append(max(spx, key=itemgetter(1))[1])
                    xxs.append(tmx)
                # .. .. get rand results
                m = max(mxs)
                spx = spxs[mxs.index(m)]
                nrions = []
                nrapp = nrions.append
                mzx = sorted([xk[0] for xk in spx])
                for sj, rmodj, crj in randcc:
                    rqm = annotates.calpepmass(sj, rmodj)
                    rions = prd.theoionmzall_by(tuple(rqm), sj, rmodj, crj)
                    lj = len(rions)
                    bx = [bisect_left(rions, xk) for xk in mzx]
                    rn = sum(1 for k, xk in zip(bx, mzx) if (k>0 and xk-rions[k-1]<=0.2) or (k<lj and rions[k]-xk<=0.2))
                    nrapp(rn)
                six = sorted(range(len(nrions)), key=lambda k: nrions[k], reverse=True)
                randcct = [randcc[j] for j in six[:1000]]
                for jk, (spx, xxk) in enumerate(zip(spxs, xxs)):
                    rsx = []
                    rsapp = rsx.append
                    for sj, rmodj, crj in randcct:
                        rvars1 = getvariables(sj, rmodj, crj, spx, targetmod)
                        rsapp(rvars1)
                    rmvk = max(rsx, key = itemgetter(8))
                    j = rsx.index(rmvk)
                    rxx = list(randcct[j])
                    rxx.append(tuple(rmvk))
                    try:
                        rmvp = identinfos[xxk]['randidentification'][3]
                        if rmvp[8]<rmvk[8]:
                            identinfos[xxk]['randidentification'] = tuple(rxx)
                    except:
                        identinfos[xxk]['randidentification'] = rxx
            if (jj+1)%100==0:
                print(jj)

    # ================================================================
    # get similarity scores
    print('get similarity scores ...')
    for x in idents:
        mod = x[3]
        modn = ';'.join(xk for xk in mod.split(';') if not xk.startswith('No')
                        and not any(xk.startswith('%s(%s)'%(targetmod, rj)) for rj in targetresidues))
        pxk2 = prd.combinemod(x[2], modn)
        spx = identinfos[x]['spectrum']
        kx = []
        for ky1 in ppres.keys():
            for ky2 in ppres[ky1].keys():
                for xk in ppres[ky1][ky2]:
                    if not (xk[0]==x[2] and xk[2]==x[4]): continue
                    modn2 = xk[1]
                    if 'Deamidated' in modn2:
                        modn2 = ';'.join(xk for xk in modn2.split(';') if not xk.startswith('Deamidated'))
                    if prd.combinemod(xk[0], modn2)==pxk2:
                        kx.append((ky1, ky2))
                        break
        bx = []
        if kx:
            modx = identinfos[x]['parsedmod']
            modxn = prd.parsemodifications(modn, prd.PTMDB)
            for ky1, ky2 in kx:
                dtafile = os.path.join(ddir, '%s_MGFPeaklist\%s.dta'%(ky1, ky2))
                if not os.path.isfile(dtafile): continue
                spt = prd.dtareader(dtafile)
                spt = pms.centroidms(spt)
                spt = prd.removetagfrag(spt)
                sk = getsimilarityscores(spx, spt, x[2], modx, modxn, x[4])
                if sk is not None:
                    bx.append((ky1, ky2, sk))
        identinfos[x]['similarities'] = tuple(bx)

    # ================================================================
    # write results to files
    print('write to file ...')
    with open(r'modifications_%s_infos.txt'%(targetmod), 'w') as f:
        f.write('Rawset\tSpectrumID\tSequence\tModifications\tCharge\tFeatures\tSimilarityScore\t'+\
                'Decoy_sequence\tDecoy_modifications\tDecoy_charge\tDecoy_features\n')
        for x in identinfos.keys():
            mk = []
            try:
                x2 = identinfos[x]['randidentification']
            except:
                continue
            for xk in x2[1]:
                if isinstance(xk[1], str):
                    mk.append('%.6f|%s|%s'%xk)
                else:
                    if xk[2] is None:
                        mk.append('%.6f|%d|%s'%(xk[0], xk[1], str(xk[2])))
                    else:
                        mk.append('%.6f|%d|%s'%(xk[0], xk[1], xk[2]))
            bx = identinfos[x]['similarities']
            if not bx:
                ax = 'none'
            else:
                ax = ';'.join('%s#%s:%.6f'%xk for xk in bx)
            vk = identinfos[x]['variables']
            f.write('%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%s\n'%
                    (x[0], x[1], x[2], x[3], x[4], ','.join('%.8f'%xk for xk in vk), ax, x2[0], ','.join(mk), x2[2], ','.join('%.8f'%xk for xk in x2[3])))


def main():
    script = sys.argv[0]
    mod = sys.argv[1]
    resx = sys.argv[2]
    getvalidationvars(mod, resx)

main()
