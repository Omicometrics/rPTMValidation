import os, re
import process_rawdata as prd
import process_masspectrum as pms
import annotation_spectrum as annotes
import numpy as np
from _bisect import bisect_left
from itertools import combinations
from operator import itemgetter


def getpepxml(filename):
    """
    Read Comet search results (pep.xml)
    """
    res = []
    with open(filename, 'r') as f:
        hits = []
        for line in f:
            sx = re.findall('"([^"]*)"', line.rstrip())
            if '<spectrum_query' in line:
                queryinfo = sx[1].split(':')
                ck = int(sx[5])
            elif '<search_hit' in line:
                pk = sx[1]
                cix = [j+1 for j, xk in enumerate(pk) if xk=='C']
                nk = 'decoy' if sx[4].startswith('DECOY_') else 'normal'
            elif '<modification_info' in line:
                modj = []
                sx2 = line.rstrip().split('=')
                bk = [i for i,xk in enumerate(sx2) if xk.endswith('mod_nterm_mass')]
                if bk:
                    jk = bk[0]
                    if int(float(sx[jk]))==305:
                        modj.append((0, 'iTRAQ8plex@N-term'))
                    elif int(float(sx[jk]))==58:
                        modj.append((0, 'Carbamidomethyl@N-term'))
                    elif int(float(sx[jk]))==44:
                        modj.append((0, 'Carbamyl@N-term'))
                if cix:
                    modj+=[(j, 'Carbamidomethyl(C)@%d'%j) for j in cix]
            elif '<mod_aminoacid_mass' in line:
                j = int(sx[0])
                vkj = float(sx[2])
                if abs(vkj-44.98)<1:
                    modj.append((j, 'Nitro(Y)@%d'%j))
                elif abs(vkj-15.99)<1:
                    modj.append((j, 'Oxidation(%s)@%d'%(pk[j-1], j)))
                elif abs(vkj+17.03)<1:
                    modj.append((j, 'Gln->pyro-Glu(%s)@%d'%(pk[j-1], j)))
                elif abs(vkj-0.98)<1:
                    modj.append((j, 'Deamidated(%s)@%d'%(pk[j-1], j)))
                elif abs(vkj-304.2054)<1:
                    modj.append((j, 'iTRAQ8plex(%s)@%d'%(pk[j-1], j)))
            elif '<search_score' in line:
                if sx[0]=='xcorr': xj = float(sx[1])
            elif '</search_hit>' in line:
                modj = sorted(modj)
                hits.append((pk, ';'.join(mk for _,mk in modj), ck, xj, nk))
            elif '</spectrum_query>' in line:
                res.append((queryinfo[0], queryinfo[1], hits))
                hits = []
    return res


def getppxml(filename):
    """
    Read ProteinPilot search results (.xml)
    """
    res = []
    with open(filename, 'r') as f:
        hits, t = [], False
        for line in f:
            sx = re.findall('"([^"]*)"', line.rstrip())
            if line.startswith('<SPECTRUM'):
                queryid = sx[6]
                pms = float(sx[4])
                t = True
            elif t:
                if line.startswith('<MATCH'):
                    sline = line.rstrip().split('=')
                    for i in range(len(sx)):
                        if sline[i].endswith('charge'):
                            ck = int(sx[i]) # charge
                        if sline[i].endswith('confidence'):
                            conf = float(sx[i])   # confidence
                        if sline[i].endswith('seq'):
                            pk = sx[i]   # sequence
                        if sline[i].endswith('type'):
                            nk = 'decoy' if int(sx[i])==1 else 'normal'
                        if sline[i].endswith('score'):
                            sk = float(sx[i])
                    modj = []
                elif line.startswith('<MOD_FEATURE'):
                    j = int(sx[1])
                    modj.append('%s(%s)@%d'%(sx[0], pk[j-1], j))
                elif line.startswith('<TERM_MOD_FEATURE'):
                    if not sx[0].startswith('No'):
                        modj.insert(0, 'iTRAQ8plex@N-term')
                elif line.startswith('</MATCH>'):
                    hits.append((pk, ';'.join(modj), ck, pms, conf, sk, nk))
                elif line.startswith('</SPECTRUM>'):
                    res.append((queryid, hits))
                    hits, t = [], False
    return res


def getfullmgf(filename):
    """
    Get all information of an mgf file, including RT, pep m/z,
    charge, and mass spectrum
    """
    mspectra = []
    with open(filename, 'r') as f:
        msk = []
        for line in f:
            if line.startswith('END IONS'):
                mspectra.append((spid, (pepmass, c, rt, msk)))
                c = 0
                msk = []
            elif line.startswith('TITLE'):
                spid = line.rstrip().split()[0].split(':')[1]
            elif line.startswith('PEPMASS'):
                pepmass = float(line.rstrip().split('=')[1])
            elif line.startswith('RTINSECONDS'):
                # get retention time and convert seconds to minutes
                rt = float(line.rstrip().split('=')[1])/60.
            elif line.startswith('CHARGE'):
                c = int(line.rstrip().split('=')[1][0])
            else:
                try:
                    msk.append(tuple([float(xk) for xk in line.rstrip().split()[:2]]))
                except:
                    pass
    return dict(mspectra)    


def getmgfinfos():
    """
    Get retention times, spectrum ID, precursor m/z, estimated charge state for
    PTN validation
    """
    mgfinfos = {}
    rawsets = ['I08', 'I17', 'I18', 'I19']
    for rj in rawsets:
        c, mjx = 0, []
        with open(r'D:\Data\PTN\MGF_complete\%s_MGFPeaklist.mgf'%rj, 'r') as f:
            for line in f:
                if line.startswith('END IONS'):
                    mjx.append((spid, (pepmass, c, rt)))
                    c = 0
                elif line.startswith('TITLE'):
                    spid = line.rstrip().split()[0].split(':')[1]
                elif line.startswith('PEPMASS'):
                    pepmass = float(line.rstrip().split('=')[1])
                elif line.startswith('RTINSECONDS'):
                    # get retention time and convert seconds to minutes
                    rt = float(line.rstrip().split('=')[1])/60.
                elif line.startswith('CHARGE'):
                    c = int(line.rstrip().split('=')[1][0])
        mgfinfos[rj] = dict(mjx)
    return mgfinfos


def getfullmsp(filename):
    """
    Get full information from NIST peptide mass spectral library
    """
    mspectra = []
    with open(filename, 'r') as f:
        msk, badmod = [], False
        for line in f:
            if line.startswith('Name'):
                if msk and not badmod:
                    mspectra.append((seq, (c, ';'.join(modx), msk)))
                msk, badmod = [], False
                # get peptide sequences, charge and modifications
                modinfo, pepinfo = prd.getvaluesinbracket(line.rstrip(), '()')
                pcx = pepinfo.replace(' ','').split(':')[1].split('/')
                seq = pcx[0]
                c = int(pcx[1].split('_')[0])
                modx = []
                if modinfo:
                    for modstr, _ in modinfo:
                        k, aa, modname = tuple(modstr.split(','))
                        if modname not in 'CAM,Oxidation' or\
                           not (aa=='C' and modname=='CAM'):
                            badmod = True
                            modx = ''
                            break
                        if modname=='CAM':
                            modx.append('Carbamidomethyl(C)@%d'%(int(k)+1))
            else:
                if not line.rstrip(): continue
                try:
                    msk.append(tuple([float(xk) for xk in line.rstrip().split()[:2]]))
                except:
                    pass
    if msk and not badmod:
        mspectra.append((seq, (c, ';'.join(modx), msk)))
    return dict(mspectra)


def getmascotcsv(filename):
    """
    Get Mascot exported search results in ..csv file
    """

    def parsemod(params, pos, seq):
        if not pos: return ''
        bnterm = [ky for ky, (_, vk) in params.items() if vk=='N-term']
        mods = []
        if pos[0] in bnterm:
            mods.append('%s@%s'%params[pos[0]])
        for ij, vk in enumerate(pos[1]):
            if vk=='0': continue
            mods.append('%s(%s)@%d'%(params[vk][0], seq[ij], ij+1))
        return ';'.join(mods)
    
    res = {}
    with open(filename, 'r') as f:
        t, protein, t, tv, modparams = 0, {}, 0, False, {}
        for line in f:
            sline = line.rstrip().split(',')
            # get modification settings
            if line.startswith('"Variable modifications"'):
                tv = True
            elif line.startswith('"Search Parameters"'):
                tv = False
            elif tv:
                if line.startswith('"Identifier"'):
                    ix1 = line.rstrip().split(',').index('"Identifier"')
                    ix2 = line.rstrip().split(',').index('"Name"')
                elif line.rstrip():
                    bk = sline[ix2].split('"')[1].split()
                    bk2, _ = prd.getvaluesinbracket(bk[1], '()')
                    modparams[sline[ix1]] = (bk[0], bk2[0][0])
            # reading hits
            if line.startswith('prot_hit_num'):
                ix1 = [i for i,xk in enumerate(sline) if xk=='pep_exp_z'][0]
                ix2 = [i for i,xk in enumerate(sline) if xk=='pep_miss'][0]
                ix3 = [i for i,xk in enumerate(sline) if xk=='pep_score'][0]
                ix4 = [i for i,xk in enumerate(sline) if xk=='pep_seq'][0]
                ix5 = [i for i,xk in enumerate(sline) if xk=='pep_var_mod'][0]
                ix6 = [i for i,xk in enumerate(sline) if xk=='pep_scan_title'][0]
                ix7 = [i for i,xk in enumerate(sline) if xk=='prot_desc'][0]
                ix8 = [i for i,xk in enumerate(sline) if xk=='prot_score'][0]
                #ix9 = [i for i,xk in enumerate(sline) if xk=='prot_cover'][0]
                ix10 = [i for i,xk in enumerate(sline) if xk=='pep_query'][0]
                ix11 = [i for i,xk in enumerate(sline) if xk=='prot_acc'][0]
                ix12 = [i for i,xk in enumerate(sline) if xk=='pep_var_mod_pos'][0]
                t, nk = 1, len(sline)
            elif 'Queries' in line:
                res['unassigned'] = peptides
                t = 0
                break
            elif t and 'Peptide matches not assigned to protein hits' in line:
                protein['peptides'] = peptides
                res['proteins_%d'%(protein_no)] = protein.copy()
                peptides = []
            elif t and len(sline)>1 and sline[1]:
                bx = re.findall('"([^"]*)"', line.rstrip())
                pname = bx[1]   # protein name
                offset = pname.count(',')
                protein_no = int(sline[0])
                if protein_no>1:
                    protein['peptides'] = peptides
                    res['proteins_%d'%(protein_no-1)] = protein.copy()
                protein['accession'] = sline[ix11]
                protein['name'] = pname
                #protein['coverage'] = float(sline[ix9+offset])
                protein['score'] = float(sline[ix8+offset])
                if len(sline) == nk+offset and sline[ix4+offset]:
                    # parse modification
                    if sline[ix12+offset]:
                        modk = parsemod(modparams,
                                        sline[ix12+offset].split('.'),
                                        sline[ix4+offset])
                    else:
                        modk = ''
                    # peptide hit
                    peptides = [(sline[ix6+offset],
                                int(sline[ix10+offset]),
                                sline[ix4+offset],
                                modk,
                                int(sline[ix1+offset]),
                                float(sline[ix3+offset]),
                                int(sline[ix2+offset]))]
                else:
                    peptides = []
            elif t:
                if len(sline) == nk and sline[ix4]:
                    # parse modification
                    if sline[ix12]:
                        modk = parsemod(modparams,
                                        sline[ix12].split('.'),
                                        sline[ix4])
                    else:
                        modk = ''
                    # peptide hit
                    peptides.append((sline[ix6],
                                     int(sline[ix10]),
                                     sline[ix4],
                                     modk,
                                     int(sline[ix1]),
                                     float(sline[ix3]),
                                     int(sline[ix2])))
    return res


def getfastaseqs(fastafile):
    """
    Get protein sequences from fasta file
    """
    proteins = []
    with open(fastafile, 'r') as f:
        bx = []
        for line in f:
            if line.startswith('>'):
                if bx:
                    proteins.append((name, ''.join(bx)))
                bx = []
                name = line.rstrip()
            else:
                bx.append(line.rstrip())
        proteins.append((name, ''.join(bx)))
    return dict(proteins)


def getsimscoreini(spx, spt, seq, mod, c, targetmod):
    """
    Get similarity score between two mass spectrum
    """
    sqrt = np.sqrt
    ddir = r'D:\Data\PTN\MGF_complete'
    if isinstance(mod, str):
        mod = ';'.join(xk for xk in mod.replace(' ','').split(';') if not xk.startswith('No'))
        mod = ';'.join(xk.split('ProteinTerminal')[1] if xk.startswith('ProteinTerminal') else xk for xk in mod.split(';'))
        try:
            modx = prd.parsemodifications(mod, prd.PTMDB)
        except:
            return None
    else:
        modx = mod
    # load tandem mass spectrum from spectrum id1
##    spx = pms.centroidms(spx)
##    spx = prd.removetagfrag(spx)
##    if len(spx)<10: return None
    # load tandem mass spectrum from spectrum id2
##    spt = pms.centroidms(spt)
##    spt = prd.removetagfrag(spt)
##    if len(spt)<10: return None
    # spx
    qmx = annotes.calpepmass(seq, modx)
    theoionsx = prd.allions(qmx, seq, modx, c)
    ions = prd.simpleannotation(theoionsx, spx)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spx)]
    slx = sorted(prd.adaptivedenoise(spx, ionsx))
    ions1 = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    spxk = [spx[j] for j in slx]
    # spt
    if isinstance(targetmod, str):
        modx2 = [mk for mk in modx if not targetmod in mk]
    else:
        modx2 = [mk for mk in modx if not any(tk in mk for tk in targetmod)]
    qmx = annotes.calpepmass(seq, modx2)
    theoionsx = prd.allions(qmx, seq, modx2, c)
    ions = prd.simpleannotation(theoionsx, spt)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spt)]
    slx = sorted(prd.adaptivedenoise(spt, ionsx))
    ions2 = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    sptk = [spt[j] for j in slx]
    # similarity score
    mixs = prd.matchNonMod(spxk, ions1, sptk, ions2)
    n1 = sum(sqrt(spxk[i][1])*sqrt(sptk[j][1]) for i,j in mixs if i is not None and j is not None)
    m1 = sqrt(sum(spxk[i][1] for i,_ in mixs if i is not None))
    m2 = sqrt(sum(sptk[i][1] for _,i in mixs if i is not None))
    return n1/(m1*m2)
	
	
def getsimscore2(spx, spt, seq, mod, modn, c):
    """
    Get similarity score between two mass spectrum
    """
    sqrt = np.sqrt
    ddir = r'D:\Data\PTN\MGF_complete'
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
    qmx = annotes.calpepmass(seq, modx)
    theoionsx = prd.allions(qmx, seq, modx, c)
    ions = prd.simpleannotation(theoionsx, spx)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spx)]
    slx = sorted(prd.adaptivedenoise(spx, ionsx))
    ions1 = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    spxk = [spx[j] for j in slx]
    # spt
    if isinstance(modn, str):
        modn = ';'.join(xk for xk in modn.replace(' ','').split(';') if not xk.startswith('No'))
        modn = ';'.join(xk.split('ProteinTerminal')[1] if xk.startswith('ProteinTerminal') else xk for xk in modn.split(';'))
        try:
            modx2 = prd.parsemodifications(modn, prd.PTMDB)
        except:
            return None
    else:
        modx2 = modn
    qmx = annotes.calpepmass(seq, modx2)
    theoionsx = prd.allions(qmx, seq, modx2, c)
    ions = prd.simpleannotation(theoionsx, spt)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spt)]
    slx = sorted(prd.adaptivedenoise(spt, ionsx))
    ions2 = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    sptk = [spt[j] for j in slx]
    # similarity score
    mixs = prd.matchNonMod(spxk, ions1, sptk, ions2)
    n1 = sum(sqrt(spxk[i][1])*sqrt(sptk[j][1]) for i,j in mixs if i is not None and j is not None)
    m1 = sqrt(sum(spxk[i][1] for i,_ in mixs if i is not None))
    m2 = sqrt(sum(sptk[i][1] for _,i in mixs if i is not None))
    return n1/(m1*m2)


def getsimscore(rt1, rt2, id1, id2, seq, mod, c, targetmod):
    """
    Get similarity score between two mass spectrum
    """
    sqrt = np.sqrt
    ddir = r'D:\Data\PTN\MGF_complete'
    mod = ';'.join(xk for xk in mod.replace(' ','').split(';') if not xk.startswith('No'))
    mod = ';'.join(xk.split('ProteinTerminal')[1] if xk.startswith('ProteinTerminal') else xk for xk in mod.split(';'))
    try:
        modx = prd.parsemodifications(mod, prd.PTMDB)
    except:
        return None
    # load tandem mass spectrum from spectrum id1
    try:
        spx = prd.dtareader(os.path.join(ddir, '%s_MGFPeaklist\%s.dta'%(rt1, id1)))
    except:
        return None
    spx = pms.centroidms(spx)
    spx = prd.removetagfrag(spx)
    # load tandem mass spectrum from spectrum id2
    try:
        spt = prd.dtareader(os.path.join(ddir, '%s_MGFPeaklist\%s.dta'%(rt2, id2)))
    except:
        return None
    spt = pms.centroidms(spt)
    spt = prd.removetagfrag(spt)
    # spx
    qmx = annotes.calpepmass(seq, modx)
    theoionsx = prd.allions(qmx, seq, modx, c)
    ions = prd.simpleannotation(theoionsx, spx)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spx)]
    slx = sorted(prd.adaptivedenoise(spx, ionsx))
    ions1 = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    spxk = [spx[j] for j in slx]
    # spt
    if isinstance(targetmod, str):
        modx2 = [mk for mk in modx if not targetmod in mk]
    else:
        modx2 = [mk for mk in modx if not any(tk in mk for tk in targetmod)]
    qmx = annotes.calpepmass(seq, modx2)
    theoionsx = prd.allions(qmx, seq, modx2, c)
    ions = prd.simpleannotation(theoionsx, spt)
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spt)]
    slx = sorted(prd.adaptivedenoise(spt, ionsx))
    ions2 = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    sptk = [spt[j] for j in slx]
    # similarity score
    mixs = prd.matchNonMod(spxk, ions1, sptk, ions2)
    n1 = sum(sqrt(spxk[i][1])*sqrt(sptk[j][1]) for i,j in mixs if i is not None and j is not None)
    m1 = sqrt(sum(spxk[i][1] for i,_ in mixs if i is not None))
    m2 = sqrt(sum(sptk[i][1] for _,i in mixs if i is not None))
    return n1/(m1*m2)


def getshortseqs(idents, proteindb, modlib, length):
    """
    Get short sequences from protein sequence database proteindb.
    idents --- protein sequence database search identifications
    proteindb --- protein sequence database
    modlib --- targe modifications so that residues bearing modifications
               in the list will locate at the center of short sequence
    length --- number of residues at each site of modified residue
    """
    shortseqs = []
    modseqs = [prd.combinemod(x[0], x[1]) for x in idents]
    for x in set(modseqs):
        j = modseqs.index(x)
        x2 = idents[j]
        # get indices from database
        dbseqs = {}
        for dbseq in proteindb.values():
            if x2[0] in dbseq:
                dbseqs[dbseq] = [i for i in range(len(dbseq)) if dbseq[i:].startswith(x2[0])]
        # get short sequences
        modx = [xk for xk in x2[1].replace(' ','').split(';')
                if any(xb in xk for xb in modlib)]
        for mk in modx:
            try:
                k = int(mk.split('@')[1])-1
            except:
                print(mk)
                k = 0
            for dbseq, ixs in dbseqs.items():
                for i in ixs:
                    istart, iend = i-length+k, i+k+length+1
                    if istart<0:
                        shortseqs.append('X'*abs(istart)+dbseq[:iend])
                    elif iend>len(dbseq):
                        shortseqs.append(dbseq[istart:]+'X'*(i+k+length-len(dbseq)))
                    else:
                        shortseqs.append(dbseq[istart:iend])
    return list(set(shortseqs))
        

def getvariables(spectrum, seq, modx, c):
    """
    Get variables
    """
    qmx = annotes.calpepmass(seq, modx)
    theoionsx = prd.allions(qmx, seq, modx, c)
    ions = prd.simpleannotation(theoionsx, spectrum)
    # adaptive denoise
    iixt = set(vt[0] for vt in ions.values())
    ionsx = [(xk[0], True) if j in iixt else (xk[0], False) for j, xk in enumerate(spectrum)]
    slx = sorted(prd.adaptivedenoise(spectrum, ionsx))
    ions = dict((ky, (bisect_left(slx, ions[ky][0]), ions[ky][2])) for ky in ions.keys() if ions[ky][0] in slx)
    spxk = [spectrum[j] for j in slx]
    m = max(xj[1] for xj in spxk)
    spx2 = [[xj[0], xj[1]/m] for xj in spxk]
    vx = prd.getvariables(seq, modx, c, ions, spx2, 'Nitro', 0.2)
    tmc = sum(xk in 'RK' and seq[j+1]!='P' for j, xk in enumerate(seq[:-1]))
    vx.append(tmc)
    return vx
