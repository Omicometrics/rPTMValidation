from operator import itemgetter
import re

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


def getmascotdatres(datfile):
    """
    Get results from Mascot dat files
    """
    # parse modification string
    def parsemodstr(modstring):
        """
        parse modification string
        """
        dk, modinfo = tuple(modstring.rstrip().split('='))
        masstr, modname = tuple(modinfo.split(','))
        modresidue, modk = getvaluesinbracket(
            modname.replace(' ',''),'()')
        return dk[-1], float(masstr), modk, modresidue[0][0]
    
    def getmass(string):
        """
        parse summary string to get mass and assigned charge state
        """
        item, expinfo = tuple(string.split('='))
        mz, c = tuple(expinfo.split(','))
        return item.lstrip('qexp'), float(mz), int(c[0])
        
    def parseidentification(string, varmods, fixmods):
        """
        parse identification string
        """
        split_string = string.split(',')
        modstring = split_string[6]
        # peptide sequence
        pepstr = split_string[4]
        # Ion Score
        score = float(split_string[7])
        # delta between theoretical and experimentally determined peptide
        # mass
        delta = float(split_string[2])
        # modifications
        mods, termmod, modified_nterm, modified_cterm = [], [], False, False
        for i,x in enumerate(modstring):
            if x=='0': continue
            # value of 'X' indicates this is the modification site during
            # tolerant search
            if x=='X':
                mods.append((None, i, None))
                if i==0:
                    modified_nterm = True
                elif i==len(modstring)-1:
                    modified_cterm = True
                continue
            # normal modifications
            if i==0:
                termmod.append((varmods[x]['mass'], 'nterm', varmods[x]['name']))
            elif i==len(modstring)-1:
                termmod.append((varmods[x]['mass'], 'cterm', varmods[x]['name']))
            else:
                mods.append((varmods[x]['mass'], i, varmods[x]['name']))
        for nk in fixmods.keys():
            if fixmods[nk]['residues']=='N-term':
                if not modified_nterm:
                    termmod.append((
                        fixmods[nk]['mass'], 'nterm', fixmods[nk]['name']))
                continue
            if fixmods[nk]['residues']=='C-term':
                if not modified_cterm:
                    termmod.append((
                        fixmods[nk]['mass'], 'cterm', fixmods[nk]['name']))
                continue
            mods += [(fixmods[nk]['mass'], i+1, fixmods[nk]['name'])
                for i,x in enumerate(pepstr) if x==fixmods[nk]['residues']]
        try:
            mods = sorted(mods, key=itemgetter(1))
        except TypeError:
            print(mods, string)
            raise
        mods = termmod + mods
        return pepstr, mods, score, delta

    # =========================================================================
    # get identifications
    res = {}
    varmods, fixmods ={}, {}
    error_tol_search, decoy_search, content = False, False, None
    with open(datfile, 'r') as f:
        for line in f:
            tline = line.rstrip()
            if not tline or line.startswith('--'): continue
            # headers
            if tline.startswith('Content-Type'):
                val = re.findall('"(.*)"', tline)
                if not val: continue
                content = val[0]
                if content.endswith('summary') or content.endswith('peptides'):
                    contentinfo = content.split('_')
                    ky = 'target'
                    if '_' in content: ky, content = tuple(contentinfo)
                elif content.startswith('query'):
                    queryno = content.lstrip('query')
                    ky, content = 'spid', 'query'
                continue
            # modification parameters
            if content == 'parameters':
                if tline.startswith('ERRORTOLERANT'):
                    tid = tline.lstrip('ERRORTOLERANT=')
                    if tid:
                        error_tol_search = True
                elif tline.startswith('DECOY'):
                    did = tline.lstrip('DECOY=')
                    if did:
                        decoy_search = True
            # masses to get modification settings
            elif content == 'masses':
                if tline.startswith('delta'):
                    nk, mass, modname, rk = parsemodstr(tline.rstrip())
                    varmods[nk] = {}
                    varmods[nk]['mass'] = float(mass)
                    varmods[nk]['name'] = modname
                    varmods[nk]['residues'] = rk
                elif tline.startswith('FixedMod') and\
                    not tline.startswith('FixedModResidues'):
                    nk, mass, modname, rk = parsemodstr(tline.rstrip())
                    fixmods[nk] = {}
                    fixmods[nk]['mass'] = float(mass)
                    fixmods[nk]['name'] = modname
                    fixmods[nk]['residues'] = rk
            # PSMs
            else:
                if content=='summary':
                    if not tline.startswith('qexp'): continue
                    queryno, mz, c = getmass(tline)
                    if queryno is not None:
                        try:
                            res[queryno][ky] = {}
                        except KeyError:
                            res[queryno] = {}
                            res[queryno][ky] = {}
                        res[queryno][ky]['mz'] = mz
                        res[queryno][ky]['charge'] = c
                elif content=='peptides':
                    if tline.endswith('-1'): continue
                    sline = tline.split('=')
                    # peptide identifications
                    if sline[0].count('_')==1:
                        queryno, rank = re.findall('q(.+?)_p(.+)',sline[0])[0]
                        pepstring, proteinstring = tuple(sline[1].split(';'))
                        pepseq, mods, score, delta = parseidentification(
                            pepstring, varmods, fixmods)
                        proteins = tuple(re.findall('"([^"]*)"', proteinstring))
                        if int(rank)==1:
                            res[queryno][ky]['peptides'] = {}
                        res[queryno][ky]['peptides'][rank] = {}
                        res[queryno][ky]['peptides'][rank]['sequence']=pepseq
                        res[queryno][ky]['peptides'][rank]['modifications']=tuple(mods)
                        res[queryno][ky]['peptides'][rank]['ionscore']=score
                        res[queryno][ky]['peptides'][rank]['deltamass']=delta
                        res[queryno][ky]['peptides'][rank]['proteins']=proteins
                    # error tolerant search modifications
                    elif error_tol_search:
                        if not sline[0].endswith('et_mods'): continue
                        queryno, rank = tuple([xk[1:]
                            for xk in sline[0].split('_')[:2]])
                        ermods = sline[1].split(',')
                        m = float(ermods[0])
                        modname, modr = re.findall('(.+?) \((.+?)\)',ermods[2])[0]
                        try:
                            seq = res[queryno][ky]['peptides'][rank]['sequence']
                        except KeyError:
                            print(tline, res[queryno])
                            raise
                        mods2 = []
                        for xk in res[queryno][ky]['peptides'][rank]['modifications']:
                            if xk[0] is not None:
                                mods2.append(xk)
                                continue
                            i = xk[1]
                            # N-terminus
                            if i==0:
                                mods2.append((m, 'nterm', modname))
                            # C-terminus
                            elif i==len(seq)+1:
                                mods2.append((m, 'cterm', modname))
                            else:
                                if modr[0][0]==seq[i-1]:
                                    mods2.append((m, i, modname))
                        res[queryno][ky]['peptides'][rank]['modifications']=tuple(mods2)
                elif content=='query':
                    # spectrum ID
                    if tline.startswith('title'):
                        spid = tline.split('=')[1].replace('%3a',':').\
                            replace('%2e','.').replace('a%22',' ').\
                            replace('%20',' ').replace('%22','"')
                        res[queryno]['spectrumid'] = spid
    return res


def getpepxml(filename):
    """
    Read Comet search results (pep.xml)
    """
    res = []
    with open(filename, 'r') as f:
        for line in f:
            sx = dict(re.findall(' (.+?)\="([^"]*)"', line.rstrip().lstrip()))
            if '<spectrum_query' in line:
                rid = sx['spectrum'].split('.')
                rawfile = rid[0]
                scannum = int(rid[1])
                ck = int(sx['assumed_charge'])
                spectrumid = sx['spectrumNativeID']
                hits = []
            elif '<search_hit' in line:
                modj, modnterm, hitscore = [], [], {}
                pk = sx['peptide']
                cix = [j+1 for j, xk in enumerate(pk) if xk=='C']
                nk = 'decoy' if sx['protein'].startswith('DECOY_') else 'normal'
                # get rank of the hit
                hitrank = int(sx['hit_rank'])
            elif '<modification_info' in line:
                modnterm = []
                if 'mod_nterm_mass' in sx:
                    mnterm = float(sx['mod_nterm_mass'])
                    if int(mnterm)==305:
                        modnterm.append((mnterm,'N-term','iTRAQ8plex'))
                    elif int(mnterm)==58:
                        modnterm.append((mnterm,'N-term','Carbamidomethyl'))
                    elif int(mnterm)==44:
                        modnterm.append((mnterm,'N-term','Carbamyl'))
            elif '<mod_aminoacid_mass' in line:
                j = int(sx['position'])
                try:
                    mmod = float(sx['static'])
                except KeyError:
                    mmod = float(sx['variable'])
                if abs(mmod-44.98)<1:
                    modj.append((mmod, j, 'Nitro'))
                elif abs(mmod-15.99)<1:
                    modj.append((mmod, j, 'Oxidation'))
                elif abs(mmod+17.03)<1:
                    modj.append((mmod, j, 'Gln->pyro-Glu'))
                elif abs(mmod-0.98)<1:
                    modj.append((mmod, j, 'Deamidated'))
                elif abs(mmod-304.2054)<1:
                    modj.append((mmod, j, 'iTRAQ8plex'))
                elif abs(mmod-57.02) < 1:
                    modj.append((mmod, j, 'Carbamidomethyl'))
            elif '<search_score' in line:
                hitscore[sx['name']] = float(sx['value'])
            elif '</search_hit>' in line:
                modj = sorted(modj, key=itemgetter(1))
                if modnterm:
                    modj = modnterm+modj
                hits.append((hitrank, pk, tuple(modj), ck, hitscore, nk))
            elif '</spectrum_query>' in line:
                if hits:
                    res.append((rawfile, scannum, spectrumid, hits))
                hits = []
    return res