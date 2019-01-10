from csv import DictReader
import process_rawdata as prd

AARES = {'G': {'mono': 57.02146, 'avg': 57.052, 'formula': {'C': 2, 'H': 3, 'N': 1, 'O': 1}},
         'A': {'mono': 71.03711, 'avg': 71.078, 'formula': {'C': 3, 'H': 5, 'N': 1, 'O': 1}},
         'S': {'mono': 87.03203, 'avg': 87.078, 'formula': {'C': 3, 'H': 5, 'N': 1, 'O': 2}},
         'P': {'mono': 97.05276, 'avg': 97.117, 'formula': {'C': 5, 'H': 7, 'N': 1, 'O': 1}},
         'V': {'mono': 99.06841, 'avg': 99.133, 'formula': {'C': 5, 'H': 9, 'N': 1, 'O': 1}},
         'T': {'mono': 101.04768, 'avg': 101.105, 'formula': {'C': 4, 'H': 7, 'N': 1, 'O': 2}},
         'C': {'mono': 103.00918, 'avg': 103.144, 'formula': {'C': 3, 'H': 5, 'N': 1, 'O': 1, 'S': 1}},
         'I': {'mono': 113.08406, 'avg': 113.160, 'formula': {'C': 6, 'H': 11, 'N': 1, 'O': 1}},
         'L': {'mono': 113.08406, 'avg': 113.160, 'formula': {'C': 6, 'H': 11, 'N': 1, 'O': 1}},
         'N': {'mono': 114.04292, 'avg': 114.104, 'formula': {'C': 4, 'H': 6, 'N': 2, 'O': 2}},
         'D': {'mono': 115.02693, 'avg': 115.089, 'formula': {'C': 4, 'H': 5, 'N': 1, 'O': 3}},
         'Q': {'mono': 128.05857, 'avg': 128.131, 'formula': {'C': 5, 'H': 8, 'N': 2, 'O': 2}},
         'K': {'mono': 128.09495, 'avg': 128.174, 'formula': {'C': 6, 'H': 12, 'N': 2, 'O': 1}},
         'E': {'mono': 129.04258, 'avg': 129.116, 'formula': {'C': 5, 'H': 7, 'N': 1, 'O': 3}},
         'M': {'mono': 131.04048, 'avg': 131.198, 'formula': {'C': 5, 'H': 9, 'N': 1, 'O': 1, 'S': 1}},
         'H': {'mono': 137.05891, 'avg': 137.142, 'formula': {'C': 6, 'H': 7, 'N': 3, 'O': 1}},
         'F': {'mono': 147.06841, 'avg': 147.177, 'formula': {'C': 9, 'H': 9, 'N': 1, 'O': 1}},
         'R': {'mono': 156.10110, 'avg': 156.188, 'formula': {'C': 6, 'H': 12, 'N': 4, 'O': 1}},
         'Y': {'mono': 163.06332, 'avg': 163.170, 'formula': {'C': 9, 'H': 9, 'N': 1, 'O': 2}},
         'W': {'mono': 186.07931, 'avg': 186.213, 'formula': {'C': 11, 'H': 10, 'N': 2, 'O': 1}}
         }


def parsemodformula(modinfo, PTMDB=prd.PTMDB, mtype='mono'):
    """
    Parse formula of modifications
    """
    m, formula = [], {}
    masskey = 'Monoisotopic mass' if mtype=='mono' else 'Average mass'
    for modi in modinfo.split(';'):
        mx = modi.strip().split('@')[0]
        if not mx: continue
        # get modification formula
        if '(' in mx:
            j = max(i for i,x in enumerate(mx) if x=='(')
            mx = mx[:j]

        # get formula
        tx2 = mx.replace(' ','').lower()
        if tx2.startswith('delta'):
            s = mx.split(':')[1].replace(')', ') ').rstrip()
        else:
            if mx in PTMDB['PSI-MS Name']:
                i = PTMDB['PSI-MS Name'].index(mx)
            elif mx in PTMDB['Interim name']:
                i = PTMDB['Interim name'].index(mx)
            else:
                t = False
                for i, sk in enumerate(PTMDB['Description']):
                    if sk.replace(' ','').lower() == mx:
                        t = True
                        break
                if not t:
                    raise NameError('Unrecognizable modification: %s.'%tx)
            s = PTMDB['Composition'][i]
        # get formula
        sites, elex = prd.getvaluesinbracket(s,'()')
        # .. fill the lost number
        sites2, elex2 = [], elex.split()
        if len(sites2)<len(elex2):
            bix = [i for i,xk in enumerate(elex) if not xk.rstrip()]
            bix.append(len(elex))
            for i in bix:
                if not any(i==ji for _,ji in sites):
                    sites2.append(('1', i))
                else:
                    sites2 += [jk for jk in sites if jk[1]==i]
                
        for (nk, _), ej in zip(sites2, elex2):
            try:
                formula[ej] += int(nk)
            except:
                formula[ej] = int(nk)
    return sum(prd.MELEMENT[ky][mtype]*vk
               for ky, vk in formula.items()), formula


def getformula(seq, mods):
    """
    Get formula of the peptide sequence
    """
    m, modfm = parsemodformula(mods, mtype='mono')
    # sequence formula
    seqfm = {}
    for aa in seq:
        for ky in AARES[aa]['formula'].keys():
            try:
                seqfm[ky] += AARES[aa]['formula'][ky]
            except:
                seqfm[ky] = AARES[aa]['formula'][ky]
    # combine sequence and modification formulas
    fms = modfm.copy()
    for ky, vk in seqfm.items():
        try:
            fms[ky] += vk
        except:
            fms[ky] = vk
    # add a water
    fms['H'] += 2
    fms['O'] += 1
    print((sum(prd.MELEMENT[ky]['mono']*vk for ky, vk in fms.items())))
    return ' '.join('%s%d'%(ky, vk) for ky, vk in fms.items())
