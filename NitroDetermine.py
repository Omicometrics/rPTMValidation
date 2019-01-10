"""
NitroDetermine class for validation of assignment of mass spectra
to nitrated peptides.
Usage:
    p = NitroDetermine(sequence, modifications)
    inputs are:
    ::sequence
    Peptide sequence.
    ::modifications
    Modification information in this sequence, current class accepts
    two types of formats for describing modifications. The first one
    is,
            "mod_name@site_in_seq"
        where 'mod_name' must be 'Interim name' in UNIMOD
        (http://www.unimod.org). So please note that the name is
        probably different with that is commonly used in practice,
        for example, oxidation of Met should be 'Hydroxylation' since
        'Oxidation' and 'Hydroxylation' are combined in Unimod.
        'site_in_seq' is the location of modified amino acid residue
        counted from N-terminus.
        If no modification exists, input as None or [] or ''.
        If multiple modifications exist, input as
            ["mod_name1@site_in_seq1","mod_name2@site_in_seq2",...]
        or
            "mod_name1@site_in_seq1;mod_name2@site_in_seq2;..."
    The other is integrated into the peptide sequence directly, with
    mass of modified residue included into a square bracket, all of
    which is located adjacent the residue that is modified. For N- or
    C- terminal modifications, use n[x] and c[x], as these
    modifications do not consider what the residues are. For
    example,
        'n[305]M[147]TSSVAPASQ[129]RSIRLR[170]'
    will have a N-terminal modification of 304 Da which can probably
    be iTRAQ 8-plex labeling, oxidation of methionine, Gln->Glu and
    methylation of arginine. Why N-terminal modification is presented
    as is can be referred to a poster in SPCTool Google Groups
    (https://groups.google.com/forum/#!topic/comet-ms/EA4TqXy-W8A).
    These two types of formats are inspired by AB SCIEX ProteinPilot
    search results and PepXML identification results open file.
"""
from annotation_spectrum import *
import re, scorer

_PTMDB = getmodificationdb()


class NitroDetermine(object):
    """
    Main class
    """
    def __init__(self, sequence, modification, charge, MTYPE='mono', MODDB=_PTMDB, PARSEMOD=True):
        mods = modification
        if PARSEMOD:
            mods = parsemodifications(mods, MODDB, MTYPE)
        self.__seq = sequence.strip()
        self.__mods = mods
        self.__modinput = mods
        self.__seqmass = calpepmass(sequence.strip(), mods, MTYPE)
        self.__charge = charge
        self.__mtype = 'mono'
        self.__hydrogenmass = 1.0073 if MTYPE=='mono' else 1.0078
        self.__watermass = 18.006067
        
    def getsequence(self, sequencec):
        """
        Get pure peptide sequence from complex sequence expression
        in which generally modifications are included.
        """
        # remove all contents in parentheses, square brackets, curly
        # brackets, since sometimes the information of, for example,
        # the formula of modications, can be included in the sequence
        # in sequare brackets, which can add characters that do not
        # belong to the sequence
        def removebks(string, bksym):
            if not bksym[0] in string:
                return string
            sstr = string.split(bksym[0])
            return sstr[0]+''.join([sx.split(bksym[1])[1] for sx in sstr[1:]])
        seq_t = removebks(sequencec, '()')
        seq_t = removebks(seq_t, '[]')
        seq_t = removebks(seq_t, '{}')
        
        return ''.join(re.findall('[ACDEFGHIKLMNPQRSTVWY]',seq_t))
    
    def theoreticalions(self, NEUTRAL=False):
        """
        Get theoretical ions according to the peptide sequence and
        input modifications, if the sequential ions with neutral losses
        are required to output, set NEUTRAL to True.
        """
        bs, ys, pmass = generatetheoions(self.__seqmass[1],
                                 ntermmass=self.__seqmass[0],
                                 ctermmass=self.__seqmass[2])
        ions = generateions(bs, ys, pmass, self.__charge)

        yions = [ion for ion in ions if ion[1].startswith('y') and '-' not in ion[1]]
        bions = [ion for ion in ions if ion[1].startswith('b') and '-' not in ion[1]]
        pions = [ion for ion in ions if ion[1].startswith('p') and '-' not in ion[1]]
        ynions = [ion for ion in ions if ion[1].startswith('y') and '-' in ion[1]]
        bnions = [ion for ion in ions if ion[1].startswith('b') and '-' in ion[1]]
        pnions = [ion for ion in ions if ion[1].startswith('p') and '-' in ion[1]]
        
        if not NEUTRAL:
            return yions+bions+pions
        return yions+bions+pions+ynions+bnions+pnions

    def immonium(self):
        """
        Immonium ion of current sequence
        """
        seq = self.__seq
        seqlist, l = list(seq), len(seq)
        # add modification information to the sequences
        if self.__mods:
            for _, j, modname in self.__mods:
                if not isinstance(j, str):
                    seqlist[j-1] = '%s[%s]'%(seqlist[j-1], modname)
        # get unique sequence
        uniqueseq = {}
        for i, r in enumerate(seqlist):
            if i==0 and self.__seqmass[0]: continue
            if i==l-1 and self.__seqmass[2]: continue
            if r not in uniqueseq:
                uniqueseq[r] = i
        # immonium ions
        immmz = list(generateimmoniums(self.__seqmass[1]))
        # add immonium ions into ion list
        immions = []
        for r, i in uniqueseq.items():
            qx = 'imm(%s)'%('%s*'%seq[i] if '[' in r else r)
            immions.append((immmz[i], qx))

        return immions

    def masstype(self):
        """
        Type of mass used in calculating properties of peptide,
        for example, peptide mass, theoretical fragment ion m/z,
        modification mass, etc.
        """
        print('Monoisotopic Mass' if self.__mtype == 'mono' else 'Average Mass')
    
    def sequence(self):
        """
        Return peptide sequence.
        """
        return self.__seq
    
    def modifications(self):
        """
        Return modification information.
        """
        if not self.__modinput:
            print("No modification(s) exist.")
            return None
        return self.__mods
    
    def precursormass(self):
        """
        Return mass of the input peptide.
        """
        nm, seqmass, cm = tuple(self.__seqmass)
        if nm and cm:
            terminus_mass = nm+cm+self.__hydrogenmass
        elif nm and not cm:
            terminus_mass = nm+self.__watermass
        elif not nm and cm:
            terminus_mass = self.__hydrogenmass+cm
        else:
            terminus_mass = self.__watermass
        return sum(seqmass)+terminus_mass
    
    def charge(self):
        """ charge state of the peptide """
        return self.__charge

    def mz(self):
        """ Mass to charge ratio """
        return self.precursormass()/self.__charge+self.__hydrogenmass

    def annotations(self, spectrum, MZTOL=1., TOLUNIT="Da"):
        """
        Annotation of input tandem mass spectrum.
        Inputs:
            ::spectrum:
            Tandem mass pectrum input for annotation.
            ::MZTOL:
            m/z tolerance for matching fragment ions
            ::TOLUNIT:
            Unit for defining exact m/z tolerance in match of fragment ions.
            Valid values are {'Da', 'ppm'}.
        """
        bs, ys, pmass = generatetheoions(self.__seqmass[1],
                                          ntermmass=self.__seqmass[0],
                                          ctermmass=self.__seqmass[2])
        ions = generateions(bs, ys, pmass, self.__charge)
        seq = self.__seq
        # immonium ions
        immions = self.immonium()
        ions += immions
        # p-iTRAQ8
        pm = self.precursormass()
        if self.__mods:
            for mj,_,x in self.__mods:
                if x=='iTRAQ8plex':
                    ions.append((pm-mj+self.__hydrogenmass, 'p-iT8[+]'))
                    break
        # annotation ...
        ionnames, ionerr = annotatespectrum(spectrum, ions, MZTOL, TOLUNIT)
        ionnames, ionerr = neutralassign(spectrum,ionnames, ionerr)
        ionnames = removerepions(ionnames, ionerr)
        # consider the internal fragments and immonium ions, assign
        # the unannotated fragments to these ions
        # .. get unannotated peaks
        unx = [i for i, name in enumerate(ionnames) if not name]
        spec_unannotated = [spectrum[i] for i in unx]
        # .. assign internal fragments and immonium ions to the peaks
        if spec_unannotated:
            internalions = generateinternals(self.__seqmass[1])
            ions_add = []
            for (i,j), internalmz in internalions:
                ions_add.append((internalmz, self.__seq[i:j]))
            names_add, err_add = annotatespectrum(spec_unannotated, ions_add,
                                                MZTOL, TOLUNIT)
            # .. add the internal fragments into annotation list
            names_add = removerepions(names_add, err_add)
            for j, i in enumerate(unx):
                if names_add[j]:
                    ionnames[i] = names_add[j]
        
        return [(peak[0], ionnames[i]) for i, peak in enumerate(spectrum)]

    def score(self, spectrum, MZTOL=1.):
        """
        Score of the nitrated peptides to the spectrum.
        """
        sites = checknitro(self.__modinput)
        bs, ys, pmass = theoions(self.__seqmass[1],
                                 ntermmass=self.__seqmass[0],
                                 ctermmass=self.__seqmass[2])
        ions = generateions(bs, ys, pmass, self.__charge)
        # get codes
        costs, cost_clean = [], 0.
        n = len(self.__seq)
        for site in sites:
            cost_site = []
            for c in range(self.__charge):
                ionc = '[+]' if c==0 else '[%d+]'%(c+1)
                # b ions
                ions4score = [ion for ion in ions
                              if ionc in ion[1] and '-' not in ion[1]
                              and ion[1].startswith('b')]
                s1 = scorer.score(spectrum, ions4score, site, MATCHTOL=MZTOL)

                # y ions
                ions4score = [ion for ion in ions
                              if ionc in ion[1] and '-' not in ion[1]
                              and ion[1].startswith('y')]
                s2 = scorer.score(spectrum, ions4score, n-site+1, MATCHTOL=MZTOL)

                cost_site.append(min(s1, s2))
            if cost_site:
                costs.append(min(cost_site))
            else:
                costs.append(3.15)

        return min(costs)#, cost_clean

    def score_all(self, spectrum, MZTOL=1.):
        """
        All score of the nitrated peptides to the spectrum.
        """
        sites = checknitro(self.__modinput)
        bs, ys, pmass = theoions(self.__seqmass[1],
                                 ntermmass=self.__seqmass[0],
                                 ctermmass=self.__seqmass[2])
        ions = generateions(bs, ys, pmass, self.__charge)
        # get codes
        costs = []
        n = len(self.__seq)
        for site in sites:
            cost_site = []
            for c in range(self.__charge):
                ionc = '[+]' if c==0 else '[%d+]'%(c+1)
                # b ions
                ions4score = [ion for ion in ions
                              if ionc in ion[1] and '-' not in ion[1]
                              and ion[1].startswith('b')]
                s_b = scorer.score_all(spectrum, ions4score, site, MATCHTOL=MZTOL)

                # y ions
                ions4score = [ion for ion in ions
                              if ionc in ion[1] and '-' not in ion[1]
                              and ion[1].startswith('y')]
                s_y = scorer.score_all(spectrum, ions4score, n-site+1, MATCHTOL=MZTOL)

                cost_site.append(min(s_b, s_y))
                
            costs.append(min(cost_site))

        return min(costs)
