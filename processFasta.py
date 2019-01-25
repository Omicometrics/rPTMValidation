"""
Process protein sequence database (in ..fasta file format) to
generate, such as decoy database for peptide identifications.
Enzymes and cleavage rules are set according to Mascot document at
http://www.matrixscience.com/help/enzyme_help.html.
Note that the 'none' type cleavage is not supported currently.
"""
import re
		

ENZYME = {'Trypsin': {'site':('KR',), 'except':('P',), 'terminal':('C',)},
	'Trypsin/P': {'site':('KR',), 'except':('',), 'terminal':('C',)},
	'Arg-C': {'site':('R',), 'except':('P',), 'terminal':('C',)},
	'Asp-N': {'site':('BD',), 'except':('',), 'terminal':('N',)},
	'Asp-N_ambic': {'site':('DE',), 'except':('',), 'terminal':('N',)},
	'Chymotrypsin': {'site':('FYWL',), 'except':('P',), 'terminal':('C',)},
	'CNBr': {'site':('M',), 'except':('',), 'terminal':('C',)},
	'CNBr+Trypsin':{'site': ('M','KR'),
		'except':('','P'), 'terminal': ('C','C')},
	'Formic_acid': {'site':('D',), 'except':('',), 'terminal':('C',)},
	'Lys-C': {'site':('K',), 'except':('P',),'terminal':('C',)},
	'Lys-C/P': {'site':('K',), 'except':('',), 'terminal':('C',)},
	'LysC+AspN':{'site':('K', 'DB'),
		'except':('P','P'), 'terminal':('C', 'N')},
	'Lys-N': {'site':('K',), 'except':('',), 'terminal':('N',)},
	'PepsinA': {'site':('FL',), 'except':('',), 'terminal':('C',)},
	'semiTrypsin': {'site':('KR',), 'except':('P',), 'terminal':('C',)},
	'TrypChymo': {'site':('FYWLKR',), 'except':('P',), 'terminal':('C',)}, 
	'TrypsinMSIPI': {'site':('KR',), 'except':('P',), 'terminal':('C',)},
	'TrypsinMSIPI/P': {'site':('KR',), 'except':('',), 'terminal':('C',)},
	'V8-DE': {'site':('BDEZ',), 'except':('P',), 'terminal':('C',)},
	'V8-E': {'site':('EZ',), 'except':('P',), 'terminal':('C',)}
	}
RESIDUES = set('ACDEFGHIKLMNPQRSTVWY')
	
	
class proteolysis(object):
	def __init__(self, enzyme):
		"""
		Enzyme project using the proteolytic cleavage rules set up
		in proteolysis dictionary in ENZYME, according to the input
		"enzyme".
		Argument
		- enzyme: proteolysis, if not in the dictionary, raise an exception
		"""
		self.enzyme = enzyme
		self.__check()
		self.__cleavagesite = ENZYME[enzyme]['site']
		self.__removenaa()
		self.__exceptions = ENZYME[enzyme]['except']
		self.__terminal = ENZYME[enzyme]['terminal']
		self.__parserules()
		
	def __check(self):
		"""
		If not in the enzyme dictionary, raise the key error.
		"""
		if not self.enzyme in ENZYME:
			raise KeyError('Undefined or unsupported enzyme type!')
			
	def __removenaa(self):
		"""
		Remove unrecognized residues not any one of 20 common residues.
		"""
		sites = [''.join(set(sk)&RESIDUES) for sk in self.__cleavagesite]
		if not any(len(sk)>0 for sk in sites):
			raise KeyError('Unsupported cleavage sites using' +\
				' the enzyme %s!'%self.enzyme)
		self.__cleavagesite = tuple(sites)
	
	def __parserules(self):
		"""
		Parse cleavage rules, and remove the one if the cleavage
		sites not exist in the protein sequence.
		"""
		enzrules, csx, excepts, termins, siteterminal = [], [], [], [], {}
		for i,sk in enumerate(self.__cleavagesite):
			if not sk: continue
			csx.append(sk)
			excepts.append(self.__exceptions[i])
			termins.append(self.__terminal[i])
			# set up string split rule
			rulej = r'([%s])'%sk
			if self.__exceptions[i]:
				rulej = r'(?<![%s])'%self.__exceptions[i]+rulej\
					if self.__terminal[i]=='N' else\
					rulej+r'(?![%s])'%self.__exceptions[i]
			enzrules.append(rulej)
			# get the combine direction
			for rk in sk:
				siteterminal[rk] = self.__terminal[i]
		self.__cleavagesite = tuple(csx)
		self.__exceptions = tuple(excepts)
		self.__terminal = tuple(termins)
		self.proteolyticrules = '|'.join(enzrules)
		self.__siteterminal = siteterminal
			
	def cleavage(self, sequence, numbermissed=1, lenrange=(7,60)):
		"""
		Cleavage of the input sequence using the constructed enzyme
		object, with number of missed cleavage allowed.
		Arguments
		- sequence: protein sequence in ..fasta file
		- numbermissed: number of missed cleavage allowed
		"""
		minlen, maxlen = tuple(lenrange)
		# split the sequence according to the cleavage rules of the
		# enzyme
		splittedseq = [sk for sk in re.split(self.proteolyticrules,sequence)
			if sk]
		n = len(splittedseq)
		if n==1:
			splittedseq = [x for x in splittedseq if maxlen>=len(x)>=minlen
				and RESIDUES.issuperset(x)]
			return tuple(splittedseq)
		# get all peptides with zero missed cleavage
		combinedpeps = []
		for i in range(n):
			nmk, j0, ci = 0, i, splittedseq[i]
			try:
				cterm = self.__siteterminal[ci]
			except KeyError:
				# the last splitted sequence
				if i==n-1:
					try:
						if self.__siteterminal[splittedseq[i-1]]=='C':
							combinedpeps.append(splittedseq[i])
					except KeyError:
						break
				else:
					continue
			# set up initial peptide sequence for searching next
			# sequence if number of missed cleavage larger than 0
			if cterm == 'C':
				if i==0:
					sk = ci
				else:
					cj = splittedseq[i-1]
					sk = cj+ci
					if splittedseq[i-1] in self.__siteterminal and\
						self.__siteterminal[cj]==cterm:
						nmk += 1
			elif cterm == 'N' and i<n-1:
				sk = ci+splittedseq[i+1]
				j0 += 1
				if splittedseq[i+1] in self.__siteterminal and\
					self.__siteterminal[splittedseq[i+1]]==cterm:
					nmk += 1
			# no missed cleavage
			combinedpeps.append(sk)
			# get peptides with larger number of missed cleavage
			if nmk==numbermissed: continue
			for j in range(j0+1,n):
				cj = splittedseq[j]
				if cj in self.__siteterminal:
					if self.__siteterminal[cj]=='C':
						sk += cj
						nmk += 1
						if nmk == numbermissed: break
					else:
						nmk += 1
						if nmk > numbermissed: break
						sk += cj
				else:
					sk += cj
			combinedpeps.append(sk)
		return tuple([x for x in combinedpeps if maxlen>=len(x)>=minlen
			and RESIDUES.issuperset(x)])
		

def getsequence(fastafile):
	"""
	Get sequence from the input protein sequence database file.
	"""
	subseqs = []
	for line in fastafile:
		if line.startswith('>'):
			if subseqs:
				yield title, ''.join(subseqs)
			title = line.rstrip()
			subseqs = []
		else:
			subseqs.append(line.rstrip())
	if subseqs:
		yield title, ''.join(subseqs)
		
		
def generatedecoys(seq):
	"""
	Generate decoy protein sequences by reversing.
	"""
	return seq[::-1]
		

# def readfasta(fastafile):
	# """
	# Get sequences from fasta file.
	# """
	# sequences = []
	# with open(fastafile, 'r') as f:
		# n = 0
		# for title, seq in getsequence(f):
			# n += 1
			# sequences.append(seq)
			# if n==1000:
				# pass
				
	# return sequences
