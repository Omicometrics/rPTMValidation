"""
Generate decoy peptides.
"""
import argparse, os
import processFasta as pfasta
import process_rawdata as prd

parser = argparse.ArgumentParser(
	description='''Generate decoy peptides from fasta protein sequence
		database file for peptide identification/validation''')
parser.add_argument('fasta', metavar='*.fasta|*.fa',
	help='FASTA file of target proteins sequences for which to create decoys')
parser.add_argument('--enzyme', '-e', dest='enzyme', default='Trypsin',
	help='Enzyme for proteolytic cleavage. Default = Trypsin')
parser.add_argument('--peptide_length_range', '-l', dest='lenrg',
	default=(7,60), type=int,
	help='Set minimum length of peptides to compare '+\
		'between target and decoy. Default = (7, 60)')
parser.add_argument('--decoy_prefix', '-d', dest='dprefix', default='DECOY_',
	help='Set accesion prefix for decoy proteins in output. Default=DECOY_')
parser.add_argument('--output_fasta', '-o', dest='dout',
	help='Set file to write decoy proteins to.'+\
		' Default=decoy_input file name.fasta')
args = parser.parse_args()

enz = pfasta.proteolysis(args.enzyme)
fastafilename = os.path.basename(args.fasta)
aares = prd.AARES
mh2o = 18.006067


with open('%s%s'%(args.dprefix, fastafilename), 'w') as w2f:
	w2f.write('Protein_Name\tSequence\tMonoisotopic_Mass\n')
	with open(args.fasta, 'r') as f:
		seqs, n, nt = [], 0, 0
		for title, proteinseq in pfasta.getsequence(f):
			pkx = enz.cleavage(proteinseq[::-1])
			if len(pkx)>0:
				proteinid = title.split()[0][1:]
				seqs += ['>%s%s\t%s\t%.6f\n'%(args.dprefix, proteinid, pk,
					sum(aares[sk]['mono'] for sk in pk)+mh2o)
					for pk in pkx]
				n += 1
				if n==1000:
					nt += n
					print('%d proteins processed ...'%nt)
					w2f.writelines(seqs)
					seqs, n = [], 0