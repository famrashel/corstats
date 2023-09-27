import os
import sys
import time
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from collections import Counter
from optparse import OptionParser

###############################################################################

__author__ = """
[TK] Tomohiro Kawabe <kawabe413@akane.waseda.jp>,
[FR] Fam Rashel <fam.rashel@fuji.waseda.jp>"""

__date__, __version__ = '03/03/2016', '1.0' # Original version
__date__, __version__ = '07/03/2016', '1.1' # Add sequence mode, which is ngram representation
__date__, __version__ = '15/06/2016', '1.2' # Resume overcounting when using bicorpus
__date__, __version__ = '01/05/2017', '1.3' # Separate average length of types and tokens [FR]
__date__, __version__ = '05/05/2020', '1.4' # Port to python3 [FR]
__date__, __version__ = '10/08/2020', '1.5' # Classes: Corpora and Corpus; Pretty print as table [FR]

__description__ = """
Profile corpus in detail.
Default: Output five features and TTR and FS
	 1. Corpus size
	 2. Vocabulary size
	 3. Average length of words
	 4. Size of hapax (%)
	 5. Stop words (from the top of 1st to 20th)
	 6. Type Token Ratio (TTR)
	 7. Frequency Spectrum (FS)

Plot distribution: Plot distribution of Zipf or Yule
	 1. Zipf: Logarithmic distribution between frequency and rank of words
	 2. Yule: Logarithmic distribution between frequency and size of word token that frequency is the same
			 """

__verbose__ = False		# Gives information about timing, etc. to the user.
__trace__ = False		# To be used by the developper for debugging.

###############################################################################

class Corpora_profile(list):
	"""
	Class to represent corpora: collection of corpus
	"""

	def __init__(self, corpora):
		list.__init__(self, corpora)
	
	def __repr__(self):
		if len(self) > 0:
			text = ['-']
			for i, file in enumerate(self):
				text.append( (i+1)*'\t' + file.corpus )
			for corpus_prop in self[0]:
				text.append( '\t'.join([corpus_prop] + [ str(corpus[corpus_prop]) for corpus in self ]) )
			s = '\n'.join(text)
		else:
			s = 'no corpus'
		return s

class Corpus_profile(defaultdict):
	"""
	Class to represent a corpus and calculate the statistics on it.
	"""

	def __init__(self, corpus, ngram_size=1, bicorpus=False, zipf=False, yule=False, delimiter= ' '):
		"""
		This is processing section.
		Tokenise corpus and make dictionary among words and frequency and so on.
		"""
		
		defaultdict.__init__(self, lambda:'-')
		#Input
		self.corpus = corpus
		self.ngram_size = ngram_size
		if __verbose__:
			print('\n##############################', file=sys.stderr)
			print('# Corpus file: {}'.format(self.corpus), file=sys.stderr)
			print('# N-gram size: {}'.format(self.ngram_size), file=sys.stderr)

		# Check whether the file exist or not
		if os.path.isfile(self.corpus):

			#List of words in the corpus
			self.word_list = []

			# Map of word and which frequency
			self.word_property = {}

			# Map of word frequency and size of word tokens in the same freq
			self.yule_property = {}

			# Map of word frequency and rank
			self.zipf_property = {}

			# Ranking (depend on frequency)
			self.ranking = 0

			# Read and tokenize corpus by sequence which is set in advance
			for ln,strings in enumerate(open(self.corpus,'r')):
				s_strings = strings.strip().split(delimiter)
				self.ngram(self.ngram_size,s_strings,self.word_list)
			
			# Make the map of word_property and zipf_property, and find error of the empty corpus 
			if len(self.word_list) != 0:
				self.s_counter = Counter(self.word_list)
				for word,cnt in self.s_counter.most_common():
					self.ranking += 1
					self.word_property[word] = cnt
					self.zipf_property[self.ranking] = cnt

				# Calculate corpus stats
				self.general_profiling(self.word_property,self.s_counter,'src')
				self.ttr(self.word_property,'src')
				self.frequency_spectrum(self.s_counter,'src',self.yule_property)

			# Describe the graph of distribution related with corpus 
			if zipf:
				self.plot_zipf_dist()
			if yule:
				self.plot_yule_dist()

	def ngram(self,sequence,wordlist,ngram_wordlist):
		"""
		Create ngram tokens.
		
		wordlist_A = ['he', 'is', 'Takashi']
		2gram_wordlist = []
		
		self.ngram(2,wordlist_A,2gram_wordlist)
		2gram_wordlist = ['he is','is Takashi']
		"""
		for cell_num in range((len(wordlist)-int(sequence)+1)):
			ngram = ' '.join(wordlist[cell_num:cell_num+int(sequence)])
			ngram_wordlist.append(ngram)

	def general_profiling(self,wordproperty,counterproperty,category):
		"""
		Output general properties in corpus, that is,
		1. the size of corpus
		2. the size of vocabulary in corpus
		3. the average length of words (type) and tokens
		4. stop words (upper ranking per frequency from 1 to 20)
		5. the size of hapax

		Input: dictionary linked with word and freq.
			   Counter() item
			   Category (you can set the name of this seat)

		Output: Corpus propaty from 1. to 5.  
		"""
		
		voc_size = len(wordproperty.keys())
		word_size = sum(wordproperty.values())
		stopwords = dict()
		words_len = []
		corpus_len = []
		hapax_size = 0
		count_rank = 0 # This is used to choose stop words (from 1st to 20th)

		for word in wordproperty.keys():
			words_len.append(len(word)) 
			corpus_len.extend([ len(word) ] * wordproperty[word])
			if wordproperty[word] == 1:
				hapax_size += 1
		
		self['# of tokens'] = word_size
		self['Avg token length'] = np.mean(corpus_len)
		self['Token length std'] = np.std(corpus_len)

		self['# of types'] = voc_size
		self['Avg type length'] = np.mean(words_len)
		self['Type length std'] = np.std(words_len)
		
		self['Hapax size'] = hapax_size
		self['Hapax perc'] = 1.0*100*(1.0*hapax_size/voc_size)

		if __verbose__:
			print('\n## General statistics - {}'.format(category), file=sys.stderr)
			print('Corpus size (# of tokens): {}'.format(self['# of tokens']), file=sys.stderr)
			print('Average token length: {}±{}'.format(round(self['Avg token length'], 3), round(self['Token length std'], 3)), file=sys.stderr)
			print('Vocabulary size (# of types): {}'.format(self['# of types']), file=sys.stderr)
			print('Average type length: {}{}'.format(round(self['Avg type length'], 3), round(self['Type length std'], 3)), file=sys.stderr)
			print('Hapaxes: {}% ({}/{})'.format(round(self['Hapax perc'], 3), self['Hapax size'], self['# of types']), file=sys.stderr)
			print('Stop words (top-20):', file=sys.stderr)
			for word, cnt in counterproperty.most_common():
							count_rank += 1
							if count_rank <= 20:
									print('{}\t{}'.format(word, cnt), file=sys.stderr)
		
	def ttr(self,wordproperty,category):
		"""
		Output type token ratio (TTR), which represents the wealth of words in corpus
		
		Input: dictionary linked with word and freq.
			   Category (you can set the name of this seat)

		Output: TTR from 1. to 6.
		"""
		
		v = len(wordproperty.keys())
		n = sum(wordproperty.values())
		
		self['Simple TTR'] = (1.0*v/n)
		self['Guiraud TTR'] = (1.0*v/(n**0.5))
		self['Herdan TTR'] = 1.0*math.log(v,10.0)/math.log(n,10.0)
		self['Maas TTR'] = ((1.0*math.log(n,10.0)-1.0*math.log(v,10.0))/math.log(n,10.0))**0.5
		self['Tuldava TTR'] = -1.0*(1-v**2)/((v**2)*math.log(n,10.0))
		self['Dugast TTR'] = math.log(v,10.0)/math.log(math.log(n,10.0),10.0)

		if __verbose__:
			print('\n## Type Token Ratio (TTR) data - {}'.format(category), file=sys.stderr)
			print("""
Output type token ratio (TTR) represents the wealth of words in corpus, that is (V: size of vocabulary, N: size of words),
1. simple TTR = V/N
2. Guiraud TTR = V/N^0.5
3. Herdan TTR = log(V)/log(N)
4. Maas TTR = ((log(N)-log(V))/(log(N))^2)^0.5
5. Tuldava TTR = (1-V^2)/v^2log(N)
6. Dugast TTR = log(V)/log(log(N))
			""", file=sys.stderr)
			print('simple TTR: {}'.format(round(self['Simple TTR'], 3)), file=sys.stderr)
			print('Guiraud TTR: {}'.format(round(self['Guiraud TTR'], 3)), file=sys.stderr)
			print('Herdan TTR: {}'.format(round(self['Herdan TTR'], 3)), file=sys.stderr)
			print('Maas TTR: {}'.format(round(self['Maas TTR'], 3)), file=sys.stderr)
			print('Tuldava TTR: {}'.format(round(self['Tuldava TTR'], 3)), file=sys.stderr)
			print('Dugast TTR: {}'.format(round(self['Dugast TTR'], 3)), file=sys.stderr)

	def frequency_spectrum(self,counterproperty,category,yuledict):
		"""
		Output the frequency spectrum (FS), that represents the wealth of words in corpus.
		The defference between TTR and FS is that FS consider the frequency of each vocabulary in corpus, but TTR do not.

		Input: Counter() item
			   Category (you can set the name of this seat)
			   dictionary (generate dictionary linked with freq. and token size)

		Output: FS from 1. to 4.
		"""
		
		yuleproperty = defaultdict(dict)

		Yule = 0.0
		Simpson = 0.0
		Sichel = 0.0
		Honore = 0.0

		Yule_sum = 0.0
		Simpson_sum = 0.0
		Sichel_sum = 0.0
		Honore_sum = 0.0
		
		category_count = 0
		for word,cnt in counterproperty.most_common():
			category_count += 1 # same as ranking of words
			yuleproperty[str(cnt)][category_count] = word #{"freq":{ranking:words,....},....{"50":{4500:'line',4501:'12',...}}}

		N = [float(freq) for freq in yuleproperty.keys()] # token of freq. of words
		n = sum(N) # size of words in the corpus
		Vn = 0.0

		for frequency in yuleproperty.keys():
			m = float(frequency)
			Vmn = float(len(yuleproperty[frequency].values())) # represent the size of token per words
			n = sum(N)
			Vn += Vmn
			Yule_sum += 1.0*Vmn*(m**2.0)
			Simpson_sum += 1.0*Vmn*((m*(m-1))/(n*(n-1)))
			yuledict[m] = Vmn

		Sichel_sum = len(yuleproperty['2'].values())
		Honore_sum = len(yuleproperty['1'].values())

		self['Yule'] = 1.0*(10**4)*(Yule_sum-n)/(n**2)
		self['Simpson'] = Simpson_sum
		self['Sichel'] = 1.0*Sichel_sum/Vn
		self['Honore'] = 1.0*100*(math.log(n,10.0)/(1.0-1.0*(Honore_sum/Vn)))

		if __verbose__:
			print('\n## Frequency spectrum data - {}'.format(category), file=sys.stderr)
			print("""
Output the frequency spectrum (FS), that represents the wealth of words in corpus.
The defference between TTR and FS is that FS consider the frequency of each vocabulary in corpus, but TTR do not.
The type of FS is (V(m,N): size of words that frequency is m, N: size of words),
1. Yule FS = 10^4*((sum(V(m,N)*m^2)-N)/N^2
2. Simpson FS = sum(V(m,N)*(m(m-1)/N(N-1)))
3. Sichel FS = V(2,N)/V(N)
4. Honore FS = 100*(log(N)/(1-V(1,N)/V(N)))
				""", file=sys.stderr)
			print('Yule FS: {}'.format(round(self['Yule'],3)), file=sys.stderr)
			print('Simpson FS: {}'.format(round(self['Simpson'],3)), file=sys.stderr)
			print('Sichel FS: {}'.format(round(self['Sichel'],3)), file=sys.stderr)
			print('Honore FS: {}'.format(round(self['Honore'],3)), file=sys.stderr)

	def plot_zipf_dist(self):
		"""
		Plot distribution of Zipf.
		"""
		
		if len(self.zipf_property) != 0:
			x = self.zipf_property.keys()
			y = self.zipf_property.values()
			plt.plot(x,  y, "o", label=self.corpus)
		
		plt.xlabel('Ranking')
		plt.ylabel('Frequency')
		plt.xscale('log')
		plt.yscale('log')
		plt.title('Distribution between word and ranking')
		plt.show()
	
	def plot_yule_dist(self):
		"""
		Plot distribution of Yule
		"""
		
		if len(self.yule_property) != 0:
			x = self.yule_property.keys()
			y = self.yule_property.values()
			plt.plot(x, y, "o", label=self.corpus)
		
		plt.legend()
		plt.xlabel('Frequency')
		plt.ylabel('Number of token of frequency')
		plt.xscale('log')
		plt.yscale('log')
		plt.title('Distribution between freq and size of tokens of words in the same count')
		plt.show()

###################################################################

def convert_time(duration):
	"""
	Convert time data (s) to human-friendly format
	"""
	m, s = divmod(round(duration), 60)
	h, m = divmod(m, 60)
	hms = "%d:%02d:%02d" % (h, m,s)
	return hms

def read_argv():

	from argparse import ArgumentParser, RawTextHelpFormatter
	this_version = 'v{} (c) {}{}'.format(__version__, __date__.split('/')[2], __author__)
	this_description = __description__
	this_usage = '''
	python3 %(prog)s --input_files CORPUS_FILE_PATH
	'''

	parser = ArgumentParser(description=this_description, usage=this_usage, epilog=this_version, formatter_class=RawTextHelpFormatter)
	parser.add_argument('--input_files',
				  nargs='+', action='store', dest='input_files', required=True, default=[],
                  help='list of corpora')
	parser.add_argument('-i', '--input-mode',
				  action='store', dest='input_mode', type=str, default='words',
				  help='input file mode (Not yet finished): [1] words (default), [2] vectors, [3] sigmorphon')
	parser.add_argument('-b','--bicorpus',
				  action='store_true', dest='bicorpus', default=False,
				  help='use bicorpus')
	parser.add_argument('--ngram',
				  action='store', dest='ngram', type=int, default=1,
				  help='generate the score of ngram, default is unigram')
	parser.add_argument('-z','--zipf',
				  action='store_true', dest='zipf', default=False,
				  help='output logarithmic distribution between frequency and rank of words')
	parser.add_argument('-y','--yule',
				  action='store_true', dest='yule', default=False,
				  help='output logarithmic distribution between frequency and size of word token that frequency is the same')
	parser.add_argument('-V', '--verbose',
				  action='store_true', dest='verbose', default=False,
				  help='runs in verbose mode')
	parser.add_argument('-t', '--trace',
				  action='store_true', dest='trace', default=False,
				  help='runs in trace mode')
	parser.add_argument('-T', '--test',
				  action='store_true', dest='test', default=False,
				  help='run all unitary tests')

	return parser.parse_args()

###################################################################

def _test():
	import doctest
	doctest.testmod()
	sys.exit(0)

if __name__ == '__main__':
	options = read_argv()
	
	__verbose__ = options.verbose
	__bicorpus__ = options.bicorpus
	
	t_start = time.time()

	if __verbose__:
		print('# Start profiling.......', file=sys.stderr)
		print('# N-gram: {}'.format(options.ngram), file=sys.stderr)
		print('# Number of corpus: {}'.format(len(options.input_files)), file=sys.stderr)
		print('# Corpora:\n{}'.format('\n'.join([ '\t- ' + file for file in options.input_files ])), file=sys.stderr)

	print(Corpora_profile( Corpus_profile(file) for file in options.input_files ))
	
	if __verbose__:
		print('# Total processing time: %s' % (convert_time(time.time() - t_start)), file=sys.stderr)