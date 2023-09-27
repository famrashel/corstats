# corstats

A Python script to measure some statistics of given corpora:

- Default: Output five features and TTR and FS
	 1. Corpus size
	 2. Vocabulary size
	 3. Average length of words
	 4. Size of hapax (%)
	 5. Stop words (from the top of 1st to 20th)
	 6. Type Token Ratio (TTR)
	 7. Frequency Spectrum (FS)
- Plot distribution: Plot distribution of Zipf or Yule
	 1. Zipf: Logarithmic distribution between frequency and rank of words
	 2. Yule: Logarithmic distribution between frequency and size of word token that frequency is the same

## Author(s)
- Originally created by Tomohiro KAWABE
- Improved by FAM Rashel