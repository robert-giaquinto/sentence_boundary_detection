from __future__ import division
import os
import re
from nltk.stem import SnowballStemmer
from collections import Counter
import time


def frequency_analysis(input_path, output_path, stopwords, n_most_common=50):
	recipes = []
	with open(input_path, 'r') as f:
		for i, line in enumerate(f):
			if line == '\n':
				break
			if i == 0:
				continue  # skip header
			fields = line.split('\t')
			recipes.append(fields[1].replace("\n", ""))
	recipe_text = re.sub("[^a-z ]", "", ' '.join(recipes))
	recipe_words = re.split("\s+", recipe_text)
	stemmer = SnowballStemmer("english")
	recipe_stems = [stemmer.stem(w) for w in recipe_words]
	recipe_stems = filter(None, [s for s in recipe_stems if s not in stopwords])
	top_words = Counter(recipe_stems).most_common(n_most_common)

	# write to a file
	# do a second pass of the recipe to determine how many of the documents the term is in
	freq_table = open(output_path, 'wb')
	for elt in top_words:
		doc_freq = sum([elt[0] in recipe for recipe in recipes])
		freq_table.write(','.join([str(e) for e in elt]) +','+ str(doc_freq) + '\n')
	freq_table.close()


def get_most_freq():
	words = []
	doc_freqs = []
	with open('/Users/robert/Documents/UMN/5551_Robots/Project/data/word_frequencies.csv', 'r') as f:
		for line in f:
			if line == '\n':
				break
			fields = line.split(',')
			words.append(fields[0])
			doc_freqs.append(int(fields[2]))
	return words, doc_freqs


if __name__ == "__main__":
	input_path = '/Users/robert/Documents/UMN/5551_Robots/Project/data/recipes.txt'
	output_path = '/Users/robert/Documents/UMN/5551_Robots/Project/data/word_frequencies.csv'
	stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 't', 'can', 'will', 'just', 'don', 'should', 'now']
	start_time = time.time()
	frequency_analysis(input_path, output_path, stopwords, 100)
	end_time = time.time()
	print "Finished frequency analysis in", round(end_time - start_time, 1), "seconds"