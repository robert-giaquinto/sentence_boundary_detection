from __future__ import division
import os
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import cmudict


def num_syllables(word, syllable_dict):
	std_word = str(word).lower()
	if std_word in syllable_dict:
		# at least one pronunciation found, just use first
		syllables = syllable_dict[std_word]
		rval = len(list(y for y in syllables[0] if y[-1].isdigit()))
	else:
		# must be a weird word or number, assume 1 syllable...?
		rval = 1
	return rval


def lookup_recipe(recipe_key, data_dir):
	with open(data_dir + 'recipes.txt', 'r') as f:
		for line in f:
			if line == '\n':
				break
			fields = line.split('\t')
			if fields[0] == recipe_key:
				return fields[1].replace("\n", "")
			else:
				continue


def analyze_recipe(recipe):
	"""
	extract recipe level features
	:param recipe:
	:return:
	"""
	if recipe is None:
		return [0, 0]
	recipe_char_len = len(recipe)
	# split tokens into words
	tokens = filter(None, re.split('\s+', recipe))
	num_recipe_tokens = len(tokens)
	rval = [recipe_char_len, num_recipe_tokens]
	return rval


def pos_analyzer(tokens):
	tokens_pos = nltk.pos_tag(tokens)
	pos = [p for t, p in tokens_pos]


	return 0


def analyze_tokens(string, stopwords, syllable_dict):
	"""
	extract token level features
	:param tokens:
	:return:
	"""
	tokens_char_len = len(string)
	num_percents = string.count('%')
	num_pounds = string.count('#')
	num_slashes = string.count('/')
	num_dashes = string.count('-')

	# split tokens into words
	tokens = filter(None, re.split('\s+', string))
	num_tokens = len(tokens)

	# analyze all tokens given
	stemmer = SnowballStemmer("english")
	go_tokens = filter(None, [stemmer.stem(w).lower().strip() for w in tokens])
	go_tokens = [w for w in go_tokens if w not in stopwords]
	num_go_tokens = len(go_tokens)
	num_stop_tokens = num_tokens - len(go_tokens)
	pct_stop_tokens = 1.0 * num_stop_tokens / num_tokens

	avg_token_length = 1.0 * sum([len(t) for t in tokens]) / num_tokens
	is_a_number = [int(re.search('[0-9]', t) is not None) for t in tokens]
	any_numbers = max(is_a_number)
	ct_numbers = sum(is_a_number)
	pct_numbers = 1.0 * ct_numbers/ num_tokens

	# part of speech tagging
	# pos_features = pos_analyzer(tokens)

	# analyze just the last token given
	last = tokens[num_tokens-1]
	last_has_number = int(re.search('[0-9]', last) is not None)
	last_has_percent = int(re.search('[%]', last) is not None)
	last_has_pound = int(re.search('[#]', last) is not None)
	last_has_slash = int(re.search('[/]', last) is not None)
	last_has_dash = int(re.search('[\-]', last) is not None)
	last_is_stopword = int(last in stopwords)
	last_num_chars = len(last)
	last_syllable_count = num_syllables(last, syllable_dict)
	rval = [tokens_char_len, num_tokens, num_go_tokens, num_stop_tokens, pct_stop_tokens,
		num_percents, num_pounds, num_slashes, num_dashes,
		avg_token_length,
		any_numbers, ct_numbers, pct_numbers,
		last_has_number, last_has_percent, last_has_pound, last_has_slash, last_has_dash,
		last_is_stopword, last_num_chars, last_syllable_count
	]
	return rval


def extract_features():
	stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 't', 'can', 'will', 'just', 'don', 'should', 'now']
	syllable_dict = cmudict.dict()
	data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
	token_table = open(data_dir + 'tokens.txt', 'r')
	training_table = open(data_dir + 'training.csv', 'wb')
	training_table.write('recipe_key,token_key,recipe_char_len,num_recipe_tokens,proportion_of_recipe,pct_tokens_seen,tokens_char_len,num_tokens,num_go_tokens,num_stop_tokens,pct_stop_tokens,num_percents,num_pounds,num_slashes,num_dashes,avg_token_length,any_numbers,ct_numbers,pct_numbers,last_has_number,last_has_percent,last_has_pound,last_has_slash,last_has_dash,last_is_stopword,last_num_chars,last_syllable_count\n')

	prev_recipe_key = ''
	tokens_encountered = 0
	for i, token_line in enumerate(token_table):
		if token_line == '\n' or i > 1000:
			break
		if i == 0:
			continue  # skip header
		token_fields = token_line.split('\t')
		recipe_key = token_fields[0]
		token_key = token_fields[2]
		tokens = token_fields[4].replace("\n", "")

		if recipe_key != prev_recipe_key:
			# found a new recipe
			recipe = lookup_recipe(recipe_key, data_dir)
			recipe_features = analyze_recipe(recipe)
			tokens_encountered = 0

		token_features = analyze_tokens(tokens, stopwords, syllable_dict)
		num_tokens = token_features[1]
		tokens_encountered += num_tokens
		proportion_of_recipe = num_tokens * 1.0 / recipe_features[1]
		pct_tokens_seen = tokens_encountered * 1.0 / recipe_features[1]

		# write out result
		training_table.write(recipe_key + ',' + token_key + ',' + ','.join([str(f) for f in recipe_features]) + ',' + str(proportion_of_recipe) + ',' + str(pct_tokens_seen) + ',' + ','.join([str(f) for f in token_features]) + "\n")
	token_table.close()
	training_table.close()
	return i


if __name__ == "__main__":
	num_obs = extract_features()
	print "finished running, extracted features for", num_obs, "files"
