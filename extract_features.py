from __future__ import division
import os
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import cmudict
from nltk.tag import pos_tag
from frequency_analysis import get_most_freq
import time

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 't', 'can', 'will', 'just', 'don', 'should', 'now']
syllable_dict = cmudict.dict()


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


def extract_pos(tokens, simple=True):
	"""
	Simple parts of speech of speech are:
	VERB - verbs (all tenses and modes)
	NOUN - nouns (common and proper)
	PRON - pronouns
	ADJ - adjectives
	ADV - adverbs
	ADP - adpositions (prepositions and postpositions)
	CONJ - conjunctions
	DET - determiners
	NUM - cardinal numbers
	PRT - particles or other function words
	X - other: foreign words, typos, abbreviations
	. - punctuation
	:param tokens:
	:return:
	"""
	tokens_pos = pos_tag(tokens)
	pos = [p for t, p in tokens_pos]
	if simple:
		# translate larger set of part of speech tags into small, simpler set
		pos_dict = nltk.tagset_mapping('en-ptb', 'universal')
		pos = [pos_dict[p] for p in pos]
	return pos


def analyze_pos(pos, simple=True):
	"""
	Extract POS features, currently only suited to hand the simple POS

	VERB - verbs (all tenses and modes)
	NOUN - nouns (common and proper)
	PRON - pronouns
	ADJ - adjectives
	ADV - adverbs
	ADP - adpositions (prepositions and postpositions)
	CONJ - conjunctions
	DET - determiners
	NUM - cardinal numbers
	PRT - particles or other function words
	X - other: foreign words, typos, abbreviations
	. - punctuation
	:param pos:
	:return:
	"""
	if simple:
		pos_types = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.']
	else:
		print "only suited to handle simple parts of speech right now"
		return [], [], [], []
	pos_counts = [pos.count(p) for p in pos_types]
	pos_pct = [1.0 * pos.count(p) / len(pos) for p in pos_types]
	pos_last = [int(pos[-1] == p) for p in pos_types]
	pos_types[-1] = 'punc'
	return pos_counts, pos_pct, pos_last, pos_types


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

	# part of speech tagging
	pos = extract_pos(tokens, simple=True)
	pos_counts, pos_pct, pos_last, pos_types = analyze_pos(pos)

	# analyze all tokens given
	avg_token_length = 1.0 * sum([len(t) for t in tokens]) / num_tokens
	is_a_number = [int(re.search('[0-9]', t) is not None) for t in tokens]
	any_numbers = max(is_a_number)
	ct_numbers = sum(is_a_number)
	pct_numbers = 1.0 * ct_numbers / num_tokens

	# analyze stop words
	stemmer = SnowballStemmer("english")
	go_tokens = filter(None, [stemmer.stem(w).lower().strip() for w in tokens])
	go_tokens = [w for w in go_tokens if w not in stopwords]
	num_go_tokens = len(go_tokens)
	num_stop_tokens = num_tokens - len(go_tokens)
	pct_stop_tokens = 1.0 * num_stop_tokens / num_tokens

	# frequency based features
	top_words, top_doc_freqs = get_most_freq(filename='word_frequencies.csv')
	any_stats = [int(w in go_tokens) for w in top_words]
	# tf * idf
	tfidf_stats = [1.0 * go_tokens.count(w) * 5589 / f for w, f in zip(top_words, top_doc_freqs)]

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

	# frequency based features
	all_top_words, all_top_doc_freqs = get_most_freq(filename='all_word_frequencies.csv')
	last_any_stats = [int(w == last) for w in all_top_words]
	# tf * idf
	last_tfidf_stats = [1.0 * (last == w) * 5589 / f for w, f in zip(all_top_words, all_top_doc_freqs)]

	feature_names = ["tokens_char_len", "num_tokens", "num_go_tokens", "num_stop_tokens", "pct_stop_tokens",
		"num_percents", "num_pounds", "num_slashes", "num_dashes",
		"avg_token_length", "any_numbers", "ct_numbers", "pct_numbers",
		"last_has_number", "last_has_percent", "last_has_pound", "last_has_slash", "last_has_dash",
		"last_is_stopword", "last_num_chars", "last_syllable_count"]
	feature_names += ['pos_last_' + p for p in pos_types]
	feature_names += ['last_is_' + w for w in all_top_words]
	feature_names += ['last_tfidf_' + w for w in all_top_words]
	feature_names += ['pos_counts_' + p for p in pos_types]
	feature_names += ['pos_pct_' + p for p in pos_types]
	feature_names += ['any_' + w for w in top_words]
	feature_names += ['tfidf_' + w for w in top_words]
	# print ','.join(feature_names)
	rval = [tokens_char_len, num_tokens, num_go_tokens, num_stop_tokens, pct_stop_tokens,
		num_percents, num_pounds, num_slashes, num_dashes,
		avg_token_length, any_numbers, ct_numbers, pct_numbers,
		last_has_number, last_has_percent, last_has_pound, last_has_slash, last_has_dash,
		last_is_stopword, last_num_chars, last_syllable_count]
	rval += pos_last + last_any_stats + last_tfidf_stats + pos_counts + pos_pct + any_stats + tfidf_stats
	return rval


def extract_features(token_path, output_path, verbose=False):
	token_table = open(token_path, 'r')
	training_table = open(output_path, 'wb')
	training_table.write('recipe_key,token_key,target,recipe_char_len,num_recipe_tokens,proportion_of_recipe,pct_tokens_seen,tokens_char_len,num_tokens,num_go_tokens,num_stop_tokens,pct_stop_tokens,num_percents,num_pounds,num_slashes,num_dashes,avg_token_length,any_numbers,ct_numbers,pct_numbers,last_has_number,last_has_percent,last_has_pound,last_has_slash,last_has_dash,last_is_stopword,last_num_chars,last_syllable_count,pos_last_VERB,pos_last_NOUN,pos_last_PRON,pos_last_ADJ,pos_last_ADV,pos_last_ADP,pos_last_CONJ,pos_last_DET,pos_last_NUM,pos_last_PRT,pos_last_X,pos_last_punc,last_is_and,last_is_the,last_is_in,last_is_to,last_is_a,last_is_with,last_is_until,last_is_add,last_is_minut,last_is_of,last_is_for,last_is_stir,last_is_or,last_is_on,last_is_heat,last_is_into,last_is_bake,last_is_cook,last_is_over,last_is_serv,last_is_mix,last_is_mixtur,last_is_water,last_is_pan,last_is_sugar,last_is_butter,last_is_about,last_is_at,last_is_salt,last_is_is,last_is_place,last_is_cup,last_is_egg,last_is_bowl,last_is_flour,last_is_cover,last_is_from,last_is_combin,last_is_use,last_is_well,last_is_remov,last_is_oven,last_is_sauc,last_is_pepper,last_is_top,last_is_onion,last_is_boil,last_is_oil,last_is_it,last_is_brown,last_tfidf_and,last_tfidf_the,last_tfidf_in,last_tfidf_to,last_tfidf_a,last_tfidf_with,last_tfidf_until,last_tfidf_add,last_tfidf_minut,last_tfidf_of,last_tfidf_for,last_tfidf_stir,last_tfidf_or,last_tfidf_on,last_tfidf_heat,last_tfidf_into,last_tfidf_bake,last_tfidf_cook,last_tfidf_over,last_tfidf_serv,last_tfidf_mix,last_tfidf_mixtur,last_tfidf_water,last_tfidf_pan,last_tfidf_sugar,last_tfidf_butter,last_tfidf_about,last_tfidf_at,last_tfidf_salt,last_tfidf_is,last_tfidf_place,last_tfidf_cup,last_tfidf_egg,last_tfidf_bowl,last_tfidf_flour,last_tfidf_cover,last_tfidf_from,last_tfidf_combin,last_tfidf_use,last_tfidf_well,last_tfidf_remov,last_tfidf_oven,last_tfidf_sauc,last_tfidf_pepper,last_tfidf_top,last_tfidf_onion,last_tfidf_boil,last_tfidf_oil,last_tfidf_it,last_tfidf_brown,pos_counts_VERB,pos_counts_NOUN,pos_counts_PRON,pos_counts_ADJ,pos_counts_ADV,pos_counts_ADP,pos_counts_CONJ,pos_counts_DET,pos_counts_NUM,pos_counts_PRT,pos_counts_X,pos_counts_punc,pos_pct_VERB,pos_pct_NOUN,pos_pct_PRON,pos_pct_ADJ,pos_pct_ADV,pos_pct_ADP,pos_pct_CONJ,pos_pct_DET,pos_pct_NUM,pos_pct_PRT,pos_pct_X,pos_pct_punc,any_add,any_minut,any_stir,any_heat,any_bake,any_cook,any_serv,any_mix,any_mixtur,any_water,any_pan,any_sugar,any_butter,any_salt,any_place,any_cup,any_egg,any_bowl,any_flour,any_cover,any_combin,any_use,any_well,any_remov,any_oven,any_sauc,any_pepper,any_top,any_onion,any_boil,any_oil,any_brown,any_pour,any_ingredi,any_cream,any_chees,any_larg,any_beat,any_chicken,any_cool,any_remain,any_make,any_hour,any_cut,any_inch,any_degre,any_medium,any_hot,any_sprinkl,any_mc,tfidf_add,tfidf_minut,tfidf_stir,tfidf_heat,tfidf_bake,tfidf_cook,tfidf_serv,tfidf_mix,tfidf_mixtur,tfidf_water,tfidf_pan,tfidf_sugar,tfidf_butter,tfidf_salt,tfidf_place,tfidf_cup,tfidf_egg,tfidf_bowl,tfidf_flour,tfidf_cover,tfidf_combin,tfidf_use,tfidf_well,tfidf_remov,tfidf_oven,tfidf_sauc,tfidf_pepper,tfidf_top,tfidf_onion,tfidf_boil,tfidf_oil,tfidf_brown,tfidf_pour,tfidf_ingredi,tfidf_cream,tfidf_chees,tfidf_larg,tfidf_beat,tfidf_chicken,tfidf_cool,tfidf_remain,tfidf_make,tfidf_hour,tfidf_cut,tfidf_inch,tfidf_degre,tfidf_medium,tfidf_hot,tfidf_sprinkl,tfidf_mc\n')

	prev_recipe_key = ''
	for i, token_line in enumerate(token_table):
		if token_line == '\n':
			break
		if i == 0:
			continue  # skip header
		token_fields = token_line.split('\t')
		recipe_key = token_fields[0]
		token_key = token_fields[1]
		target = token_fields[2]
		string = token_fields[3].replace("\n", "")
		token_in_recipe_start = int(token_fields[4])

		if recipe_key != prev_recipe_key:
			# found a new recipe
			if verbose:
				print "Onto recipe:", recipe_key
			recipe = lookup_recipe(recipe_key, data_dir)
			recipe_features = analyze_recipe(recipe)

		token_features = analyze_tokens(string, stopwords, syllable_dict)
		proportion_of_recipe = token_features[1] * 1.0 / recipe_features[1]
		pct_tokens_seen = token_in_recipe_start * 1.0 / recipe_features[1]

		# write out result
		training_table.write(recipe_key + ',' + token_key + ',' + target + ',' +
							 ','.join([str(f) for f in recipe_features]) + ',' +
							 str(proportion_of_recipe) + ',' + str(pct_tokens_seen) + ',' +
							 ','.join([str(f) for f in token_features]) + "\n")
		prev_recipe_key = recipe_key
	token_table.close()
	training_table.close()
	return i


if __name__ == "__main__":
	data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
	token_path = data_dir + 'tokens.txt'
	output_path = data_dir + 'training.csv'
	start_time = time.time()
	num_obs = extract_features(token_path=token_path, output_path=output_path, verbose=False)
	end_time = time.time()
	print "Run time:", round((end_time - start_time)/60, 1), "minutes"
	print "finished running, extracted features for", num_obs, "files"
