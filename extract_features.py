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


def pos_analyzer(tokens):
	"""
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
	tokens = re.split("\s+", "heat water brown sugar chocolate cloves and cinnamon.")
	tokens_pos = pos_tag(tokens)
	pos = [p for t, p in tokens_pos]

	pos_dict = nltk.tagset_mapping('en-ptb', 'universal')
	[pos_dict[p] for p in pos]
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

	# part of speech tagging
	# pos_features = pos_analyzer(tokens)

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
	top_words, top_doc_freqs = get_most_freq()
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

	feature_names = ["tokens_char_len", "num_tokens", "num_go_tokens", "num_stop_tokens", "pct_stop_tokens",
		"num_percents", "num_pounds", "num_slashes", "num_dashes",
		"avg_token_length", "any_numbers", "ct_numbers", "pct_numbers",
		"last_has_number", "last_has_percent", "last_has_pound", "last_has_slash", "last_has_dash",
		"last_is_stopword", "last_num_chars", "last_syllable_count"]
	feature_names += ['any_' + w for w in top_words]
	feature_names += ['tfidf_' + w for w in top_words]
	rval = [tokens_char_len, num_tokens, num_go_tokens, num_stop_tokens, pct_stop_tokens,
		num_percents, num_pounds, num_slashes, num_dashes,
		avg_token_length, any_numbers, ct_numbers, pct_numbers,
		last_has_number, last_has_percent, last_has_pound, last_has_slash, last_has_dash,
		last_is_stopword, last_num_chars, last_syllable_count] + any_stats + tfidf_stats
	return rval


def extract_features(verbose=False):
	data_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'
	token_table = open(data_dir + 'tokens.txt', 'r')
	training_table = open(data_dir + 'training.csv', 'wb')
	training_table.write('recipe_key,token_key,target,recipe_char_len,num_recipe_tokens,proportion_of_recipe,pct_tokens_seen,tokens_char_len,num_tokens,num_go_tokens,num_stop_tokens,pct_stop_tokens,num_percents,num_pounds,num_slashes,num_dashes,avg_token_length,any_numbers,ct_numbers,pct_numbers,last_has_number,last_has_percent,last_has_pound,last_has_slash,last_has_dash,last_is_stopword,last_num_chars,last_syllable_count,any_bring,any_dough,any_oven,any_dish,any_chicken,any_milk,any_sprinkl,any_cut,any_cup,any_lake,any_sugar,any_add,any_spread,any_kingdom,any_brown,any_sourc,any_format,any_beat,any_flour,any_spoon,any_half,any_cook,any_cool,any_recip,any_meat,any_onion,any_sauc,any_small,any_side,any_set,any_hour,any_powder,any_tender,any_bake,any_slice,any_preheat,any_melt,any_larg,any_simmer,any_saucepan,any_pepper,any_water,any_let,any_greas,any_befor,any_dri,any_turn,any_degre,any_place,any_one,any_skillet,any_use,any_top,any_boil,any_butter,any_medium,any_heat,any_warm,any_togeth,any_refriger,any_serv,any_ingredi,any_mc,any_f,any_veget,any_remov,any_smooth,any_pour,any_remain,any_light,any_minut,any_cover,any_high,any_cream,any_chop,any_make,any_bowl,any_mix,any_hot,any_inch,any_you,any_stir,any_roll,any_pan,any_juic,any_oil,any_mixtur,any_possum,any_garlic,any_tomato,any_drain,any_potato,any_egg,any_well,any_salt,any_chees,any_combin,any_time,any_cookbook,any_blend,count_bring,count_dough,count_oven,count_dish,count_chicken,count_milk,count_sprinkl,count_cut,count_cup,count_lake,count_sugar,count_add,count_spread,count_kingdom,count_brown,count_sourc,count_format,count_beat,count_flour,count_spoon,count_half,count_cook,count_cool,count_recip,count_meat,count_onion,count_sauc,count_small,count_side,count_set,count_hour,count_powder,count_tender,count_bake,count_slice,count_preheat,count_melt,count_larg,count_simmer,count_saucepan,count_pepper,count_water,count_let,count_greas,count_befor,count_dri,count_turn,count_degre,count_place,count_one,count_skillet,count_use,count_top,count_boil,count_butter,count_medium,count_heat,count_warm,count_togeth,count_refriger,count_serv,count_ingredi,count_mc,count_f,count_veget,count_remov,count_smooth,count_pour,count_remain,count_light,count_minut,count_cover,count_high,count_cream,count_chop,count_make,count_bowl,count_mix,count_hot,count_inch,count_you,count_stir,count_roll,count_pan,count_juic,count_oil,count_mixtur,count_possum,count_garlic,count_tomato,count_drain,count_potato,count_egg,count_well,count_salt,count_chees,count_combin,count_time,count_cookbook,count_blend\n')

	prev_recipe_key = ''
	for i, token_line in enumerate(token_table):
		if token_line == '\n':
			break
		if i == 0:
			continue  # skip header
		token_fields = token_line.split('\t')
		recipe_key = token_fields[0]
		token_key = token_fields[2]
		target = token_fields[3]
		tokens = token_fields[4].replace("\n", "")
		token_in_recipe_start = int(token_fields[5])

		if recipe_key != prev_recipe_key:
			# found a new recipe
			print "Onto recipe:", recipe_key
			recipe = lookup_recipe(recipe_key, data_dir)
			recipe_features = analyze_recipe(recipe)

		token_features = analyze_tokens(tokens, stopwords, syllable_dict)
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
	start_time = time.time()
	num_obs = extract_features(verbose=True)
	end_time = time.time()
	print "Run time:", round((end_time - start_time)/60, 1), "minutes"
	print "finished running, extracted features for", num_obs, "files"
