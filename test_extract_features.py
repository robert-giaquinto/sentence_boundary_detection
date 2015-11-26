from nltk.corpus import cmudict
from extract_features import num_syllables, analyze_tokens, analyze_recipe


def test_num_syllables():
	s = cmudict.dict()
	tests = ['animal', 'i', '0', 1]
	expected = [3, 1, 1, 1]
	results = []
	for t in tests:
		results.append(num_syllables(t, s))
	rval = 1.0 * sum([e == r for e, r in zip(expected, results)]) / len(expected)
	return rval


def test_analyze_tokens():
	stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 't', 'can', 'will', 'just', 'don', 'should', 'now']
	syllable_dict = cmudict.dict()
	tests = 'this is a test'
	expected = [14, 4, 1, 3, 0.75,
		0, 0, 0, 0,
		11.0/4.0,
		0, 0, 0,
		0, 0, 0, 0, 0,
		0, 4, 1]
	results = analyze_tokens(tests, stopwords, syllable_dict)
	rval = 1.0 * sum([e == r for e, r in zip(expected, results)]) / len(expected)
	return rval


def test_analyze_recipe():
	tests = ['stir all ingredients', 'recipe', '1', None]
	expected = [[20, 3], [6,1], [1,1], [0,0]]
	results = []
	for t in tests:
		results.append(analyze_recipe(t))
	rval = 1.0 * sum([e == r for e, r in zip(expected, results)]) / len(expected)
	return rval



x = test_num_syllables()
print "num_syllables:", x

x = test_analyze_tokens()
print "analyze_tokens:", x

x = test_analyze_recipe()
print "analyze_recipe:", x




