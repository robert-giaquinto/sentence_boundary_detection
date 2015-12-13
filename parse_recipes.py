from __future__ import division
import os
import re
import math
import random

# get list of recipes
# [f for f in os.listdir(input_dir)]
filenames = ['AlcoholicBeverages.txt', 'AllAppetizerRecipes.txt', 'AllBeverageRecipes.txt', 'AllBreadRecipes.txt', 'AllBreakfastRecipes.txt', 'AllCondimentRecipes.txt', 'AllCookieRecipes.txt', 'AllDesertRecipes.txt', 'AllSaladRecipes.txt', 'AllSoupandStewRecipes.txt', 'AllVegetableandSideDishRecipes.txt', 'Asparagus.txt', 'BakedGoods.txt', 'Beef.txt', 'Biscotti.txt', 'BiscottiRecipes.txt', 'BiscuitsandScones.txt', 'BreadMachineRecipes.txt', 'Brownies.txt', 'Cajun.txt', 'Cakes.txt', 'Candy.txt', 'Casseroles.txt', 'Cereals.txt', 'Cheesecakes.txt', 'ChickenWings.txt', 'Chili.txt', 'Chinese.txt', 'Chowders.txt', 'Christmas.txt', 'CincodeMayo.txt', 'Crawfish.txt', 'Cucumbers.txt', 'CustardsandPuddings.txt', 'Diabetic.txt', 'Dips.txt', 'DipsandSpreads.txt', 'DogBiscuits.txt', 'Easter.txt', 'Eggs.txt', 'Fish.txt', 'Frosting.txt', 'Fruit.txt', 'German.txt', 'Greek.txt', 'HomeBrew.txt', 'Indian.txt', 'Italian.txt', 'JamsandJellies.txt', 'JustForFunRecipes.txt', 'Korean.txt', 'Lamb.txt', 'Marinades.txt', 'Mexican.txt', 'Muffins.txt', 'NonalcoholicBeverages.txt', 'Oysters.txt', 'Pasta.txt', 'Pastries.txt', 'PickledDishes.txt', 'Pies.txt', 'Popcorn.txt', 'Pork.txt', 'Potatoes.txt', 'Poultry.txt', 'QuickBreads.txt', 'Rice.txt', 'Sandwiches.txt', 'Sauces.txt', 'Shellfish.txt', 'Smoothies.txt', 'Soups.txt', 'Sourdough.txt', 'Spreads.txt', 'Stews.txt', 'Sweetbread.txt', 'Szechwan.txt', 'Thanksgiving.txt', 'Tomatoes.txt', 'Tuna.txt', 'Vegetarian.txt', 'VegetarianMainDishes.txt', 'YeastBreads.txt', 'Zucchini.txt']
input_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/recipe_text/'
output_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'

recipe_table = open(output_dir + 'recipes.txt', 'wb')
recipe_table.write('recipe_key\trecipe\n')
token_table = open(output_dir + 'tokens.txt', 'wb')
token_table.write('recipe_key\ttoken_key\ttarget\ttokens\ttoken_in_recipe_start\ttoken_in_recipe_end\tum_tokens\n')

recipe_count = 0
recipe_key = 0
token_table_size = 0
token_key = 0
target_sum = 0
target_count = 0
for recipe_type in filenames:
	with open(input_dir + recipe_type, 'r') as f:
		recipe_lines = f.readlines()

	recipe_txt = ' '.join(recipe_lines)
	# some recipes separate commands with semicolon, convert to conventional sentence
	callback = lambda pat: '. ' + pat.group(1).upper()
	recipe_txt = re.sub('(?i)[;]\s*([a-z])', callback, recipe_txt)
	# remove some punctuation that can through off sentence parsing
	recipe_txt = re.sub('["\(\)<>]', ' ', recipe_txt)
	split_recipes = [r.strip() for r in recipe_txt.split('* Exported from MasterCook *\r\n \r\n')]
	split_recipes = filter(None, split_recipes)

	# for each recipe remove list of ingredients
	with_breaks = [
		re.sub('\s+', ' ',
			re.sub('(\s*<br>\s*)+', ' <br> ',
				re.sub('-+', ' ',
					re.sub('- - - - - - - - - - - - - - - - - - -', ' ',
						re.sub('\r\n \r\n', ' <br> ', s))))) for s in split_recipes]
	# use breaks in recipes to filter out ingredients and serving size info
	new_recipes = [b.split('<br>') for b in with_breaks]
	new_recipes = [b[2:-1] for b in new_recipes]

	# final clean up the the instructions
	recipes = [filter(None, [d.strip() + '.' if d.strip()[-1] != '.' else d.strip() for d in dir]) for dir in new_recipes]
	recipe_count += len(recipes)
	# combine multiple instructions in a recipe into a single string
	# then write out each
	prev_recipe_key = -1
	for recipe_text in recipes:
		cumulative_token_ct = 0
		# combine multiple elements in a recipe list into a single string
		instructions = ' '.join(recipe_text)
		recipe_table.write(str(recipe_key) + '\t' + re.sub('[^a-z0-9\-/ ]', '', instructions.lower()).strip() + '\n')

		# begin splitting recipe into its sentences
		# sentences = filter(None, [s.strip() for s in re.split('(?<=(?<![0-9])[.!?])(\s+)(?=[A-Z0-9])', instructions)])
		sentences = filter(None, [s.strip() for s in re.split('(?i)(?<=(?<![0-9])[.!?])(\s+)(?=[A-Z0-9])', instructions)])
		prev_tokens = []
		for sentence_text in sentences:
			tokens = filter(None, [re.sub('[^a-z0-9\-/]', '', w.lower()).strip() for w in re.split('\s+', sentence_text)])
			num_tokens = len(tokens)

			# write out a short incomplete sentence (if it meets requirements
			if len(tokens) > 3:
				t = random.sample(xrange(2, len(tokens)), 1)[0]
				token_key += 1
				target = 0
				target_count += 1
				token_in_recipe_start = cumulative_token_ct
				token_in_recipe_end = cumulative_token_ct + t
				token_table.write(str(recipe_key) +'\t'+ str(token_key) +'\t'+
					str(target) +'\t'+ ' '.join(tokens[0:t]) +'\t'+
					str(token_in_recipe_start) +'\t'+ str(token_in_recipe_end) +'\t'+ str(t) +'\n')

			# write out the complete sentence
			token_key += 1
			target = 1
			target_count += 1
			target_sum += 1
			token_in_recipe_start = cumulative_token_ct
			token_in_recipe_end = cumulative_token_ct + len(tokens)
			token_table.write(str(recipe_key) +'\t'+ str(token_key) +'\t'+
				str(target) +'\t'+ ' '.join(tokens) +'\t'+
				str(token_in_recipe_start) +'\t'+ str(token_in_recipe_end) +'\t'+ str(len(tokens)) +'\n')

			# include 1 extra long sentence continuing off last sentence if possible
			# if recipe_key == prev_recipe_key and len(prev_tokens) > 0 and len(tokens) > 1 and random.random() < 0.75:
			# 	# write out
			# 	token_key += 1
			# 	target = 0
			# 	target_count += 1
			# 	t = random.sample(xrange(1, len(tokens)), 1)[0]
			# 	out_tokens = prev_tokens + tokens[0:t]
			# 	token_in_recipe_start = cumulative_token_ct
			# 	token_in_recipe_end = cumulative_token_ct + len(out_tokens)
			# 	token_table.write(str(recipe_key) +'\t'+ str(token_key) +'\t'+
			# 					  str(target) +'\t'+ ' '.join(out_tokens) +'\t'+
			# 					  str(token_in_recipe_start) +'\t'+ str(token_in_recipe_end) +'\t'+ str(len(out_tokens)) +'\n')

			cumulative_token_ct += num_tokens
			prev_recipe_key = recipe_key
			prev_tokens = tokens
		recipe_key += 1

recipe_table.close()
token_table.close()
print "number of recipes found:", recipe_count
print "number of lines in token table:", token_key
print "mean y:", round(1.0 * target_sum / target_count, 3)
print "pulled out of", len(filenames), "files"



