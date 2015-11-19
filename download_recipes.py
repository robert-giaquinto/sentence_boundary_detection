from __future__ import division
import os
from urllib2 import urlopen, URLError, HTTPError


def dlfile(url, data_dir):
	# Open the url
	try:
		f = urlopen(url)
		print "downloading " + url

		# Open our local file for writing
		with open(data_dir + os.path.basename(url), "wb") as local_file:
			local_file.write(f.read())

	# handle errors
	except HTTPError, e:
		print "HTTP Error:", e.code, url
	except URLError, e:
		print "URL Error:", e.reason, url

filenames = ['AlcoholicBeverages.txt', 'AllAppetizerRecipes.txt', 'AllBeverageRecipes.txt', 'AllBreadRecipes.txt', 'AllBreakfastRecipes.txt', 'AllCondimentRecipes.txt', 'AllCookieRecipes.txt', 'AllDesertRecipes.txt', 'AllSaladRecipes.txt', 'AllSoupandStewRecipes.txt', 'AllVegetableandSideDishRecipes.txt', 'Asparagus.txt', 'BakedGoods.txt', 'Beef.txt', 'Biscotti.txt', 'BiscottiRecipes.txt', 'BiscuitsandScones.txt', 'BreadMachineRecipes.txt', 'Brownies.txt', 'Cajun.txt', 'Cakes.txt', 'Candy.txt', 'Casseroles.txt', 'Cereals.txt', 'Cheesecakes.txt', 'ChickenWings.txt', 'Chili.txt', 'Chinese.txt', 'Chowders.txt', 'Christmas.txt', 'CincodeMayo.txt', 'Crawfish.txt', 'Cucumbers.txt', 'CustardsandPuddings.txt', 'Diabetic.txt', 'Dips.txt', 'DipsandSpreads.txt', 'DogBiscuits.txt', 'Easter.txt', 'Eggs.txt', 'Fish.txt', 'Frosting.txt', 'Fruit.txt', 'German.txt', 'Greek.txt', 'HomeBrew.txt', 'Indian.txt', 'Italian.txt', 'JamsandJellies.txt', 'JustForFunRecipes.txt', 'Korean.txt', 'Lamb.txt', 'Marinades.txt', 'Mexican.txt', 'Muffins.txt', 'NonalcoholicBeverages.txt', 'Oysters.txt', 'Pasta.txt', 'Pastries.txt', 'PickledDishes.txt', 'Pies.txt', 'Popcorn.txt', 'Pork.txt', 'Potatoes.txt', 'Poultry.txt', 'QuickBreads.txt', 'Rice.txt', 'Sandwiches.txt', 'Sauces.txt', 'Shellfish.txt', 'Smoothies.txt', 'Soups.txt', 'Sourdough.txt', 'Spreads.txt', 'Stews.txt', 'Sweetbread.txt', 'Szechwan.txt', 'Thanksgiving.txt', 'Tomatoes.txt', 'Tuna.txt', 'Vegetarian.txt', 'VegetarianMainDishes.txt', 'YeastBreads.txt', 'Zucchini.txt']
base_url = 'http://mc6help.tripod.com/RecipeLibrary/'
output_dir = '/Users/robert/Documents/UMN/5551_Robots/Project/data/'

for f in filenames:
	dlfile(base_url + f, output_dir)


