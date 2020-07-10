# The Local Host
*This project was created during my time at Insight (https://insightfellows.com/data-science).*

The aim of this project is to make it easier for users to do the majority of their shopping & meal planning via NYC farmers markets. 

__You can view the website at www.thelocalhost.nyc__ 

Users input a link to a recipe they want to make, and my code:
*  Uses filtering to assess if the ingredient is available locally
* If not locally available, uses combined word2vec output and a dictionary of aisle information to replace it with a suitable local substitute
* Validates the appropriateness of the substitution using cosine similarity of full input recipe vs a recipe using the replacement ingredients
* Provides a farmers market specific grocery list.


Inside greenmarkets/ you will find the following files:

* __0_fetch_bigovenapi.ipynb__ a notebook for saving recipes from the BigOven API
  * You will need your own BigOven API key to run this file
* __0_fetch_recipesspoontacular.ipynb__ a notebook for saving recipes from Spoonacular API  
  * You will need your own Spoonacular API key to run this file
* __1_cleandata.ipynb__ takes the data generated from the above notebooks and puts them into useful formats for analysis
* __2_ModelingW2V.ipynb__ runs Word2Vec model on ingredient data generated above
* __2_ModelingW2V.ipynb__ runs TFIDF and Cosine Similarity for validation 
* __3_validation_analysis.ipynb__ shows validation results from user test


Inside greenmarkets/eatlocal
* The Dockerfile and requirements necessary to put on AWS

Inside greenmarkets/eatlocal/thelocalhost
* __localeats_twostage.py__ 
  * Allows for the input of a URL
  * assesses which ingredients are available at the farmers markets
  * uses W2V modeling to swap out ingredients where necessary
  * uses TD-IDF & Cosine Similarity to validate
  * Outputs new ingredient list

* __app.py__ Which calls the __localeats_twostage.py__ file
* Templates & items to make the website less ugly
