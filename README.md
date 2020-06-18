# The Local Host
*This project was created during my time at Insight (https://insightfellows.com/data-science).*

The aim of this project is to make it easier for users to do the majority of their shopping & meal planning via NYC farmers markets. 

__You can view the website at www.thelocalhost.nyc__ 

Users input a link to a recipe they want to make, and my code:
*  Uses NLP tools to assess
 * If the ingredient is available locally
 * If not, replace it with a suitable local substitute
* Validates (also) using NLP that the substitute is reasonable
* Provides a farmers market specific grocery list.


Inside greenmarkets/ you will find the following files:

* __0_fetch_bigovenapi.ipynb__ a notebook for saving recipes from the BigOven API
  * You will need your own BigOven API key to run this file
* __0_fetch_recipesspoontacular.ipynb__ a notebook for saving recipes from Spoonacular API  
  * You will need your own Spoonacular API key to run this file
* __1_cleandata.ipynb__ takes the data generated from the above notebooks and puts them into useful formats for analysis
* __2_ModelingW2V.ipynb__ runs Word2Vec model on ingredient data generated above
* __2_ModelingW2V.ipynb__ runs TFIDF and Cosine Similarity for validation 
* __production_notebook.ipynb__ 
  * Allows for the input of a URL
  * assesses which ingredients are available at the farmers markets
  * uses W2V modeling to swap out ingredients where necessary
  * uses TD-IDF & Cosine Similarity to validate
  * Outputs new ingredient list



Inside greenmarkets/eatlocal
* The Dockerfile and requirements necessary to put on AWS

Inside greenmarkets/eatlocal/thelocalhost
* __localeats_twostage.py__ Which is essentially the same as the __production_notebook__ described above
* __app.py__ Which calls the __localeats_twostage.py__ file
* Templates & items to make the website less ugly
