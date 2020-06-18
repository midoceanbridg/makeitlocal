# The Local Host
This project was created during my time at Insight (https://insightfellows.com/data-science).

It includes the following files

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

