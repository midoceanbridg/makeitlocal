import requests
from sklearn.manifold import TSNE
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

def load_data():
    # TODO: make this come from env variable
    vec = pickle.load(open("../bigoven/vect.pkl", 'rb'))
    features = pickle.load(open("../bigoven/features.pkl", 'rb'))
    recipes = pickle.load(open("../bigoven/recipes.pkl", 'rb'))
    fmproducts = pickle.load(open("../bigoven/FMproducts.pkl", 'rb'))
    return vec, features, recipes, fmproducts

def request_comparison(userinput):
    mykey = open('../spoonac/apikey.txt').read().strip()
    params = {'url': userinput, 'forceExtraction': 'true', 'apiKey': mykey, 'analyze': 'true'}
    response = requests.get('https://api.spoonacular.com/recipes/extract', params=params)
    return response

def input_to_data(response, vec, fmproducts):
    rec = response.json()
    ingcomp = rec['extendedIngredients']
    ingredients = [', '.join([ing['name'].lower() for ing in ingcomp ])]
    both = set(ingredients).intersection(fmproducts)
    percent = len(both)/len(ingredients)*100
    new_features = vec.transform(ingredients)
    return new_features, percent 

def fetch_similar(new_features, features, vec, recipes, percent):
    cosine_similarities = linear_kernel(new_features, features).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    similar = [[recipes.iloc[i]['percent'] , recipes.iloc[i]['WebURL']] for i in related_docs_indices]
    return similar