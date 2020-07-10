import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from fuzzywuzzy import fuzz, process
from functools import lru_cache
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
from nltk.corpus import stopwords 
import nltk
import os

GENDIR = os.environ['LH_GENDIR']
with open(f'{GENDIR}/english') as fh:
    STOP_WORDS = set(fh.read().split('\n'))


def get_results(ingredients, cur_rec):
    w2vm, aisledict, noise, atFM, FMinfo, ingvect, ingfeatures, fulling, recvect, recfeatures, recdoc = load_data()
    noise_free_ing = removenoise(ingredients, noise)
    allout, wheretoshop = rulesofsimilarity(noise_free_ing, w2vm, aisledict, atFM, FMinfo) 
    validationstep(allout, fulling, ingvect, ingfeatures, recvect, recfeatures, recdoc, cur_rec, FMinfo, wheretoshop)
    
    return allout, wheretoshop

#function definiton block


@lru_cache()
def load_data():
    ''' Load previously generated data, all of this can be created via the other notebooks in this codebase
    except the farmers market data, which can be found here:
    https://docs.google.com/spreadsheets/d/1MOWl8Cg4xyCvAmR06cFhJ9obYR5ToZD_XhSEcgekjzY/edit#gid=1829695724'''
    w2vm = pickle.load(open(f"{GENDIR}/model_w2v.pkl", 'rb')) #
    aisledict = pickle.load(open(f"{GENDIR}/ingredient_aisle.pkl", 'rb')) #
    noise = pickle.load(open(f"{GENDIR}/noiselist.pkl", 'rb')) #
    atFM = pickle.load(open(f"{GENDIR}/FMproducts.pkl", 'rb')) #
    FMinfo = pickle.load(open(f"{GENDIR}/FMfull.pkl", 'rb')) #
    
    ingvect = pickle.load(open(f"{GENDIR}/tfidfvect_ingredients.pkl", 'rb')) #
    ingfeatures = pickle.load(open(f"{GENDIR}/features_ingredients.pkl", 'rb'))
    fullnningredients = pickle.load(open(f"{GENDIR}/cleaned_ingredients.pkl", 'rb'))
    fulling = []
    fulling.extend([', '.join(n) for n in fullnningredients])

    recvect = pickle.load(open(f"{GENDIR}/tfidfvect_recipes.pkl", 'rb'))
    recfeatures = pickle.load(open(f"{GENDIR}/features_recipes.pkl", 'rb'))
    recdoc = pickle.load(open(f"{GENDIR}/full_recipedoc.pkl", 'rb'))

    return w2vm, aisledict, noise, atFM, FMinfo, ingvect, ingfeatures, fulling, recvect, recfeatures, recdoc

def request_comparison(userinput):
    '''Takes the user input recipe URL and extracts the recipe information 
    Note you must have a spoonacular API key for this to work
    
    Returns a comma separated string of ingredients and a list of strings including the recipe title, ingredients and instructions
    '''
    #grabbing info using Spoonacular
    mykey = open(f'{GENDIR}/spoonapikey.txt').read().strip()
    params = {'url': userinput, 'forceExtraction': 'true', 'apiKey': mykey, 'analyze': 'true'}
    response = requests.get('https://api.spoonacular.com/recipes/extract', params=params)
    #parsing output
    rec = response.json()
    ingcomp = rec['extendedIngredients']
    ingredients = [lemmatizer.lemmatize(re.sub(r'[^a-z ]', '', ing['name'].lower().strip())) for ing in ingcomp if ing['name'] ]

    #error handling
    if '404' in rec['title']: #they don't return an actual 404 which is annoying
        cur_rec = None
    else:
        cur_rec = ''
        if rec['title'] is not None:
            cur_rec += rec['title']
        cur_rec += ','.join(ingredients)
        if rec['instructions'] is not None:
            cur_rec += rec['instructions']
        cur_rec = [cur_rec]
    
    return ingredients, cur_rec

def removenoise(ingredients, noise): 
    '''Do some simple noise removal on the loaded ingredients 
    
    returns the cleaned ingredient string
    
    '''
  
    assert type(ingredients) == list, type(ingredients)
    noise_free_ing = []
    for word in ingredients:
        checked = []
        splitit = word.split()
        for it in splitit:
            if it not in noise and it not in STOP_WORDS:
                tag = nltk.pos_tag([it])
                if tag[0][1] in ['JJ', 'NN', 'NNP', 'VBN']: #check if it is a adj, noun, proper noun
                    checked.append(it)
                  
                elif tag[0][1] in ['JJR', 'JJS', 'NNS', 'NNPS']: # check if adj, noun, propr noun, but comparitive or plural
                    checked.append(lemmatizer.lemmatize(it)) #lemmatize these words
                    
        if len(checked) > 0:
            noise_free_ing.append(' '.join(list(filter(None, checked))))
    return noise_free_ing
    
    
    #     noise_free_ing = []
#     for word in ingredients.split(','):
#         checked = []
#         splitit = word.split()
#         checked.extend(i for i in splitit if i not in noise)
#         noise_free_ing.append(' '.join(checked))

#     return noise_free_ing

def rulesofsimilarity(noise_free_ing, w2vm, aisledict, atFM, FMinfo):
    '''Goes through ingredients, assesses the similarity of them
    
    Returns dicts of the ingredient info compiled and the shopping list dict
    
    '''
    combineding = noise_free_ing.copy()
    
    #Create starting dict to be filled out
    output = {
        'ingredient': None,
        'where_available': None,
        'unknown': None, #unknown to word2vec
        'baking': False, #true if is baking
        'spices': False, #true if spice/seasoning
        'spice_businesses': None,
        'match': None,
        'similar_vendor': None,
        'try_fresh': None,
        'store_hasreplacement': None,
        'aisle': None,
        'cos_sim': None #this is defined in the validation call
    }
    
    #Shopping list dict
    wheretoshop = {}
    allout = []

    for i in noise_free_ing:
        
        thisout = output.copy()
        thisout['ingredient'] = i
        
        
        #Figure out if it is available at the farmers market
        highest = process.extractOne(i,atFM)
        if highest[1] >= 90:
            handle_atFM(highest[0], FMinfo, thisout, wheretoshop)
            
        #If it is NOT available
        else:  
            #find what aisle it belongs in
            curaislelist = aisledict.get(i)
            curaisles = handle_toaisles(curaislelist)
            
            thisout['aisle'] = curaisles
            #assess if it is a pantry item
            ispantry = handle_pantry(thisout, curaisles, FMinfo)
            if ispantry:
                allout.append(thisout)
                continue
                    
            #use w2v to find similar ingredients
            try:
                similar = w2vm.wv.most_similar(i, topn=5)
                opposite = w2vm.wv.most_similar(similar[0][0], topn=100)
            # Just in case we find an entirely new ingredient!
            except KeyError: 
                thisout['unknown'] = i
                allout.append(thisout)
                continue
                
            #now go through these suggestions, see what's at the market and is in the right aisle    
            item = []
            handle_matching(thisout, opposite, atFM, FMinfo, item, combineding, curaisles, aisledict)
            
            #place the item into the shopping list with a vendor suggestion
            if thisout['store_hasreplacement'] is not None:
                handle_shoppinglist(thisout, wheretoshop)
            
            
            # If it is something usually prepackaged, suggest making it fresh. If it is unavailable, find a helpful vendor
            if curaisles is not None:
                 handle_tryfresh(thisout, curaisles, aisledict, opposite)
 
        allout.append(thisout)
    return allout, wheretoshop

def validationstep(allout, fulling, ingvect, ingfeatures, recvect, recfeatures, recdoc, cur_rec, FMinfo, wheretoshop):
    '''Validate how well our suggestions fit based on cosine similarity of entire recipe'''
    initinglist = []
    thingstoremove = []
    thingstoadd = []
    #loop through ingredients
    for out in allout:
        initinglist.append(out['ingredient'])
        # If it is an item we have swapped out
        if out['where_available'] is None and out['match'] is not None and 'No Match' not in out['match']:
            thingstoremove.append(out['ingredient'])
            thingstoadd.append(out['match'][0])

    # create a new ingredient list withe swapping in place
    for rem, add in zip(thingstoremove, thingstoadd):
        newlist = initinglist.copy()
        newlist.remove(rem)
        newlist.append(add)

        # based on ingredients alone find the most similar recipe we know
        newlistj = [', '.join(newlist)]
        nsf = ingvect.transform(newlistj)
        cosine_similarities = linear_kernel(nsf, ingfeatures).flatten()
        related_rec_index = cosine_similarities.argsort()[-1]
        

        #now find out, based on more features, how similar these two recipes are
        currecfeat = recvect.transform(cur_rec)
        recipe_similarity = linear_kernel(currecfeat, recfeatures[related_rec_index]).flatten()
        
        # take what we have calculated and place it into our dicts
        for out in allout:
            if out['ingredient'] == rem:
                out['cos_sim'] = recipe_similarity
                if recipe_similarity < 0.2:
                    aisle = out['aisle']
                    handle_notvalid(out, FMinfo, aisle, wheretoshop)
            if out['match'] is not None and 'No Match' in out['match']:
                out['cos_sim'] = 0
        
            
# Helper functions!

def handle_atFM(highest, FMinfo, thisout, wheretoshop):
    '''When an item IS available at the famers market this function figures out where and fills out the info dict appropriately'''
    matchaisle = FMinfo.loc[FMinfo['TYPES OF PRODUCTS AVAILABLE'].str.contains(highest)]
    thebiz = matchaisle['BUSINESS NAME'].tolist()
    thisout['where_available'] = thebiz

    # place in shopping list
    found = False
    for vendor in thisout['where_available']:
        if vendor in wheretoshop:
            wheretoshop[vendor].append(thisout['ingredient'])
            found = True
            break
    if not found:
        wheretoshop[thisout['where_available'][0]] = [thisout['ingredient']] 
        
def handle_pantry(thisout, curaisles, FMinfo):
    '''returning true if this ingredient is considered baking or a spice''' 
    
    #Deal with the baking outliers
    ispantry = False
    if curaisles is not None and 'baking' in curaisles:
        thisout['baking'] = True
        ispantry = True

    #Deal with the seasoning outliers
    if curaisles is not None and 'spices and seasonings' in curaisles:          
        matchaisle = FMinfo.loc[FMinfo['aisles'] == 'spices and seasonings']
        thebiz = matchaisle['BUSINESS NAME'].tolist()
        thisout['spices'] = True
        thisout['spice_businesses'] = thebiz
        ispantry = True
    return ispantry


def handle_matching(thisout, opposite, atFM, FMinfo, item, combineding, curaisles, aisledict):
    ''' Finding replacement for when it is not available at FM'''
    #loop through all possibilities
    for opp in opposite:
        
        # first extract aisle from this potential replacement ingredient
        opaislelist = aisledict.get(opp[0])
        opaisles = handle_toaisles(opaislelist)

        
        #assess if it is placed in the same aisle as the thing we want to replace
        shelved = False
        if opaisles is None and curaisles is None:
            shelved = True
        elif opaisles is not None and curaisles is not None:
            if opaisles.intersection(curaisles):
                shelved = True
        
        #if it IS in the same aisle, see if it is at the FM (we do this last cause FW is slow even with the c speedup)        
        if shelved:
        
            exactmatch = opp[0] in atFM #first see if exact match, again cause FW is slow
            if exactmatch:
                ophighest = (opp[0], 100)
            else:
                ophighest = fw_forcache(opp[0], tuple(atFM))

            item = []
            if ophighest[1] >= 90 and ophighest[0] not in combineding:
                #double check that FW didn't put us in a different aisle!
                matchedaislelist = aisledict.get(ophighest[0])
                matchedaisles = handle_toaisles(matchedaislelist)
                if matchedaisles is not None and curaisles is not None:
                    if matchedaisles.intersection(curaisles):
                        item.append(ophighest[0])
                        break
                elif matchedaisles is None and curaisles is None:
                    item.append(ophighest[0])
                    break
                    
                    
    #once we have found the replacement, put it into our dicts 
    if len(item) > 0:
        thisout['match'] = item
        matchaisle = FMinfo.loc[FMinfo['TYPES OF PRODUCTS AVAILABLE'].str.contains(item[0])]
        thebiz = matchaisle['BUSINESS NAME'].tolist()
        thisout['store_hasreplacement'] = thebiz
        combineding.append(thisout['match'][0])
    else:
        thisout['match'] = 'No Match'

        
def handle_shoppinglist(thisout, wheretoshop):
    ''' updates the shopping list dict with items'''
    
    found = False
    for vendor in thisout['store_hasreplacement']:
        if vendor in wheretoshop:
            wheretoshop[vendor].append(thisout['match'][0])
            found = True
            break
    if not found:
        wheretoshop[thisout['store_hasreplacement'][0]] = [thisout['match'][0]]


def handle_tryfresh(thisout, curaisles, aisledict, opposite):
    '''In the case that something is usually packaged but COULD be made from scratch, provide that info'''
    
    if 'pasta and rice' in curaisles or 'canned and jarred' in curaisles:
        for opp in opposite:
            a = aisledict[opp[0]]
            if a and a[0] == 'Produce':
                trythis = opp[0]
                thisout['try_fresh'] = trythis

                
def handle_notvalid(thisout, FMinfo, curaisles, wheretoshop):
    '''if something is not available, find a vendor that might be able to help '''
    
    suggestion = 'Ask about ' + thisout['ingredient']
    #if there is an aisle associated with the item
    if curaisles is not None: 
        # find a similar aisle at the farmers market and point to it
        for shelf in curaisles:
            matchaisle = FMinfo.loc[FMinfo['aisles'] == shelf]
            if len(matchaisle) > 0:
                thebiz = matchaisle['BUSINESS NAME'].tolist()
                thisout['similar_vendor'] = thebiz
                found = False
                for vendor in thisout['similar_vendor']:
                    if vendor in wheretoshop:
                        wheretoshop[vendor].append(suggestion)
                        found = True
                        break
                if not found:
                    wheretoshop[thisout['similar_vendor'][0]] = [suggestion]
                    

@lru_cache(maxsize=5000)
def fw_forcache(opp, relevant_atFM):
    ophighest = process.extractOne(opp,relevant_atFM) #otherwise use FW
    return ophighest


def handle_toaisles(aisle_grabbedfromdict):
    if aisle_grabbedfromdict is not None and aisle_grabbedfromdict[0] is not None:
        tostring = aisle_grabbedfromdict[0]
        matchedaisles = set(tostring.lower().split(';'))
    else:
        matchedaisles = None
    return matchedaisles
        
        
