import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from fuzzywuzzy import fuzz, process
from functools import lru_cache
import os

GENDIR = os.environ['LH_GENDIR'] 

@lru_cache()
def load_data():
    w2vm = pickle.load(open(f"{GENDIR}/model_w2v.pkl", 'rb')) #
    aisledict = pickle.load(open(f"{GENDIR}/ingredient_aisle.pkl", 'rb')) #
    noise = pickle.load(open(f"{GENDIR}/noiselist.pkl", 'rb')) #
    atFM = pickle.load(open(f"{GENDIR}/FMproducts.pkl", 'rb')) #
    FMinfo = pickle.load(open(f"{GENDIR}//FMfull.pkl", 'rb')) #
    ingvect = pickle.load(open(f"{GENDIR}/tfidfvect_ingredients.pkl", 'rb')) #
    ingfeatures = pickle.load(open(f"{GENDIR}/features_ingredients.pkl", 'rb'))
    fullnningredients = pickle.load(open(f"{GENDIR}/cleaned_ingredients.pkl", 'rb'))
    fulling = []
    fulling.extend([', '.join(n) for n in fullnningredients])

    recvect = pickle.load(open(f"{GENDIR}/tfidfvect_recipes.pkl", 'rb'))
    recdoc = pickle.load(open(f"{GENDIR}/full_recipedoc.pkl", 'rb'))
    recfeatures = pickle.load(open(f"{GENDIR}/features_recipes.pkl", 'rb'))

    return w2vm, aisledict, noise, atFM, FMinfo, ingvect, ingfeatures, fulling, recvect, recfeatures, recdoc

def request_comparison(userinput):
    mykey = open(f"{GENDIR}/apikey.txt").read().strip()
    params = {'url': userinput, 'forceExtraction': 'true', 'apiKey': mykey, 'analyze': 'true'}
    response = requests.get('https://api.spoonacular.com/recipes/extract', params=params)
    rec = response.json()
    ingcomp = rec['extendedIngredients']
    ingredients = [','.join([ing['name'].lower() for ing in ingcomp if ing['name'] ])]
    
    cur_rec = [rec['title'] + ' ' + ingredients[0] + ' ' + rec['instructions']]
    
    return ingredients[0], cur_rec

def removenoise(ingredients, noise): #call on ingredients[0]
    noise_free_ing = []
    for word in ingredients.split(','):
        checked = []
        splitit = word.split()
        checked.extend(i for i in splitit if i not in noise)
        noise_free_ing.append(' '.join(checked))
    return noise_free_ing

def rulesofsimilarity(noise_free_ing, w2vm, aisledict, atFM, FMinfo):
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
        'cos_sim': None #this is defined in the validation call
    }
    
    wheretoshop = {}

    allout = []

    for i in noise_free_ing:
        thisout = output.copy()
        thisout['ingredient'] = i
        
        highest = process.extractOne(i,atFM)
        if highest[1] >= 90:
            matchaisle = FMinfo.loc[FMinfo['TYPES OF PRODUCTS AVAILABLE'].str.contains(highest[0])]
            thebiz = matchaisle['BUSINESS NAME'].tolist()
            thisout['where_available'] = thebiz
        # print(f'{i} is available at {thebiz} ')
            
            found = False
            for vendor in thisout['where_available']:
                if vendor in wheretoshop:
                    wheretoshop[vendor].append(thisout['ingredient'])
                    found = True
                    break
            if not found:
                wheretoshop[thisout['where_available'][0]] = [thisout['ingredient']] 
            


        else:
            #these are unavailable ingredients
            curaisle = aisledict.get(i)
            
            #use w2v to find similar ingredients
            try:
                similar = w2vm.wv.most_similar(i, topn=100)
                opposite = w2vm.wv.most_similar(similar[0][0], topn=1000)
                #toreplace.append(i)
            # print(f'{i} is not but ')
            except KeyError:
                thisout['unknown'] = i
    #            print(f'We have never heard of {i}, sorry about that')
                continue
                
            #get rid of baking for now
            if curaisle[0] is not None and curaisle[0] == 'Baking':
                thisout['baking'] = True
            #   print(f'{i} is a baking product, which is likely in your pantry!')
                continue
                
            #deal with the seasoning issue
            if curaisle[0] is not None and curaisle[0] == 'Spices and Seasonings':
                matchaisle = FMinfo.loc[FMinfo['aisles'] == curaisle[0].lower()]
                thebiz = matchaisle['BUSINESS NAME'].tolist()
                #print(f'Dried Spices and and seasonings are rare, you may have this in your pantry, otherwise get fresh ones at: {thebiz}\n')
                thisout['spices'] = True
                thisout['spice_businesses'] = thebiz
                
                # if thebiz[0] not in wheretoshop:    
                #     wheretoshop[thebiz[0]] = ['SPICETIME']
                # elif 'SPICETIME' not in wheretoshop[thebiz[0]]:
                #     wheretoshop[thebiz[0]].append('SPICETIME')
                    
                
                continue
                
                
            # here is thing our algorithm thinks is similar and IS available
            item = []
            for opp in opposite:
                ophighest = process.extractOne(opp[0],atFM)
                if ophighest[1] >= 90:
                    item.append(opp[0])
                    if len(item) == 1:
                        break
            #print(f'Here is the item our algorithm thinks is most similar and is available: {item}\n')
            thisout['match'] = item
            matchaisle = FMinfo.loc[FMinfo['TYPES OF PRODUCTS AVAILABLE'].str.contains(ophighest[0])]
            thebiz = matchaisle['BUSINESS NAME'].tolist()
            thisout['store_hasreplacement'] = thebiz
            
            found = False
            for vendor in thisout['store_hasreplacement']:
                if vendor in wheretoshop:
                    wheretoshop[vendor].append(thisout['match'][0])
                    found = True
                    break
            if not found:
                wheretoshop[thisout['store_hasreplacement'][0]] = [thisout['match'][0]] 
            
            
            
            # if it is something usually prepackaged, suggest making it fresh
            if curaisle[0] == 'Pasta and Rice' or curaisle[0] == 'Canned and Jarred':
                for sim in similar:
                    a = aisledict[sim[0]]
                    if a and a[0] == 'Produce':
                        trythis = sim[0]
                        thisout['try_fresh'] = trythis
                        continue
                    
                    
                #print(f'Canned/Jarred items are rare at the Market, but you can make this fresh using {trythis}\n')
                
                    
            if curaisle[0] is not None: 
                #if something is not available, this vendor might be able to help you
                matchaisle = FMinfo.loc[FMinfo['aisles'] == curaisle[0].lower()]
                thebiz = matchaisle['BUSINESS NAME'].tolist()
            # print(f'This list of vendors often has products similar to {i}, try asking them: {thebiz}\n')
                if len(thebiz) > 0:
                    thisout['similar_vendor'] = thebiz



        allout.append(thisout)
    return allout, wheretoshop

def validationstep(allout, fulling, ingvect, ingfeatures, recvect, recfeatures, recdoc, cur_rec):

    initinglist = []
    thingstoremove = []
    thingstoadd = []
    for out in allout:
        initinglist.append(out['ingredient'])
        if out['where_available'] is None:
            thingstoremove.append(out['ingredient'])
            thingstoadd.append(out['match'][0])


    for rem, add in zip(thingstoremove, thingstoadd):
        newlist = initinglist.copy()
        newlist.remove(rem)
        newlist.append(add)

     
        #now based on ingredients alone find the most similar recipe we know
        newlistj = [', '.join(newlist)]
        nsf = ingvect.transform(newlistj)
        cosine_similarities = linear_kernel(nsf, ingfeatures).flatten()
        related_rec_index = cosine_similarities.argsort()[-1]
        

        #now find out, based on more features, how similar these two recipes are
        currecfeat = recvect.transform(cur_rec)
        recipe_similarity = linear_kernel(currecfeat, recfeatures[related_rec_index]).flatten()
        
        for out in allout:
            if out['ingredient'] == rem:
                out['cos_sim'] = recipe_similarity
       
            
