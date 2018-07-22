#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:53:33 2017

@author: mahipal
"""
import pandas as pd
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from PIL import Image
import requests
import shutil
from io import BytesIO
from scipy.sparse import hstack
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense



def load_core():
    try:
        data = pd.read_pickle('my_pickle/180000_filtered_data.pkl')
    except:
        
        data = pd.read_json("tops_fashion.json")
        print("Number of sample points: %d \n Number of columns: %d" %(data.shape[0], data.shape[1]))
        #print("Columns are: %s" %(data.columns))
        
        data = data[["asin", "brand", "color", "medium_image_url","title", "formatted_price", "product_type_name"]]
        print("\nNumber of sample points: %d \n Number of columns: %d" %(data.shape[0], data.shape[1]))
        #print(data.head())
        
        print("\nProduct type description: ",Counter(list(data.product_type_name)).most_common(10))
        print("\nBrand Name description: ",Counter(list(data.brand)).most_common(10))
        print("\nColor description: ",Counter(list(data.color)).most_common(10))
        
        data.to_pickle('my_pickle/180000_filtered_data.pkl')
    return data
    

def removeNullPriceColor(data):
    try:
        data = pd.read_pickle("my_pickle/28000_filtered_data")
    except:
        #data = load_core()
        #Eliminate null price and null color data points
        data = data[~data.formatted_price.isnull()]
        print("\nNumber of data sample after eliminating null price: ",data.shape[0] )
        
        data = data[~data.color.isnull()]
        print("\nNumber of data sample after eliminating null color: ",data.shape[0] )
        
        data.to_pickle("my_pickle/28000_filtered_data")
    return data



def remove_duplicateFromEnd(data):
    try:
        data = pd.read_pickle('my_pickle/17000_dup1done')
    except:
        
        #data = removeNullPriceColor()
        #Observe duplicate products with the help of title to avoid recommendation of same product
        print("\nTotal duplicate product: ", data.duplicated('title').sum())
        
        #Remove product with less than 4 words in title
        data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
        print("After removal of products with short description:", data_sorted.shape[0])
            
        data_sorted.sort_values('title',inplace=True, ascending=False)
        data_sorted.head()
        
        indices = []
        for i,row in data_sorted.iterrows():
            indices.append(i)
            
            
        stage1_dedupe_asins = []
        i = 0
        j = 0
        num_data_points = data_sorted.shape[0]
        while i < num_data_points and j < num_data_points:
            
            previous_i = i
        
            # store the list of words of ith string in a, ex: a = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
            a = data['title'].loc[indices[i]].split()
        
            # search for the similar products sequentially 
            j = i+1
            while j < num_data_points:
        
                # store the list of words of jth string in b, ex: b = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'Small']
                b = data['title'].loc[indices[j]].split()
        
                # store the maximum length of two strings
                length = max(len(a), len(b))
        
                # count is used to store the number of words that are matched in both strings
                count  = 0
        
                # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
                # example: a =['a', 'b', 'c', 'd']
                # b = ['a', 'b', 'd']
                # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
                for k in itertools.zip_longest(a,b): 
                    if (k[0] == k[1]):
                        count += 1
        
                # if the number of words in which both strings differ are > 2 , we are considering it as those two apperals are different
                # if the number of words in which both strings differ are < 2 , we are considering it as those two apperals are same, hence we are ignoring them
                if (length - count) > 2: # number of words in which both sensences differ
                    # if both strings are differ by more than 2 words we include the 1st string index
                    stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])
        
                    # if the comaprision between is between num_data_points, num_data_points-1 strings and they differ in more than 2 words we include both
                    if j == num_data_points-1: stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[j]])
        
                    # start searching for similar apperals corresponds 2nd string
                    i = j
                    break
                else:
                    j += 1
            if previous_i == i:
                break
            
        
        data = data.loc[data['asin'].isin(stage1_dedupe_asins)]
        print('Number of data points : ', data.shape[0])
        data.to_pickle('my_pickle/17000_dup1done')
    return data

def remove_duplicateMatchingKeyword(data):
    try:
        data = pd.read_pickle('my_pickle/16000_dup2done')
    except:
        #data = remove_duplicateFromEnd()
        
            
        indices = []
        for i,row in data.iterrows():
            indices.append(i)
        
        stage2_dedupe_asins = []
        while len(indices)!=0:
            i = indices.pop()
            stage2_dedupe_asins.append(data['asin'].loc[i])
            # consider the first apperal's title
            a = data['title'].loc[i].split()
            # store the list of words of ith string in a, ex: a = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
            for j in indices:
                
                b = data['title'].loc[j].split()
                # store the list of words of jth string in b, ex: b = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
                
                length = max(len(a),len(a))
                
                # count is used to store the number of words that are matched in both strings
                count  = 0
        
                # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
                # example: a =['a', 'b', 'c', 'd']
                # b = ['a', 'b', 'd']
                # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
                for k in itertools.zip_longest(a,b): 
                    if (k[0]==k[1]):
                        count += 1
        
                # if the number of words in which both strings differ are < 3 , we are considering it as those two apperals are same, hence we are ignoring them
                if (length - count) < 3:
                    indices.remove(j)
        
        data = data.loc[data['asin'].isin(stage2_dedupe_asins)]
        print('Number of data points after stage two of dedupe: ',data.shape[0])
        
        data.to_pickle('my_pickle/16000_dup2done')
        
    return data

#Text Processing

def removeStopWords(text, data, index, column):
    stop_words = set(stopwords.words('english'))
    if type(text) is not int:
        string = ""
        for words in text.split():
            word = ("".join(e for e in words if e.isalnum()))
            word = word.lower()
            if not word in stop_words:
                string += word + " "
        data[column][index] = string
    
    return data

def text_preprocessing(data):
    
    for index, row in data.iterrows():
        data = removeStopWords(row['title'], data, index, 'title') #data updating after every iteration, dont worry about that
    
    data.to_pickle('my_pickle/stopWordRemoved')
    
    return data

def bow_vector(data):
    title_vectorizer = CountVectorizer()
    vectorized_matrix = title_vectorizer.fit_transform(data)
    return vectorized_matrix


def tfidf_vector(data):
    title_vectorizer = TfidfVectorizer()
    vectorized_matrix = title_vectorizer.fit_transform(data)
    return vectorized_matrix

def n_containing(word, data):
    # return the number of documents which had the given word
    return sum(1 for blob in data['title'] if word in blob.split())

def idf(word, data):
    # idf = log(#number of docs / #number of docs which had the given word)
    return math.log(data.shape[0] / (n_containing(word, data)))

def idf_vector(data):
    idf_title_vectorizer = CountVectorizer()
    idf_title_features = idf_title_vectorizer.fit_transform(data['title'])
    idf_title_features  = idf_title_features.astype(np.float)

    for i in idf_title_vectorizer.vocabulary_.keys():
        #idf_title_vectorizer.vocabulary_.keys() <= this contains 12609 unique words and we are going to find its idf value
        # for every word in whole corpus we will find its idf value
        idf_val = idf(i, data)
    
        # to calculate idf_title_features we need to replace the count values with the idf values of the word
        # idf_title_features[:, idf_title_vectorizer.vocabulary_[i]].nonzero()[0] will return all documents in which the word i present
        for j in idf_title_features[:, idf_title_vectorizer.vocabulary_[i]].nonzero()[0]:
            
            # we replace the count values of word i in document j with  idf_value of word i 
            # idf_title_features[doc_id, index_of_word_in_courpus] = idf value of word
            idf_title_features[j,idf_title_vectorizer.vocabulary_[i]] = idf_val
            
    
    #idf_title_vectorizer.vocabulary_.keys() => contains all words
    #idf_title_vectorizer.vocabulary_['faded'] => it will give index of faded
    #idf_title_features[:, idf_title_vectorizer.vocabulary_['faded']].nonzero()[0] will give array of documents which contain word faded
    #
    return idf_title_features



def similar_title(vectorized_matrix, id, numResult):
    pairwise_dist = pairwise_distances(vectorized_matrix, vectorized_matrix[id])  
    
    indices = np.argsort(pairwise_dist.flatten())[0:numResult]
    distances = np.sort(pairwise_dist.flatten())[0:numResult]
    
    return indices, distances

def showImage(url):
    response = requests.get(url)
    Image.open(BytesIO(response.content)).show()  #can also be done with StringIO
    return  


def getInfoByIndex(data, indices, query_index, distances):
    
    print("="*60)
    print("="*60)
    print("="*60)
    print("Info of queried product:")
    print('ASIN :',data['asin'].loc[query_index])
    print ('Brand:', data['brand'].loc[query_index])
    print ('Title:', data['title'].loc[query_index])
    print("="*60)
    print("="*60)
    print("="*60)
    
    for i in range(len(indices)):
        url = data['medium_image_url'].loc[indices[i]]
        showImage(url)
        print("Similar product: ", i+1)
        print('ASIN :',data['asin'].loc[indices[i]])
        print ('Brand:', data['brand'].loc[indices[i]])
        print ('Title:', data['title'].loc[indices[i]])
        print ('Euclidean similarity with the query image :', distances[i])
        print("="*60)
        
    return

def word2vec(data):
    import pickle
    
    try:
       with open('required_word_vec.pickle', 'rb') as handle:
           required_word_vec = pickle.load(handle)
    except:
        print("importing google news")
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        required_word_vec = {}
        for title in data:
            words = title.split()
            for word in words:
                try:    
                    required_word_vec[word] = model.wv[word]
                except:
                    required_word_vec[word] = 0
                    
        
        
        with open('required_word_vec.pickle', 'wb') as handle:
            pickle.dump(required_word_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return required_word_vec


def avg_individual(title, vectorized_matrix):
    
    avg_vec = np.zeros((300))
    words = title.split()
    for word in words:
        avg_vec += vectorized_matrix[word]
    
    avg_vec /= len(words)
    return avg_vec

def weighted_avg_individual(title, vectorized_matrix, data):
    
    avg_vec = np.zeros((300))
    words = title.split()
    for word in words:
        avg_vec += vectorized_matrix[word]*idf(word, data)
    
    avg_vec /= len(words)
    return avg_vec

def build_avg(titles, vectorized_matrix, method, data):
    w2v_title = []
    
    if method == 'average':
        for title in titles:
            w2v_title.append(avg_individual(title, vectorized_matrix))
        
    elif method == 'weighted':
         for title in titles:
            w2v_title.append(weighted_avg_individual(title, vectorized_matrix, data))

    return np.array(w2v_title)

def similar_title_from_w2v(avg_vec_title, id, numResult):
    
    pairwise_dist = pairwise_distances(avg_vec_title, avg_vec_title[id].reshape(1,-1))  
    
    indices = np.argsort(pairwise_dist.flatten())[0:numResult]
    distances = np.sort(pairwise_dist.flatten())[0:numResult]
    
    return indices, distances
    

def addColorAndBrands(data):
    
    data.brand.fillna('Not given', inplace=True)
    
    brands = [x.replace(" ", "-") for x in data.brand.values]
    colors = [x.replace(" ", "-") for x in data.color.values]
    types = [x.replace(" ", "-") for x in data.product_type_name.values]
    
    brand_vectorizer = CountVectorizer()
    brand_features = brand_vectorizer.fit_transform(brands)

    type_vectorizer = CountVectorizer()
    type_features = type_vectorizer.fit_transform(types)
    
    color_vectorizer = CountVectorizer()
    color_features = color_vectorizer.fit_transform(colors)

    extra_features = hstack((brand_features, type_features, color_features)).tocsr()
    
    return extra_features

def similarProduct(idf_vec_title, extra_features, id, numResult, w1, w2):
    
    idf_w2v_dist  = pairwise_distances(idf_vec_title, idf_vec_title[id].reshape(1,-1))
    ex_feat_dist = pairwise_distances(extra_features, extra_features[id])
    pairwise_dist   = (w1 * idf_w2v_dist +  w2 * ex_feat_dist)/float(w1 + w2)
    
    indices = np.argsort(pairwise_dist.flatten())[0:numResult]
    
    distances  = np.sort(pairwise_dist.flatten())[0:numResult]

    return indices, distances

def download_all_img():
    df = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
    img_url = df.medium_image_url
    asin_list = df.asin
    
    for i, url in enumerate(img_url):
        response = requests.get(url, stream=True)
        with open('16k_all_img/'+asin_list.iloc[i]+'.'+url.split('.')[-1], 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        print('image downloaded: ',i)
        del response
    return



def load_img_features():
    try:
        last_layer_data = np.load('numpy/16k_data_vgg16.npy')
        asins = np.load('numpy/16k_data_cnn_info.npy')
    except:
        
        img_width, img_height = 224, 224
        train_data_dir = '16k_images/'
        nb_train_samples = 16042
        batch_size = 1
        
        asins = []  #for retriving original index
        datagen = ImageDataGenerator(rescale = 1./255)
        
        #VGG16
        model = applications.VGG16(include_top=False, weight='imagenet')
        generator = datagen.flow_from_directory(
                train_data_dir,
                target_size = (img_width, img_height),
                batch_size = batch_size,
                class_mode = None,
                shuffle = False)
        
        for i in generator.filenames:
            asins.append(i[2:-5])
        
        last_layer_data = model.predict_generator(generator, nb_train_samples// batch_size)
        last_layer_data = last_layer_data.reshape((16042,25088))  #25088 nodes in last layer of VGG16
        
        np.save('numpy/16k_data_vgg16.npy', last_layer_data)
        np.save('numpy/16k_data_cnn_info.npy', np.array(asins))
        
    return last_layer_data, asins

def get_similar_img(modified_last_layer_data, original_index_list, id, numResult):
    
    '''df = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
    df_asins = list(df['asin'])
    asins = list(asins)
    id = asins.index(df_asins[id])
    '''
    pairwise_dist_img = pairwise_distances(modified_last_layer_data, modified_last_layer_data[id].reshape(1,-1))
    
    indices = np.argsort(pairwise_dist_img.flatten())[0:numResult]
    distances  = np.sort(pairwise_dist_img.flatten())[0:numResult]
    
#    real_index = np.zeros((indices.shape))
#    i = 0
#    for j in indices:
#        real_index[i] = int(original_index_list[j])
#        #print(real_index[i])
#        i += 1
#    
    return indices, distances


def similarProduct_text_brand_img(idf_vec_title, extra_features, modified_last_layer_data, original_index_list, id, numResult, w1, w2, w3):
    
    idf_w2v_dist  = pairwise_distances(idf_vec_title, idf_vec_title[id].reshape(1,-1))
    ex_feat_dist = pairwise_distances(extra_features, extra_features[id])
    img_feat_dist = pairwise_distances(modified_last_layer_data, modified_last_layer_data[id].reshape(1,-1))
    
    pairwise_dist   = (w1*idf_w2v_dist + w2*ex_feat_dist + w3*img_feat_dist)/float(w1 + w2 + w3)
    
    indices = np.argsort(pairwise_dist.flatten())[0:numResult]
    distances  = np.sort(pairwise_dist.flatten())[0:numResult]

    return indices, distances
    
if __name__ == '__main__':
    
    #Normal cleaning process
    data = load_core()
    data = removeNullPriceColor(data)
    
    #Remove duplicacy according to title
    data = remove_duplicateFromEnd(data)
    data = remove_duplicateMatchingKeyword(data)
    
    #Text cleaning
    data = text_preprocessing(data)
        
#Vectorize texts in title 
#===============================================================
#===============================================================  

    '''  
 #1. Baggage of words
    vectorized_matrix = bow_vector(data['title'])
    #indices of related product with title
    query_index = 1250
    num_of_recommendation = 10
    indices, distances = similar_title(vectorized_matrix, query_index, num_of_recommendation)
    
    #Similar data_index related to title
    real_indices_title = list(data.index[indices])
    
    getInfoByIndex(data, real_indices_title, data.index[query_index], distances)
#===============================================================
    '''
    
    '''
#===============================================================
    
    
 #2. tf-idf vectors
    vectorized_matrix = tfidf_vector(data['title'])
    #indices of related product with title
    query_index = 1250
    num_of_recommendation = 10
    indices, distances = similar_title(vectorized_matrix, query_index, num_of_recommendation)
    
    #Similar data_index related to title
    real_indices_title = list(data.index[indices])
    
    getInfoByIndex(data, real_indices_title, data.index[query_index], distances)

#===============================================================
    '''
    
    '''
#===============================================================
    
 #3. only idf -> try to use
 
    query_index = 12566
    num_of_recommendation = 5
    vectorized_matrix = idf_vector(data)
    indices, distance = similar_title(vectorized_matrix, query_index, num_of_recommendation)
    real_indices_title = list(data.index[indices])
    getInfoByIndex(data, real_indices_title, data.index[query_index], distances)

#===============================================================
    '''
    
    '''
#===============================================================

    
 #4. word2vec
    #Avg word2vec
    vectorized_matrix = word2vec(data['title'])
    #Averaging
    avg_vec_title = build_avg(data.title, vectorized_matrix, method = 'average')
    #getting similarities
    query_index = 12566
    num_of_recommendation = 20
    indices, distances = similar_title_from_w2v(avg_vec_title, query_index, num_of_recommendation)
    
    real_indices_title = list(data.index[indices])
    getInfoByIndex(data, real_indices_title, data.index[query_index], distances)
    
#===============================================================
    '''
    
    '''
#===============================================================
  
 #5. word2vec * idf_weight
    
    vectorized_matrix = word2vec(data['title'])
    
    try:
        idf_vec_title = np.load('idf_vec_title.npy')
    except:
        idf_vec_title = build_avg(data.title, vectorized_matrix, 'weighted', data)
    
    query_index = 12566
    num_of_recommendation = 20
    indices, distances = similar_title_from_w2v(avg_vec_title, query_index, num_of_recommendation)
    
    real_indices_title = list(data.index[indices])
    getInfoByIndex(data, real_indices_title, data.index[query_index], distances)

#===============================================================
     '''
     
    #Weighted word2vec stored in idf_vec_title
    vectorized_matrix = word2vec(data['title'])
    
    try:
        idf_vec_title = np.load('idf_vec_title.npy')
    except:
        idf_vec_title = build_avg(data.title, vectorized_matrix, 'weighted', data)
    
    #Extra features (Color+Brand+Type) stored in extra_features
    extra_features = addColorAndBrands(data)
    
    
    #Image based feature stored in modified_last_layer_data
    try:
        modified_last_layer_data = np.load('numpy/modified_last_layer_data.npy')
        original_index_list  = list(data.index)
    except:
        last_layer_data, asins = load_img_features()    
        modified_last_layer_data = np.zeros((16042,25088))
        original_index_list  = list(data.index)
        asins = list(asins)
        
        i = 0 
        for asin in asins:
            print('Copying index: ', data[data.asin==asin].index.values[0])
            index = data[data.asin==asin].index.values[0]
            index = original_index_list.index(index)
            modified_last_layer_data[index] = last_layer_data[i]
            i = i+1
        np.save('numpy/modified_last_layer_data.npy', modified_last_layer_data)    
      
    
    choice = input("Enter choice:\n1. Text based.\n2. Text+Barand+Type+Color.\n3. Image based\n4. Image+Text+Extra.\n")

    
    if int(choice) == 1:
        query_index = 12566
        num_of_recommendation = 5
        
        indices, distances = similar_title_from_w2v(idf_vec_title, query_index, num_of_recommendation)
        real_indices_title = list(data.index[indices])
        
        getInfoByIndex(data, real_indices_title, data.index[query_index], distances)
        

    elif int(choice) == 2:
        #Add colors and brands into feature vector
        query_index = 12566
        num_of_recommendation = 5
        w1 = 15   #Weight to title
        w2 = 35   #Weight to brand+color
        indices, distances = similarProduct(idf_vec_title, extra_features, query_index, num_of_recommendation, w1, w2)
        
        real_indices_title_brand_color = list(data.index[indices])
        getInfoByIndex(data, real_indices_title_brand_color, data.index[query_index], distances)
    
        '''
        query_index = 0
        num_of_recommendation = 5
        w1 = 15   #Weight to title
        w2 = 35   #Weight to brand+color
        indices, distances = similarProduct(idf_vec_title, extra_features, query_index, num_of_recommendation, w1, w2)
        
        real_indices_title_brand_color = list(data.index[indices])
        getInfoByIndex(data, real_indices_title_brand_color, data.index[query_index], distances)
        '''
    
    
    elif int(choice) == 3:
        
        #==========Add image similarity=============================
        #download_all_img()
        query_index = 12566
        num_of_recommendation = 5
        indices, distances = get_similar_img(modified_last_layer_data, original_index_list, query_index, num_of_recommendation)
        indices = list(data.index[indices])
        getInfoByIndex(data, indices, data.index[query_index], distances)
        
        #============================================================
    
    
    elif int(choice) == 4:
        
        '''
        #NOW WE HAVE TO CALCULATE DISTANCES BASED ON TEXT, (BRAND, COLOR, TYPE), IMAGE.
        # idf_vec_title : title similarity based on w2v+idf (weighted w2v)
        # extra_features : color+type+brand
        # modified_last_layer_data : based on cnn
        '''
        query_index = 12000
        num_of_recommendation = 5
        w1 = 25   #Weight to title
        w2 = 15   #Weight to brand+color
        w3 = 60

        indices, distances = similarProduct_text_brand_img(idf_vec_title, extra_features, modified_last_layer_data, original_index_list, query_index, num_of_recommendation, w1, w2, w3)
        indices = list(data.index[indices])
        getInfoByIndex(data, indices, data.index[query_index], distances)
    