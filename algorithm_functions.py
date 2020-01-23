import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
import re

import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import statistics


stop_words=set(stopwords.words("english"))
 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words=set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer() 


def cleaning_bow_for_training(data):
    
    new_data=data.dropna(how='all', axis=1)
    bow=data.iloc[:,2:].values.tolist()
    
    cleaned_data = [
    [w for w in l if isinstance(w, str)] 
    for l in bow
    ]
    return cleaned_data

def running_model(training_bow):
    model = gensim.models.Word2Vec(training_bow, min_count=1, sg=1)
    model.train(training_bow, total_examples=model.corpus_count, epochs=model.epochs)
    model.wv
    return model


def tags_from_lists(bow, model):
    tags_lists=[]
    for lst in bow:
        while True:
            if len(lst) == 0:
                break
            try:
                tags=model.wv.most_similar(positive=lst, topn=20)
                tags_lists.append(tags)
            except KeyError as e:
                search = re.search("'(\w+)'", e.args[0])
                if search:
                    word = search.group(1)
                else:
                    word = ''
                print(word)
                lst.remove(word)
            else:
                break
    return tags_lists


def transforming_in_bow(text, lemmatizer):
    
    sentence = text.translate(str.maketrans('', '',string.punctuation)).split(' ')
    bow = []
    for word in sentence:
        lowcase_text_word=word.lower()
        lemmatized_word=lemmatizer.lemmatize(lowcase_text_word)
        if lemmatized_word not in stop_words:
            bow.append(lemmatized_word)
    return bow

def tags_from_text(bow, model):
    while True:
        if len(bow) == 0:
            return []
        try:
            tags=model.wv.most_similar(positive=bow, topn=20)
        except KeyError as e:
            search = re.search("'(\w+)'", e.args[0])
            if search:
                word = search.group(1)
            else:
                word = ''        
            print(word)
            bow.remove(word)
        else:
            return tags
   






def relevance(tags_text, tags_groups, model):
    distance_lists_max=[]
    for group in tags_groups:
        group_id=[]
        for tag_g in group:
            for tag in tags_text:
                distance=model.wv.distance(tag[0], tag_g[0])
                proportion=1-distance
                relevance=proportion*tag[1]
                group_id.append(relevance)
        distance_lists_max.append(max(group_id))
    return distance_lists_max



def relevance_mean(tags_text, tags_groups, model):
    distance_lists_max=[]
    for group in tags_groups:
        group_id=[]
        for tag_g in group:
            for tag in tags_text:
                distance=model.wv.distance(tag[0], tag_g[0])
                proportion=1-distance
                relevance=proportion*tag[1]
                group_id.append(relevance)
        distance_lists_max.append(statistics.mean(group_id))
    return distance_lists_max

def relevance_count(tags_text, tags_groups, model):
    distance_lists_max=[]
    for group in tags_groups:
        group_id=[]
        for tag_g in group:
            for tag in tags_text:
                distance=model.wv.distance(tag[0], tag_g[0])
                proportion=1-distance
                relevance=proportion*tag[1]
                if relevance>=0.7:
                    group_id.append(relevance)
        distance_lists_max.append(len(group_id))
    return distance_lists_max

def relevance_words(bow, bow_gr, model):
    distance_lists_max=[]
    for group in bow_gr:
        group_id=[]
        for tag_g in group:
            for tag in bow:
                try:
                    distance=model.wv.distance(tag, tag_g)
                except:
                    if tag in bow:
                        bow.remove(tag)
                proportion=1-distance
                group_id.append(proportion)
        distance_lists_max.append(statistics.mean(group_id))
    return distance_lists_max










def get_the_recommended_titles(relevance, titles_data):
    
    distance_list_titles=pd.DataFrame(relevance, columns=['member'])
    titles_values=pd.merge(left=titles_data[['group_name']], right=distance_list_titles, left_index=True, right_index=True)
    high_values=titles_values.loc[titles_values['member']>=0.10]
    recommended_gr=high_values.loc[high_values['member']>=(high_values['member'].max()-0.03)]
    sorted_rec_gr = recommended_gr.sort_values('member', ascending=False)
    
    return  sorted_rec_gr



def get_the_recommended_titles_count(relevance, titles_data):
    
    distance_list_titles=pd.DataFrame(relevance, columns=['member'])
    titles_values=pd.merge(left=titles_data[['group_name']], right=distance_list_titles, left_index=True, right_index=True)
    high_values=titles_values.loc[titles_values['member']>=1]
    recommended_gr=high_values.loc[high_values['member']>=(high_values['member'].max()-10)]
    sorted_rec_gr = recommended_gr.sort_values('member', ascending=False)
    
    return  sorted_rec_gr




def input_text_recommendation(text, model, tags_groups, group_titles):
    bow = transforming_in_bow(text,lemmatizer)
    tags_text=tags_from_text(bow, model)
    if len(tags_text) == 0:
        return "No recommendations."
    tags_relevance=relevance(tags_text, tags_groups, model)
    recommended=get_the_recommended_titles(tags_relevance, group_titles).head(10)
    return recommended[['group_name']]


def input_text_recommendation_mean(text, model, tags_groups, group_titles):
    bow = transforming_in_bow(text,lemmatizer)
    tags_text=tags_from_text(bow, model)
    if len(tags_text) == 0:
        return "No recommendations."
    tags_relevance=relevance_mean(tags_text, tags_groups, model)
    recommended=get_the_recommended_titles(tags_relevance, group_titles).head(10)
    return recommended[['group_name']]


def input_text_recommendation_count(text, model, tags_groups, group_titles):
    bow = transforming_in_bow(text,lemmatizer)
    tags_text=tags_from_text(bow, model)
    if len(tags_text) == 0:
        return "No recommendations."
    tags_relevance=relevance_count(tags_text, tags_groups, model)
    recommended=get_the_recommended_titles_count(tags_relevance, group_titles).head(10)
    return recommended[['group_name']]

def input_text_recommendation_words(text, model, groups_bow, group_titles):
    bow = transforming_in_bow(text,lemmatizer)
#     tags_text=tags_from_text(bow, model)
#     if len(tags_text) == 0:
#         return "No recommendations."
    tags_relevance=relevance_words(bow, groups_bow, model)
    recommended=get_the_recommended_titles(tags_relevance, group_titles).head(10)
    return recommended[['group_name']]




def transform_concat_lists(df):
    lst=df.values.tolist()
    big_list=[]
    for small_lst in lst:
        big_list+=small_lst
    return big_list

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    print('TP=',TP,'FP=',FP,'TN=',TN,'FN=',FN)
    return TP, FP, TN, FN
                                                               
                                                               
                                                               
                                                               