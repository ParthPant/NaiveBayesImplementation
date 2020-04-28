#!/usr/bin/env python
# coding: utf-8

# # Movie Review Sentiment Analysis

# In[2]:


import pandas as pd
import numpy as np
import collections
#import sys
#np.set_printoptions(threshold=sys.maxsize)
import string
from tqdm import tqdm
import re


# In[3]:


data = pd.read_csv('train.tsv',delimiter='\t')
# data = data.sample(frac=1).reset_index(drop=True)


# In[4]:


# train_data = data.loc[:109242,:]
train_data = data
test_data = data.loc[109242:,:]


# In[5]:


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


# In[6]:


def extractWords(phrase):
    phrase = phrase.lower()
    words = phrase.split()
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]
    words = [w for w in words if w != "" and not hasNumbers(w)]
    return words


# In[7]:


def buildVocabulary(phrases):
    '''
    phrases - a list of phrases from which to build the vocabulary
    '''
#     print('building Vocabulary')
    
    all_words = []
    vocabulary = []#     phrases = train_data.to_numpy()[:,2]
    
    
    for phrase in tqdm(phrases):
        all_words += extractWords(phrase)

    for x in tqdm(all_words):
        if x not in vocabulary:
            vocabulary.append(x)

    vocabulary.sort()
    return vocabulary


# In[8]:


phrases = train_data.to_numpy()[:,2]

train_sentiments = train_data.to_numpy()[:,3]
test_sentiments = test_data.to_numpy()[:,3]

vocabulary = buildVocabulary(phrases)


# In[9]:


def parseData(phrases,vocabulary):
#     print('Parsing Phrases')
    
    output = np.zeros([len(phrases),len(vocabulary)],dtype='int32')
    missing_wds = 1
    iterable = tuple(enumerate(phrases))
    
    
    for i,phrase in tqdm(iterable):
        feature = np.zeros(len(vocabulary),dtype='int32')
        words = extractWords(phrase)
        for word in words:
            try:
                feature[vocabulary.index(word)] = 1.0
            except:
                missing_wds += 1
        output[i] = feature
    return output


# In[10]:


parsed_data = parseData(phrases,vocabulary)


# In[11]:


def arrangeClasswise(phrases,sentiments):
    output = {}
    for cls in np.unique(sentiments):
        output[cls]=[]
        for i,phrase in enumerate(phrases):
            if cls == sentiments[i]:
                output[cls].append(phrase)
    
    return output


# In[12]:


def trainNaiveBayes(parsed_phrases,sentiments,len_vocab):
#     print('Training')
    
    classes,counts = np.unique(sentiments,return_counts=True)
    phi_y = [cnt/len(sentiments) for cls, cnt in tuple(zip(classes,counts))]
    phi_x_y = np.zeros([len(classes),len_vocab])
    classwise_dict = arrangeClasswise(parsed_phrases,sentiments)
    
    for cls,cls_set in tqdm(classwise_dict.items()):
        phi_x_y[cls] = np.sum(cls_set,axis = 0)   #check this.............
        phi_x_y[cls] += 1
        phi_x_y[cls] /= (len(classwise_dict[cls])+len(classes))
    return phi_y, phi_x_y


# In[13]:


phi_y, phi_x_y = trainNaiveBayes(parsed_data,train_sentiments,len(vocabulary))


# In[14]:


phi_x_y


# In[33]:


# see this.........................
def predict(phrases,vocabulary):
    class_no = phi_x_y.shape[0]
    phrases = np.array(parseData(phrases,vocabulary))
    
    predictions = np.ones([len(phrases),class_no])
    
    for p,phrase in enumerate(phrases):
        for c in range(class_no):
            a = phrase
            b =1-phrase
            
            a = a * phi_x_y[c]
            #b = b * (1-phi_x_y[c])
            a[a==0] = 1
#             s = a+b
            p_x_y = np.prod(a)
            predictions[p,c] = p_x_y*phi_y[c]
        
#     predictions*phi_y
    output = []
    for prediction in predictions:
        output.append(np.argmax(prediction))
    return output


# In[35]:


predictions = predict(['boring','good movie'],vocabulary)

predictions

