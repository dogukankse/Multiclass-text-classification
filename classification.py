#!/usr/bin/env python
# coding: utf-8

# In[229]:


import glob
import nltk
import grpc
import pandas as pd

from sklearn import naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np


# In[2]:



import zemberek_grpc.language_id_pb2 as z_langid
import zemberek_grpc.language_id_pb2_grpc as z_langid_g

import zemberek_grpc.normalization_pb2 as z_normalization
import zemberek_grpc.normalization_pb2_grpc as z_normalization_g

import zemberek_grpc.preprocess_pb2 as z_preprocess
import zemberek_grpc.preprocess_pb2_grpc as z_preprocess_g

import zemberek_grpc.morphology_pb2 as z_morphology
import zemberek_grpc.morphology_pb2_grpc as z_morphology_g


# In[4]:


channel = grpc.insecure_channel('localhost:6789')


# In[5]:


langid_stub = z_langid_g.LanguageIdServiceStub(channel)
normalization_stub = z_normalization_g.NormalizationServiceStub(channel)
preprocess_stub = z_preprocess_g.PreprocessingServiceStub(channel)
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)


# In[6]:


#train data
train_ekonomi_files = glob.glob("./train/ekonomi/*.txt")
train_magazin_files = glob.glob("./train/magazin/*.txt")
train_saglik_files = glob.glob("./train/saglik/*.txt")
train_spor_files = glob.glob("./train/spor/*.txt")
#test data
test_ekonomi_files = glob.glob("./test/ekonomi/*.txt")
test_magazin_files = glob.glob("./test/magazin/*.txt")
test_saglik_files = glob.glob("./test/saglik/*.txt")
test_spor_files = glob.glob("./test/spor/*.txt")


# In[7]:


# stop words
stops = open("./stop-words-turkish.txt").readlines()
temp = []
for i in stops:
    k = str(i).replace("\r\n","").replace("\n","")
    temp.append(k)
stops = temp


# In[8]:


train_files_arr = [train_ekonomi_files, train_magazin_files, train_saglik_files, train_spor_files]
test_files_arr = [test_ekonomi_files, test_magazin_files, test_saglik_files, test_spor_files]


# In[9]:


def normalize(text):
    res = normalization_stub.Normalize(z_normalization.NormalizationRequest(input=text))
    if res.normalized_input:
        return res.normalized_input
    else:
        print('Problem normalizing input : ' + res.error)


# In[10]:


# reading train data
train_texts = []
train_labels=[]
train_file_names = []
for files in train_files_arr:
    for f in files:
        with open(f) as text:
            train_file_names.append(str(f).split('\\'))
            if "ekonomi" in str(f):
                train_labels.append("ekonomi")
            if "magazin" in str(f):
                train_labels.append("magazin")
            if "saglik" in str(f):
                train_labels.append("saglik")
            if "spor" in str(f):
                train_labels.append("spor")
            t = normalize(text.read())
            train_texts.append(t)


# In[11]:


# reading test data
test_texts, test_labels, test_file_names=[],[],[]
for files in test_files_arr:
    for f in files:
        with open(f) as text:
            test_file_names.append(str(f).split("\\"))
            if "ekonomi" in str(f):
                test_labels.append("ekonomi")
            if "magazin" in str(f):
                test_labels.append("magazin")
            if "saglik" in str(f):
                test_labels.append("saglik")
            if "spor" in str(f):
                test_labels.append("spor")
            t = normalize(text.read())
            test_texts.append(t)


# In[256]:


trainDF = pd.DataFrame()
trainDF["label"] = train_labels
trainDF["file"] = [line[1] for line in train_file_names]
trainDF["text"] = train_texts


# In[255]:


testDF = pd.DataFrame()
testDF["label"] = test_labels
testDF["file"] = [line[1] for line in test_file_names]
testDF["text"] = test_texts


# In[14]:


def tokenize(text_arr,dataframe):
    token_str = ""
    tokens = []
    for text in text_arr:
        res = preprocess_stub.Tokenize(z_preprocess.TokenizationRequest(input=text))
        for i in res.tokens:
            if i.type != "Punctuation" and  i.type != "Number" and i.type != "Date":
                token_str += i.token+" "
        tokens.append(token_str)
        token_str = ""
    dataframe["tokenized"] = tokens


# In[15]:


def stem(dataframe):
    stemmed = []
    stem_str = ""
    for text in dataframe["tokenized"]:
        for token in text.split(" "):
            if token is "":
                continue
            res = morphology_stub.AnalyzeSentence(z_morphology.SentenceAnalysisRequest(input=str(token)))
            stem_str += res.results[0].best.dictionaryItem.lemma.lower()+ " "
        stemmed.append(stem_str)
        stem_str = ""
    dataframe["stemmed"] = stemmed


# In[16]:


tokenize(test_texts,testDF)
tokenize(train_texts,trainDF)


# In[17]:


stem(testDF)
stem(trainDF)


# In[18]:


def remove_stops(dataframe):
    no_stops = []
    no_stop_str=""
    for text in dataframe["stemmed"]:
        for word in text.split(" "):
            if word not in stops:
                no_stop_str+=word+" "
        no_stops.append(no_stop_str)
        no_stop_str = ""
    dataframe["no_stop"] = no_stops


# In[19]:


remove_stops(testDF)
remove_stops(trainDF)


# In[ ]:





# In[199]:


# ngram level tf-idf 
#tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram = TfidfVectorizer(ngram_range=(1,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['no_stop'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(trainDF["no_stop"])
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(testDF["no_stop"])


# In[200]:


#ngrams
#tfidf_vect_ngram.get_feature_names()


# In[224]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier    
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    return metrics.accuracy_score(predictions, testDF["label"]),predictions


# In[225]:


accuracy, predictions = train_model(naive_bayes.MultinomialNB(),xtrain_tfidf_ngram,trainDF["label"],xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy)


# In[257]:



a = pd.DataFrame(xtrain_tfidf_ngram.toarray(),index=[line[1] for line in train_file_names],columns=tfidf_vect_ngram.get_feature_names())
b = pd.DataFrame(xvalid_tfidf_ngram.toarray(),index=[line[1].replace(" ","_") for line in test_file_names],columns=tfidf_vect_ngram.get_feature_names())
a["Sınıf"] = train_labels
b["Sınıf"] = test_labels
a=a.append(b) 


# In[258]:


a.to_csv("./tf-idf.csv")


# In[259]:


cm = confusion_matrix(test_labels, predictions)
recall = np.diag(cm) / np.sum(cm, axis = 1)
recall = np.append(recall,np.mean(recall))
precision = np.diag(cm) / np.sum(cm, axis = 0)
precision=np.append(precision,np.mean(precision))
f_score = 2*(precision*recall)/(precision+recall)


# In[260]:


pc=pd.DataFrame(index=["Precision","Recall","F-Score"],columns=["ekonomi","magazin","sağlık","spor","ortalama"])
pc.iloc[0],pc.iloc[1],pc.iloc[2] = precision,recall,f_score
pc.to_csv("./performans_olcum.csv")


# In[261]:


print(pc)


# In[ ]:




