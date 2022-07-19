import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import sklearn
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB

from tensorflow.keras import models
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
#from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.layers import LSTM, Input, Activation, Dropout
from tensorflow.keras.layers import Conv1D, BatchNormalization, Flatten
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing import sequence

#citesc date
train_data=[]

with open("train_samples.txt",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
        train_data.append(row[1])

train_labels=[]
with open("train_labels.txt",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
        train_labels.append(int(row[1]))

#train_data=np.array(train_data)
train_labels=np.array(train_labels)


validation_data=[]
validation_idx=[]
with open("validation_samples.txt",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
    	validation_idx.append(row[0])
    	validation_data.append(row[1])

validation_labels=[]
with open("validation_labels.txt",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
        validation_labels.append(int(row[1]))

#validation_data=np.array(validation_data)
validation_labels=np.array(validation_labels)

test_data=[]
test_idx=[]
with open("test_samples.txt",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter="\t")
    for row in rd:
    	test_idx.append(row[0])
    	test_data.append(row[1])
print(len(train_data))
print(len(test_data))
print(len(validation_data))
#combinat 3-5 algoritmi si gasit weights optim + confusion matrix
"""
first=[]

with open("sAnn.csv",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter=",")
    for row in rd:
        first.append(int(row[1]))
second=[]
with open("submission (1).csv",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter=",")
    for row in rd:
        second.append(int(row[1]))
third=[]
with open("finalSvc.csv",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter=",")
    for row in rd:
        third.append(int(row[1]))

fourth=[]
with open("sXgb.csv",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter=",")
    for row in rd:
        fourth.append(int(row[1]))

fifth=[]
with open("finalRandom.csv",encoding="mbcs") as fd:
    rd = csv.reader(fd, delimiter=",")
    for row in rd:
        fifth.append(int(row[1]))
alph=sklearn.metrics.f1_score(third,second,average='macro')
print(alph)
from sklearn.metrics import confusion_matrix
results = confusion_matrix(validation_labels, fourth)
print(results)
exit()
from sklearn.utils.extmath import weighted_mode
w=[1,3,1]
#print(int(weighted_mode([3,2,3,2],w)[0]))
#exit()

finak=[]
for i in range(len(first)):
    #finak.append(int(median([first[i],second[i],third[i]])))
    finak.append(int(weighted_mode([first[i],second[i],third[i]],w)[0]))
alph=sklearn.metrics.f1_score(validation_labels,finak,average='macro')
print(alph)"""
"""
print(len(finak))
#print(finak)
qq=zip(test_idx,finak)
with open("submissionTest.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()

f=open("tt1t2.csv","w")
maxq=-9999
from numpy import median

for l in range(1,4):
    for j in range(1,4):
        for k in range(1,4):
            finak=[]
            #freq=[0,0,0,0]
            #import random
            #q=0
            #fq=[0,0,0,0]
            w=[l,j,k]
            #print(int(weighted_mode([3,2,3,2],w)[0]))
            #exit()
            for i in range(len(first)):
                #finak.append(int(median([first[i],second[i],third[i]])))
                finak.append(int(weighted_mode([first[i],second[i],third[i]],w)[0]))
            #print(len(finak))

            alph=sklearn.metrics.f1_score(validation_labels,finak,average='macro')
            #print(alph, l ,j,k, sep=" ")
            #exit()
            if alph>maxq:
                maxq=alph
            f.write(f"{alph}, {l}, {j}, {k}\n")
f.write(f"{maxq}")
exit()
"""
#for i in finak:
    #freq[i]+=1
#print(freq,fq)
"""
qq=zip(test_idx,finak)
with open("submissionTest.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()"""
from unidecode import unidecode
#77.668
# un pic din 1 2 3 
from nltk.stem import WordNetLemmatizer
import nltk

#pregatit array cu substringuri posibile de eliminat
"""h=[]
lemmatizer = WordNetLemmatizer()
#().,;""-_#*@/FKLMNSTUVWXOABCDEabcdefghijklmnopqrstuvwxyzPGH0123456789IJQRYZ!~=:}{[]<>?|%& 
mor='().,;""-_#*@/FKLMNSTUVWXOABCDEabcdefghijklmnopqrstuvwxyzPGH0123456789IJQRYZ!~=:}{[]<>?|%& '
miz=[]
for i in range(21,len(mor)):
    break
    for j in range(len(mor)):
        for k in range(len(mor)):
            temp=mor[i]+mor[j]+mor[k]
            for l in range(len(train_data)):
                if (train_data[l].replace(temp,"") ) != train_data[l]:
                    miz.append(temp)
                    break
    if i==30:
        break"""
#print(miz[0],miz[1])
#exit()
strs=['; ', '.', ')f', '~', 'm)', '0391 ', 'ZHD', 'TRD', 'KKEHTY', 'p ** ', 'KZk', '-q', '(Ap', '(A ', '(g ', '( Y', ')KX', ')K0', ')gK', ')qH', ')21', ')Z*', ')Zq', '(& ', '(K', '(x', 'F!z', 'F! ', 'K;O', 'K-*', 'K*R', 'KFW', 'KLP', 'KX@', 'Kfo', 'K07', 'K4 ', 'K57', 'K5 ', 'K71', 'K :', 'K &', 'F 2', 'L)A', 'L F', 'L n', 'L q', 'L u', 'L 0', 'L &']
#K;O K-* F 2
#strs=['; ', '.', ')f', '~', 'm)', '0391 ', 'ZHD', 'TRD', 'KKEHTY', 'p ** ', 'KZk', '-q', '(Ap', '(A ', '(g ', '( Y', ')KX', ')K0', ')gK', ')qH', ')21', ')Z*', ')Zq', '(& ', '(K', '(x', 'F!z', 'F! ']
#strs=['; ', '.', ')f', '~', 'm)', '0391 ', 'ZHD', 'TRD', 'KKEHTY', 'p ** ', 'KZk', '-q', '(Ap', '(A ', '(g ', '( Y', ')KX', ')K0', ')gK', ')qH', ')21', ')Z*', ')Zq', '(& ', '(K', '(x', 'F!z', 'F! ', 'K."', 'K;O', 'K-*', 'K*R', 'KFW', 'KLP', 'KX@', 'Kfo']
#strs=['; ', '.', ')f', '~', 'm)', '0391 ', 'ZHD', 'TRD', 'KKEHTY', 'p ** ', 'KZk', '-q', '(Ap', '(A ', '(g ', '( Y', ')KX', ')K0', ')gK', ')qH', ')21', ')Z*', ')Zq', '(& ', '(K', '(x', 'F!z', 'F! ', 'K;O', 'K-*', 'K*R', 'KFW', 'KLP', 'KX@', 'Kfo', 'K07', 'K4 ', 'K57', 'K5 ', 'K71', 'K :', 'K &', 'F 2', 'L)A', 'L F', 'L n', 'L q', 'L u', 'L 0', 'L &', 'N.%', 'NgU', 'NYK', 'N ,', 'T-Y', 'TXc', 'TXq', 'TE*', 'TH-', 'T! ', 'T z', 'Uh ', 'Umx', 'U e']#['; ' , '.' , ')f', '~'  , 'm)' , '0391 ', 'ZHD' , 'TRD', 'KKEHTY', 'p ** ', 'KZk', '-q']
#strs=""
#preprocesare
for i in range(len(train_data)):
    train_data[i]=unidecode(train_data[i])
    for j in strs:
        train_data[i]=train_data[i].replace(j,"")
    #train_data[i]=' '.join([w for w in train_data[i].split() if ((len(w)>0 and len(w)<6) or (len(w)>6))])
    #train_data[i]=train_data[i].replace(";"," ")
    #train_data[i]=lemmatizer.lemmatize(train_data[i])  
for i in range(len(test_data)):
    test_data[i]=unidecode(test_data[i])

    for j in strs:
        test_data[i]=test_data[i].replace(j,"")
for i in range(len(validation_data)):
    validation_data[i]=unidecode(validation_data[i])

    for j in strs:
        validation_data[i]=validation_data[i].replace(j,"")

    #validation_data[i]=' '.join([w for w in validation_data[i].split() if ((len(w)>0 and len(w)<6) or (len(w)>6))])
    #validation_data[i]=lemmatizer.lemmatize(validation_data[i])    
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#vect = CountVectorizer(tokenizer=LemmaTokenizer())    
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,4),binary=True,max_df=0.3,max_features=79000)
"""train_data = vectorizer.fit_transform(train_data)
validation_data = vectorizer.transform(validation_data)
c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
alph=sklearn.metrics.f1_score(validation_labels,preds,average='macro')
print(alph)
exit()"""
#countvec
vec2 = CountVectorizer(analyzer='char', ngram_range=(7,7),binary=True,max_df=0.3,max_features=500000)
vec3 = CountVectorizer(binary=True,token_pattern = r"[\w\.]{1,2}")
from sklearn.pipeline import FeatureUnion
union=FeatureUnion([("s",vectorizer),("vecc",vec2)] )
train_data = union.fit_transform(train_data).toarray()
validation_data = union.transform(validation_data).toarray()
test_data=union.transform(test_data)
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
"""
c = MultinomialNB(alpha=0.1)
c.fit(train_data,train_labels)

preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)"""

for i in range(len(validation_labels)):
    validation_labels[i]-=1
for i in range(len(train_labels)):
    train_labels[i]-=1
import xgboost as xgb
"""
#c = xgb.XGBClassifier(objective ='multi:softprob',n_estimators=50,tree_method='hist')
c=LogisticRegression(max_iter=5000)
c.fit(train_data,train_labels)

preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
label_test_pred=c.predict(validation_data)
temp=c.predict_proba(test_data)
for i in temp:
    print(i)
print("xx",label_test_pred)
qq=zip(validation_idx,label_test_pred)
with open("sLogFinal.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')

exit()"""

inp=579000
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=Sequential()
from tensorflow.keras import layers 

#model.add( Embedding(200,50,input_length=inp) )
#model.add( LSTM(10))
#model.add(Conv1D(128,5,activation='relu'))
#model.add(layers.GlobalMaxPooling1D())
#model.add(Dense(170, input_dim=inp, activation='relu'))
#model.add(Dense(160, input_dim=inp, activation='relu'))
#model.add(Dense(150, input_dim=inp, activation='relu'))
#model.add(Dense(150, input_dim=inp, activation='relu'))
#model.add(Dense(100, input_dim=inp, activation='relu'))
#model.add(Dense(80, input_dim=inp, activation='relu'))
#model.add(Dense(10, input_dim=inp, activation='relu'))
#model.add(Dropout(0.4))
#model.add(BatchNormalization())
#model.add(Dense(40, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(20, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(50, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(25, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(125, input_dim=inp, activation='relu'))
model.add(Dense(3,input_dim=inp, activation='softmax'))


#model.add(Dense(10, input_dim=inp, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
#model.add(Flatten())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
def lr_scheduler(epoch, lr):
    if epoch < 50:
        lr = 0.05 #0.00005
        return lr
    return lr
model_checkpoint = ModelCheckpoint('best.hdf5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
callback=[model_checkpoint,tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

#model.fit(train_data, to_categorical(train_labels), epochs=1, batch_size=32,validation_data=(validation_data,to_categorical(validation_labels)),verbose=1)
model.load_weights('best.hdf5')
#print(model.evaluate(validation_data,to_categorical(validation_labels)))
preds=model.predict(test_data)

out=zip(test_idx,preds)
with open("hopiumAnn.csv", "w") as g:
    g.write("id,label\n")
    for predict in out:
        g.write(str(predict[0]) + ',' + str(predict[1].argmax()+1) + '\n')
exit()

c = MultinomialNB(alpha=0.1)
c.fit(train_data,train_labels)

preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
#c = MultinomialNB(alpha=0.1)\

preds=c.predict(validation_data)
out=zip(validation_labels,preds)
with open("newNB.csv", "w") as g:
    g.write("id,label\n")
    for predict in out:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()

import xgboost as xgb
c = xgb.XGBClassifier(objective ='multi:softprob',n_estimators=500,tree_method='hist')
c.fit(train_data,train_labels)

preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
label_test_pred=c.predict(validation_data)
"""temp=c.predict_proba(test_data)
for i in temp:
    print(i)"""
print("xx",label_test_pred)
qq=zip(test_idx,label_test_pred)
with open("sXgb.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]+1) + '\n')

exit()
c = LogisticRegression(max_iter=50000)
#c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
exit()
label_test_pred=c.predict(test_data)
print("xx",label_test_pred)
qq=zip(test_idx,label_test_pred)
with open("submission5.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]+1) + '\n')
exit()
for i in range(len(validation_labels)):
    validation_labels[i]-=1
for i in range(len(train_labels)):
    train_labels[i]-=1

import xgboost as xgb
c = xgb.XGBClassifier(objective ='multi:softprob',n_estimators=500,tree_method='hist')
#c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
label_test_pred=c.predict(test_data)
print("xx",label_test_pred)
qq=zip(test_idx,label_test_pred)
with open("submission1.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]+1) + '\n')
exit()

inp=579000
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=Sequential()
from tensorflow.keras import layers 

#model.add( Embedding(200,50,input_length=inp) )
#model.add( LSTM(10))
#model.add(Conv1D(128,5,activation='relu'))
#model.add(layers.GlobalMaxPooling1D())
#model.add(Dense(170, input_dim=inp, activation='relu'))
#model.add(Dense(160, input_dim=inp, activation='relu'))
#model.add(Dense(150, input_dim=inp, activation='relu'))
#model.add(Dense(150, input_dim=inp, activation='relu'))
#model.add(Dense(100, input_dim=inp, activation='relu'))
#model.add(Dense(80, input_dim=inp, activation='relu'))
#model.add(Dense(10, input_dim=inp, activation='relu'))
#model.add(Dropout(0.4))
#model.add(BatchNormalization())
#model.add(Dense(40, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(20, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(50, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.95))
model.add(Dense(25, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(125, input_dim=inp, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(3,input_dim=inp, activation='softmax'))


#model.add(Dense(10, input_dim=inp, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
#model.add(Flatten())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
def lr_scheduler(epoch, lr):
    if epoch < 50:
        lr = 0.00005
        return lr
    return lr
model_checkpoint = ModelCheckpoint('best.hdf5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
callback=[model_checkpoint,tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

model.fit(train_data, to_categorical(train_labels), epochs=1, batch_size=32,validation_data=(validation_data,to_categorical(validation_labels)),callbacks=callback,verbose=1)
model.load_weights('best.hdf5')
print(model.evaluate(validation_data,to_categorical(validation_labels)))
preds=model.predict(test_data)
out=zip(test_idx,preds)
with open("submission1.csv", "w") as g:
    g.write("id,label\n")
    for predict in out:
        g.write(str(predict[0]) + ',' + str(predict[1].argmax()+1) + '\n')
exit()
"""for i in range(len(train_labels)):
    train_labels[i]-=1
for i in range(len(validation_labels)):
    validation_labels[i]-=1"""
"""
import xgboost as xgb
c=xgb.XGBClassifier(objective='multi:softprob',n_estimators=240)
#c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
#print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
label_test_pred=c.predict(test_data)
print("xx",label_test_pred)
qq=zip(test_idx,label_test_pred)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]+1) + '\n')
exit()"""
c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
print(preds)
alph=sklearn.metrics.f1_score(preds,validation_labels,average='macro')
print(alph)
label_test_pred=c.predict(test_data)
print("xx",label_test_pred)
qq=zip(test_idx,label_test_pred)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()
exit()
min11=0.7884762330565569
g=open("submission.csv","w")

#gasit cuvinte de eliminat
for q in miz:
    strs.append(q)
    train_data=[]
    with open("train_samples.txt",encoding="mbcs") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            train_data.append(row[1])

    train_labels=[]
    with open("train_labels.txt",encoding="mbcs") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            train_labels.append(int(row[1]))

    #train_data=np.array(train_data)
    train_labels=np.array(train_labels)


    validation_data=[]
    validation_idx=[]
    with open("validation_samples.txt",encoding="mbcs") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            validation_idx.append(row[0])
            validation_data.append(row[1])

    validation_labels=[]
    with open("validation_labels.txt",encoding="mbcs") as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            validation_labels.append(int(row[1]))

    #validation_data=np.array(validation_data)
    validation_labels=np.array(validation_labels)
    for i in range(len(train_data)):
        train_data[i]=unidecode(train_data[i])
        for j in strs:
            train_data[i]=train_data[i].replace(j,"")
    	#train_data[i]=train_data[i].replace(";"," ")
    	#train_data[i]=lemmatizer.lemmatize(train_data[i])	
    for i in range(len(validation_data)):
        validation_data[i]=unidecode(validation_data[i])

        for j in strs:
            validation_data[i]=validation_data[i].replace(j,"")
    	#validation_data[i]=lemmatizer.lemmatize(validation_data[i])	
    vectorizer = CountVectorizer(min_df=1,analyzer='char', ngram_range=(3,7),binary=True)
    train_data = vectorizer.fit_transform(train_data)
    validation_data = vectorizer.transform(validation_data)
    c=MultinomialNB(alpha=0.1)
    c.fit(train_data, train_labels)
    preds=c.predict(validation_data)
    alph=sklearn.metrics.f1_score(validation_labels,preds,average='macro')
    exit()
    vec2 = CountVectorizer(min_df=1,analyzer='char', ngram_range=(7,7),binary=True)
    vec3 = CountVectorizer(min_df=1,binary=True,token_pattern = r"[\w\.]{1,2}")
    from sklearn.pipeline import FeatureUnion
    union=FeatureUnion([("vec",vectorizer),("vecc",vec2),("veccc",vec3) ] )
    train_data = union.fit_transform(train_data)
    validation_data = union.transform(validation_data)
    c=MultinomialNB(alpha=0.1)
    c.fit(train_data, train_labels)
    preds=c.predict(validation_data)
    alph=sklearn.metrics.f1_score(validation_labels,preds,average='macro')

    g.write(f"{alph}, {strs[-1]}\n")
    #print(alph, strs[-1], sep=" ")
    if alph>min11:
        min11=alph

        g.write(f"{alph}, {strs[-1]}, new {strs}\n")
        #print(alph, strs[-1],"Gasit new",strs, sep=" ")

        continue
    strs.pop()
#(Ap (A" " (g" " ( Y )KX )K0
print(strs)
print(h)
exit()
print(train_data[2],"\n")

vectorizer = CountVectorizer(min_df=1,analyzer='char', ngram_range=(3,4),binary=True)

#4 is important
#5 no
vec2 = CountVectorizer(min_df=1,analyzer='char', ngram_range=(7,7),binary=True)
#0.77376
#cu toate e 77379
vec3 = CountVectorizer(min_df=1,binary=True,token_pattern = r"[\w\.]{1,2}")

#vec4 = CountVectorizer(min_df=1,binary=True,token_pattern = r"[a-zA-Z]{1,2}")

from sklearn.metrics.pairwise import cosine_similarity
from nltk.collocations import *
from nltk.probability import FreqDist
import nltk
import re
from itertools import tee, islice

#vec4 = CountVectorizer(min_df=1,binary=True, ngram_range=(1,3))
#1 2
#vec3 = CountVectorizer(min_df=1,analyzer='char', ngram_range=(1,1),binary=True)
from sklearn.pipeline import FeatureUnion
union=FeatureUnion([("vec",vectorizer),("vecc",vec2),("veccc",vec3) ] )
#union=FeatureUnion([("vec",vectorizer),("vecc",vec2)] )
#vectorizer.fit(train_data)
from sklearn.model_selection import train_test_split

train_data = union.fit_transform(train_data)
validation_data = union.transform(validation_data)
test_data = union.transform(test_data)
c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
#print(sklearn.metrics.accuracy_score(y_test, preds))
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
label_test_pred=c.predict(test_data)
qq=zip(test_idx,label_test_pred)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()


#incearca 37 cu eliminari
te=0
te2=0
tot=0
tot2=0
#testat pt overfit
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(validation_data, validation_labels, test_size=0.06)
#testat cu alpha=1
#dai ulpoad la all cu alpha=1 to do
    c=MultinomialNB(alpha=0.1)
    c.fit(train_data, train_labels)
    preds1=c.predict(X_train)
    preds2=c.predict(X_test)
    #print(sklearn.metrics.accuracy_score(y_test, preds))
    a=sklearn.metrics.f1_score(y_train,preds1,average='macro')
    b=sklearn.metrics.f1_score(y_test,preds2,average='macro')
    print(b,a,sep=" ") 
    tot+=b
    tot2+=a
    """print(sklearn.metrics.f1_score(y_test,preds,average='macro'))
                te+=sklearn.metrics.f1_score(y_test,preds,average='macro')
                if sklearn.metrics.f1_score(y_test,preds,average='macro') < min1:
                    min1=sklearn.metrics.f1_score(y_test,preds,average='macro')"""
print("avg:")
print(tot/1000,tot2/1000,sep=" ")
#0.7722682471697401 0.7270422988013915 0.8148203150098438 toate datele

#0.7801893859346299 0.7415127384343762 doar train 0.2
#0.7802344494956608 0.771639397524449 doar train 0.8

#0.779880137753134 0.7802677963246186 ambele 0.2 0.8
exit()
"""
import xgboost as xgb
xg_reg = xgb.XGBClassifier(objective ='multi:softprob')
xg_reg.fit(train_data,train_labels)
preds = xg_reg.predict(validation_data)
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
exit()"""
#print(train_data)
'''
c=svm.SVC(C=100)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
print(sklearn.metrics.accuracy_score(validation_labels, preds))
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
'''
'''
c = sklearn.ensemble.RandomForestClassifier(n_estimators=500, n_jobs=100)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
print(sklearn.metrics.accuracy_score(validation_labels, preds))
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
'''
'''
c = sklearn.linear_model.LogisticRegression(random_state=0, multi_class='multinomial',max_iter=5000)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
print(sklearn.metrics.accuracy_score(validation_labels, preds))
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
'''
#baga cu 37all si 37train 0.1
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB, CategoricalNB
#c=RandomForestClassifier(n_estimators=500,n_jobs=3)
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.ensemble import AdaBoostClassifier
c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
#print(sklearn.metrics.accuracy_score(y_test, preds))
#print(sklearn.metrics.f1_score(y_test,preds,average='macro'),i,sep=" ")
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
label_test_pred=c.predict(test_data)
qq=zip(test_idx,label_test_pred)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()

#alte incercari

"""exit()
te=0
minim1000=999
from sklearn.model_selection import train_test_split

for i in range(1000):
	X_train, X_test, y_train, y_test = train_test_split(validation_data, validation_labels, test_size=0.2)
#testat cu alpha=1
#dai ulpoad la all cu alpha=1 to do
	c=MultinomialNB(alpha=0.1)
	c.fit(train_data, train_labels)
	preds=c.predict(X_test)
    #print(sklearn.metrics.accuracy_score(y_test, preds))
	#print(sklearn.metrics.f1_score(y_test,preds,average='macro'),i,sep=" ")
	te+=sklearn.metrics.f1_score(y_test,preds,average='macro')
	if sklearn.metrics.f1_score(y_test,preds,average='macro') < minim1000:
		minim1000=sklearn.metrics.f1_score(y_test,preds,average='macro')
print("avg:")
#0.7689182118913775 0.7205440498810258 0.2 1000
#0.7692939516555745 0.715451209642314 0.2 5000
print(te/1000, minim1000, sep=" ")
te=0
min5000=999
for i in range(5000):
	X_train, X_test, y_train, y_test = train_test_split(validation_data, validation_labels, test_size=0.2)
#testat cu alpha=1
#dai ulpoad la all cu alpha=1 to do
	c=MultinomialNB(alpha=0.1)
	c.fit(train_data, train_labels)
	preds=c.predict(X_test)
    #print(sklearn.metrics.accuracy_score(y_test, preds))
	#print(sklearn.metrics.f1_score(y_test,preds,average='macro'),i,sep=" ")
	te+=sklearn.metrics.f1_score(y_test,preds,average='macro')
	if sklearn.metrics.f1_score(y_test,preds,average='macro') < min5000:
		min5000=sklearn.metrics.f1_score(y_test,preds,average='macro')
print("avg:")
print(te/5000, min5000, sep = " ")
exit()
#avg 0.76887 - > 1000 validation 0.2 size
#avg 0.7692
#try random forest si xgboost
c=MultinomialNB(alpha=0.1)
c.fit(train_data, train_labels)
#print(c.class_log_prior_)
preds=c.predict(validation_data)
'''
qq=zip(validation_idx,preds)
with open("submission.txt", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()
'''
print(sklearn.metrics.accuracy_score(validation_labels, preds))
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
#exit()
'''
from sklearn.neighbors import KNeighborsClassifier
c = KNeighborsClassifier(n_neighbors=7)
c.fit(train_data, train_labels)
preds=c.predict(validation_data)
print(sklearn.metrics.accuracy_score(validation_labels, preds))
print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
'''

label_test_pred=c.predict(test_data)
qq=zip(test_idx,label_test_pred)
with open("submission.csv", "w") as g:
    g.write("id,label\n")
    for predict in qq:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
exit()"""
embedding_vecor_length = 32
'''
inputs = Input(name='inputs',shape=[471])
layer = Embedding(1000,50,input_length=471)(inputs)
layer = LSTM(64)(layer)
layer = Dropout(0.5)(layer)
layer = Dense(14,activation='relu')(layer)
layer = Dense(12,activation='relu')(layer)
layer = Dense(10,activation='relu')(layer)
layer = Dense(5,activation='relu')(layer)
layer = Dense(1,activation='sigmoid')(layer)
#layer = Activation('sigmoid')(layer)
model = Model(inputs=inputs,outputs=layer)
'''
from tensorflow.keras.preprocessing.text import Tokenizer
"""
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data)
train_data = tokenizer.texts_to_sequences(train_data)

validation_data = tokenizer.texts_to_sequences(validation_data)
test_data = tokenizer.texts_to_sequences(test_data)
vocab_size = len(tokenizer.word_index) + 1  
maxlen = 500
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data=train_data.todense()
validation_data=validation_data.todense()
train_data = pad_sequences(train_data, padding='post', maxlen=maxlen)
validation_data = pad_sequences(validation_data, padding='post', maxlen=maxlen)
test_data = pad_sequences(test_data, padding='post', maxlen=maxlen)"""
train_data=train_data.todense()
validation_data=validation_data.todense()

for i in range(len(validation_labels)):
	validation_labels[i]-=1
for i in range(len(train_labels)):
    train_labels[i]-=1
#exit()
#Xcnn_test = tokenizer.texts_to_sequences(review_test)
#inp=train_data.shape[1]
#vocab_size=1000
#maxlen=7129
inp=6983

model=Sequential()
model.add( Embedding(1000,50,input_length=inp) )
model.add( LSTM(64))
model.add(Dense(170, input_dim=inp, activation='relu'))
model.add(Dense(160, input_dim=inp, activation='relu'))
model.add(Dense(150, input_dim=inp, activation='relu'))
model.add(Dense(150, input_dim=inp, activation='relu'))
model.add(Dense(100, input_dim=inp, activation='relu'))
model.add(Dense(80, input_dim=inp, activation='relu'))
model.add(Dense(60, input_dim=inp, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(40, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(20, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(10, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(5, input_dim=inp, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(3,input_dim=inp, activation='softmax'))


#model.add(Dense(10, input_dim=inp, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
#model.add(Flatten())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
def lr_scheduler(epoch, lr):
    if epoch < 50:
        lr = 0.001
        return lr
    return lr
model_checkpoint = ModelCheckpoint('best.hdf5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
callback=[model_checkpoint,tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

model.fit(train_data, to_categorical(train_labels), epochs=10, batch_size=32,validation_data=(validation_data,to_categorical(validation_labels)),callbacks=callback)
model.load_weights('best.hdf5')
print(model.evaluate(validation_data,to_categorical(validation_labels)))
preds=model.predict(test_data)
out=zip(test_idx,preds)
with open("submission1.csv", "w") as g:
    g.write("id,label\n")
    for predict in out:
        g.write(str(predict[0]) + ',' + str(predict[1].argmax()) + '\n')

with open("submission2.txt", "w") as g:
    g.write("id,label\n")
    for predict in out:
        g.write(str(predict[0]) + ',' + str(predict[1]) + '\n')
# Final evaluation of the model
#print(sklearn.metrics.f1_score(validation_labels,preds,average='macro'))
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
