# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 22:52:11 2019

@author: Owais
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('labeledTrainData.tsv', delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv', delimiter = '\t', quoting = 3)

#combining dataset
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0,sort=False).reset_index(drop=True)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 9000)
X_train_test = cv.fit_transform(corpus).toarray()
Y_train = dataset.iloc[:, 1].values

'''# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)'''

#let separate training and test set 
X_train = X_train_test[:train_len]
Y_train = Y_train[:train_len]
X_test = X_train_test[train_len:]

#importing accuracy matrics
from sklearn.metrics import accuracy_score
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
gNB= GaussianNB()
gNB.fit(X_train, Y_train)
y_pred = gNB.predict(X_test)
Y_pred_acc=gNB.predict(X_train)
acc_gNB = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_gNB)

#Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=51,criterion='entropy',random_state=0)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)
Y_pred_acc=rfc.predict(X_train)
acc_rfc = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_rfc)

#Gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=50,)
gbk.fit(X_train, Y_train)
y_pred = gbk.predict(X_test)
Y_pred_acc=gbk.predict(X_train)
acc_gbk = round(accuracy_score(Y_pred_acc, Y_train) * 100, 2)
print(acc_gbk)



Classifiers = pd.DataFrame({
    'Classifier': ['GaussianNB', 'RandomForestClassifier', 'GradientBoostingClassifier'],
    'Accuracy': [acc_gNB, acc_rfc, acc_gbk]})
Classifiers.sort_values(by='Accuracy', ascending=False)


#set the output as a dataframe and convert to csv file named submission.csv
ids=test['id']
output = pd.DataFrame({ 'id' : ids, 'sentiment': y_pred })
output.to_csv('Submission.csv', index=False)