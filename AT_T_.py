# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:18:30 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as ps
import re


nltk.download('stopwords') #file contains words that has to ignore
dataset = pd.read_csv('atdata.csv')
#dataset['Titles'][0]
d#ataset['Reviews'][1]

processed_titles = []
processed_reviews = []


#X = dataset.iloc[:,3:5].values
y = dataset.iloc[:, 5]

for i in range(113):
    Titles = re.sub('@[\w]*', ' ', dataset['Titles'][i])
    Titles = re.sub('[^a-zA-Z#',' ', Titles)
    Titles = Titles.lower()
    Titles = Titles.split()
    Titles = [ps.stem(token) for token in Titles if not token  in stopwords.words('english')]   
    Titles = ' '.join(Titles)
    processed_titles.append(Titles)



for i in range(113):
    Reviews = re.sub('@[\w]*', ' ', dataset['Reviews'][i])
    Reviews = re.sub('[^a-zA-Z#',' ', Reviews)
    Reviews = Reviews.lower()
    Reviews = Reviews.split()
    Reviews = [ps.stem(token) for token in Reviews if not token  in stopwords.words('english')]   
    Reviews = ' '.join(Reviews)
    processed_reviews.append(Reviews)    
    
    
    
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(processed_titles)
X = cv.fit_transform(processed_reviews)
X = X.toarray()
#y = dataset['Label'].values


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X, y)
nb.score(X, y)

print(cv.get_feature_names())


    


            



