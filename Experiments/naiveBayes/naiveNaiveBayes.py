from __future__ import annotations
import os
import sys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


'''
Reference: https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b

Reference for Bayes Theorem: https://www.youtube.com/watch?v=HZGCoVF3YvM

Let's try to implement Naive bayes in a number of different ways.
It's a tad barbaric to implement it all by hand right? Let's try leveraging
SK-learn an the other tools we have as opposed to the other two time 
we both had manually implement it. 


'''





def naiveBayesEasyMode():
     #Load Data into dataframe - which is its own can of worms
    train_df = pd.read_csv("../../sampleData/train.csv")
    test_df = pd.read_csv("../../sampleData/test.csv")
    ps = PorterStemmer()

    #Stem Things
    for i, tweet in enumerate(train_df["text"]):
        tokens = word_tokenize(tweet)
        for index, word in enumerate(tokens):
            tokens[index] = ps.stem(word)
        print(tokens)
        newLine =" ".join(tokens)
        train_df["text"]
        break

    print(train_df)
    #sklearn's countvectorizer class turns text tokens into a frequency matrix
    count_vectorizer = feature_extraction.text.CountVectorizer()

    #This will put in a slice of the first 5 entries (as seen in training set) 
    example_train_vectors = count_vectorizer.fit_transform(
        train_df["text"][0:5])

    # example_train_vectors is a 5 x T matrix where T=numTokens

    #print(example_train_vectors[1].todense().shape) #shape gets the array dimensions
    #print(example_train_vectors[1].todense()) #todense: spare matrix -> dense matrix

def naiveBayesSpicy():
    pass


if __name__ == "__main__":
    naiveBayesEasyMode()


