from __future__ import annotations
import os
import sys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


'''
Going Through the sample Notebook:
https://www.kaggle.com/philculliton/nlp-getting-started-tutorial#NLP-Tutorial

'''

def main():
    #Load Data into dataframe - which is its own can of worms
    train_df = pd.read_csv("../sampleData/train.csv")
    test_df = pd.read_csv("../sampleData/test.csv")

    '''
    We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.
    '''
    count_vectorizer = feature_extraction.text.CountVectorizer()

    ## let's get counts for the first 5 tweets in the data
    example_train_vectors = count_vectorizer.fit_transform(
        train_df["text"][0:5])

    #The First Val that is a neg and then a positive... 
    print(train_df[train_df["target"] == 0]["text"].values[1])
    print(train_df[train_df["target"] == 1]["text"].values[1])



    ####Vector Schenanigans

    #sklearn's countvectorizer class turns text tokens into a frequency matrix
    count_vectorizer = feature_extraction.text.CountVectorizer()

    #This will put in a slice of the first 5 entries (as seen in training set) 
    example_train_vectors = count_vectorizer.fit_transform(
        train_df["text"][0:5])

    # example_train_vectors is a 5 x T matrix where T=numTokens

    print(example_train_vectors[1].todense().shape) #shape gets the array dimensions
    print(example_train_vectors[1].todense()) #todense: spare matrix -> dense matrix

    
    #fit transform vs transform: https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models'''
    train_vectors = count_vectorizer.fit_transform(train_df["text"])
    test_vectors = count_vectorizer.transform(test_df["text"])
    '''
    fit transform normalizes the data to to have an average centered on 0
    fit calcs the nu and sigma values in the std dev. and stores them into the state of the obj -> then it calls transform. 

    We're using transform here because we don't want to fit the tokens from the test set on the test set. we want the calculated fit values from the training set to be applied
    '''

    #### Model Schenanigans 
    clf = linear_model.RidgeClassifier() #We're going to do linear regression, Ridge Classifier outputs either 1 or -1


    scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1") 
    # this is the meat of the operaton so let's take some time to unpack this.
    # Cross Validation takes a portion of the set for training and uses the rest to validate. You can chunk in different ways to get different metrics.


    # "The above scores aren't terrible! It looks like our assumption will score roughly 0.65 on the leaderboard. There are lots of ways to potentially improve on this (TFIDF, LSA, LSTM / RNNs, the list is long!) - give any of them a shot!" - let's try those too

    #Fit/Classify all the training data
    clf.fit(train_vectors, train_df["target"]) 
    '''
    RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, 
        fit_intercept=True, max_iter=None, normalize=False, random_state=None,
            solver='auto', tol=0.001)
    '''

    #Open the file to write to use it as template to write to.
    sample_submission = pd.read_csv("../sampleData/sample_submission.csv")
    sample_submission["target"] = clf.predict(test_vectors) #this runs our classifier
    sample_submission.head()
    sample_submission.to_csv("../sampleData/submission.csv", index=False) #this is our sample submission lol
if __name__=="__main__":
    main()


