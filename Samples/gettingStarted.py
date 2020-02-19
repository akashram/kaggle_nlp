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

def main()
    #Load Data into dataframe
    train_df = pd.read_csv("../sampleData/train.csv")
    test_df = pd.read_csv("../sampleData/nlp-getting-started/test.csv")

    '''
    We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.
    '''
    count_vectorizer = feature_extraction.text.CountVectorizer()

    ## let's get counts for the first 5 tweets in the data
    example_train_vectors = count_vectorizer.fit_transform(
        train_df["text"][0:5])





if __name__=="__main__":
    main()
