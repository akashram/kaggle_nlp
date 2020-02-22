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
    #Load Data into dataframe
    train_df = pd.read_csv("../sampleData/train.csv")
    test_df = pd.read_csv("../sampleData/test.csv")

    '''
    We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.
    '''
    count_vectorizer = feature_extraction.text.CountVectorizer()

    train_vectors = count_vectorizer.fit_transform(train_df["text"])

    ## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
    # that the tokens in the train vectors are the only ones mapped to the test vectors -
    # i.e. that the train and test vectors use the same set of tokens.
    test_vectors = count_vectorizer.transform(test_df["text"])

    ## Our vectors are really big, so we want to push our model's weights
    ## toward 0 without completely discounting different words - ridge regression
    ## is a good way to do this.
    clf = linear_model.RidgeClassifier()

    clf.fit(train_vectors, train_df["target"])

    sample_submission = pd.read_csv("../sampleData/sample_submission.csv")

    sample_submission["target"] = clf.predict(test_vectors)

    sample_submission.to_csv("akash_linear_regression_submission.csv", index=False)


if __name__=="__main__":
    main()
