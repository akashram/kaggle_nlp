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
    train = pd.read_csv("../../sampleData/train.csv")
    test = pd.read_csv("../../sampleData/test.csv")

    tfidf = feature_extraction.text.TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train['text'])
    test_tfidf = tfidf.transform(test["text"])

    # Fitting a simple Logistic Regression on TFIDF
    clf_tfidf = linear_model.LogisticRegression(C=1.0)
    clf_tfidf.fit(train_tfidf, train["target"])
    scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

    test_vectors=test_tfidf
    sample_submission = pd.read_csv("../../sampleData/sample_submission.csv")
    sample_submission["target"] = clf_tfidf.predict(test_vectors)
    sample_submission.to_csv("akash_logistic_regression_tfidf_submission.csv", index=False)


if __name__=="__main__":
    main()
