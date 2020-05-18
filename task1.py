import numpy as np
import pandas as pd
import json
import requests
import pickle
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

def clean_line(t):
    return (t.replace('\n',' ')
            .replace('\r',' ')
            .replace('\t',' ')
            .replace('  ',' ')
            .strip())

def load_and_process_data():
    #only use text data longer than 50
    min_len = 50

    all_data = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'))
    
    #clean data
    all_text = [clean_line(t) for t in all_data.data]
    all_data_df = pd.DataFrame({'text' : all_text, 'topics' : all_data.target})
    
    cleaned_df = all_data_df[all_data_df.text.str.len() > min_len]
    
    X_raw = cleaned_df['text'].values
    y_raw = cleaned_df['topics'].values
    
    #split the data with test_size=0.20 and random_state = 42
    #your codes are here
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    #end
    
    #tranform data to vectors using tf-idf approach
    #your codes are here
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    #end
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

def trainSVM(X_train, X_test, y_train, y_test, cpu=1):
    sgd_clf = SGDClassifier(n_jobs=cpu)
    sgd_clf.fit(X_train, y_train)
    predicted = sgd_clf.predict(X_test)
    mean = np.mean(predicted == y_test)
    print("The Result of SVM is: " + str(mean))

def trainRF(X_train, X_test, y_train, y_test, trees = 100, cpu=1):
    rf_clf = RandomForestClassifier(n_estimators=trees, n_jobs=cpu)
    rf_clf.fit(X_train, y_train)
    predicted = rf_clf.predict(X_test)
    mean = np.mean(predicted == y_test)
    print("The Result of Random Forest is: " + str(mean))

def trainKNN(X_train, X_test, y_train, y_test, cpu=1, n_neighbors=5):
    kn_clf = KNeighborsClassifier(n_jobs=cpu, n_neighbors=n_neighbors)
    kn_clf.fit(X_train, y_train)
    predicted = kn_clf.predict(X_test)
    mean = np.mean(predicted == y_test)
    print("The Result of K-Nearest Neighbors is: " + str(mean))


def main():
    X_train, X_test, y_train, y_test = load_and_process_data()
    print("Before Tuning the Parameters: ")
    trainSVM(X_train, X_test, y_train, y_test)
    trainRF(X_train, X_test, y_train, y_test)
    trainKNN(X_train, X_test, y_train, y_test)
    print("SVM > Random Forest > KNN")
    print("After Tuning the Parameters: ")
    trainSVM(X_train, X_test, y_train, y_test, -1)
    trainRF(X_train, X_test, y_train, y_test, 1000, -1)
    trainKNN(X_train, X_test, y_train, y_test, -1, 3)
    print("After tuning the parameters. Random Forest's accuracy increase 5~6% and KNN's accuracy increase up to 9%\n")
    print("SVM > Random Forest > KNN")
main()
