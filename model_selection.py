import pandas as pd
import numpy as np
import seaborn as sns

import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score


# Plot
import seaborn as sns
import matplotlib.pyplot as plt


# Feature Extraction
import regex as re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# See time 
from tqdm import tqdm
import time

# PreProcessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Model Selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Parameters
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

clf_models = [LinearSVC(), RandomForestClassifier(), BernoulliNB(), MLPClassifier()]
clf_names = [' Linear Support Vector Machine', 'Random Forest', 'Nayve Bayes Bernoulli', 'Multi Layer Neural Network']


def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, models: list, names: list) -> pd.DataFrame:

    """Inputs:
        X_train = training data set
        X_test = testing data set
        y_train = training label 
        y_test = testing label 
        models = list of models to compare
        names = list of trings containing the names of the models 

        Return: comparison table 
        """

    f1_scores, precision_scores, recall_scores, accuracy_scores = [], [], [],[]
    
    for clf, name in zip(models, names):

        print(f'Start training model: {name}')

        start_time = time.time()

        clf.fit(X_train, y_train)
        print('\n')

        finish_time = time.time()

        print(f'Fiishing training model: {name}, trained in {finish_time-start_time}\n')

        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average = 'macro')
        f1_scores.append(f1)

    
        print(f'Score of {name} model performed: {f1}')

    col1 = pd.Series(names)
    col2 = pd.Series(f1_scores)
        

    result = pd.concat([col1, col2], axis = 'columns')
    result.columns = ['Model Name', 'F1 Score']
    return result



def compare_models_KF(X: pd.DataFrame, y: pd.Series, models: list, names: list, k = int) -> pd.DataFrame:

    """Inputs:
        X = training data set
        y = training label  
        models = list of models to compare
        names = list of trings containing the names of the models 
        k = int for the cross validation 

    
        Return: comparison table """

    f1_scores = []
        
    for clf, name in zip(models, names):

        print(f'Start training model: {name}')

        start_time = time.time()
        f1_score = cross_val_score(clf, X, y, cv = k, scoring = 'f1_macro')
        finish_time = time.time()

        print(f'Finishing training model: {name}, trained in {finish_time-start_time}\n')

        f1 = f1_score.mean()
        f1_scores.append(f1)

        print(f'Score of {name} model performed: {f1}\n\n\n')

    col1 = pd.Series(names)
    col2 = pd.Series(f1_scores)
        

    result = pd.concat([col1, col2], axis = 'columns')
    result.columns = ['Model Name', 'F1 Score']

    return result