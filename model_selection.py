import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

import time


def test_models(X : pd.DataFrame, y : pd.Series) -> pd.DataFrame:
    # Test 4 models and print results 
    models = [LinearSVC(random_state=42), RandomForestClassifier(random_state=42), BernoulliNB(), HistGradientBoostingClassifier(random_state=42)]
    model_names= ["Linear SVC", "Random Forest", "Bernoulli Nayve-Bayes", "Hist GBC"]
    
    f1_scores = []

    for clf, name in zip(models, model_names):

        print(f'Start training model: {name}')

        start_time = time.time()
        f1_score = cross_val_score(clf, X, y, cv = 5, scoring = 'f1_macro')
        finish_time = time.time()

        print(f'Finishing training model: {name}, trained in {finish_time-start_time}')

        f1 = f1_score.mean()
        f1_scores.append(f1)

        print(f'Score of {name} model performed: {f1}\n\n\n')

        col1 = pd.Series(model_names)
        col2 = pd.Series(f1_scores)
            

    result = pd.concat([col1, col2], axis = 'columns')
    result.columns = ['Model Name', 'F1 Score']
    
    return result


def test_diff_preprocessing(X_train: pd.DataFrame, y_train: pd.DataFrame):
    # Test all models with different preprocessing techiniques

    # WE + TF-DF + Sentiment 
    print("Test Word Embeddings + TF-DF + Sentiment: \n\n")
    
    result_1 = test_models(X_train, y_train)

    
    # TF-DF + Sentiment 
    X_train.drop(columns=["embedding_negativity", "embedding_positivity"] , inplace=True)
    print("Test TF-DF + Sentiment: \n\n")
   
    result_2 = test_models(X_train, y_train)

    
     # Only Sentiment 
    cols_not_to_drop= ["ids", "day_of_week", "month_of_year", "day_of_month", "hour_of_day", "char_count", "neg", "neu", "pos", "compound", "polarity",
    "subjectivity"]
    col_tf_df =[col for col in X_train.columns if col not in cols_not_to_drop]
    X_train.drop(columns=col_tf_df , inplace=True)
    print("Test  Sentiment: \n\n")

    result_3 = test_models(X_train, y_train)


    return result_1, result_2, result_3