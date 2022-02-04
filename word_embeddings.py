# Standard libraries
import numpy as np
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split

# FastText
import fasttext
from os import remove


def print_best_params(X_train: pd.DataFrame) -> None:
    
    # Print best parameters of FastText given from autovalidation

    print('Starting autovalidation of Fast Text')
    print("Split the dataset in training and test set")

    X_valid_train, X_valid_test, _, _ = train_test_split(X_train, X_train["sentiment"], test_size=0.2, stratify= X_train["sentiment"], random_state=42)
   
    text_train = np.array("__label__") + X_valid_train["sentiment"].astype("str").values + np.array(" ") + X_valid_train["text"].values
    text_valid = np.array("__label__") + X_valid_test["sentiment"].astype("str").values + np.array(" ") + X_valid_test["text"].values

    np.savetxt("train.txt", text_train, fmt="%s")
    np.savetxt("valid.txt", text_valid, fmt="%s")

    print("Start autovalidation")
    model = fasttext.train_supervised("train.txt", autotuneValidationFile="valid.txt")
    print(" End autovalidation")

    # Print hyper parameters of the FastText
    args_obj = model.f.getArgs()

    for hparam in dir(args_obj):
        if not hparam.startswith('__'):
            print(f"{hparam} -> {getattr(args_obj, hparam)}") 

    remove("train.txt")
    remove("valid.txt")


def add_word_embeddings(X_valid_train: pd.DataFrame, X_valid_test: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    """Add the word embeddings scores to the development and evaluation set, based on the vocabolury of the training set.
    Parameters
    ----------
    X_valid_train : pd.DataFrame
        Devolopment set dataframe
    X_valid_test : pd.DataFrame
        Evaluation set dataframe
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame, pd.Series])
        A tuple with X_train, X_test and y_train
    """

    print('Starting word embeddings')
    
    text_train = np.array("__label__") + X_valid_train["sentiment"].astype("str").values + np.array(" ") + X_valid_train["text"].values
    
    np.savetxt("train.txt", text_train, fmt="%s")   
    
    model = fasttext.train_supervised("train.txt", epoch = 4, wordNgrams=2, bucket =  966320, dim = 126, lr =0.20158291921508406, lrUpdateRate = 100, maxn = 6,
                                    minCount = 1, minCountLabel = 0 ,minn = 3, neg = 5, seed = 0, t= 0.0001, thread = 15, verbose = 2, ws =5)
    remove("train.txt")
    
    
    # Create features 
    scores_dev = []
    scores_eval = []

    for text in X_valid_train["text"]:
        prediction = model.predict(text, k=2)
        scores_dev.append(map(lambda x : x[1], sorted(zip(prediction[0],prediction[1]), key=lambda x : x[0])))

    for text in X_valid_test["text"]:
        prediction = model.predict(text, k=2)
        scores_eval.append(map(lambda x : x[1], sorted(zip(prediction[0],prediction[1]), key=lambda x : x[0])))
    
    new_features_names= ["embedding_negativity", "embedding_positivity"]
    scores_dev = pd.DataFrame(scores_dev, columns=new_features_names)
    scores_eval = pd.DataFrame(scores_eval, columns=new_features_names)
    
    
    X_train = pd.concat([X_valid_train, scores_dev], axis = 1)
    X_test = pd.concat([X_valid_test, scores_eval], axis=1)

    return X_train, X_test