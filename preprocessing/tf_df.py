import pandas as pd

from preprocessing.utils import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer


def text_mining_tfdf(X_train: pd.DataFrame, X_test: pd.DataFrame, min_df=0.01) -> tuple([pd.DataFrame, pd.DataFrame]):
    """Applies the tf-df to the text feature of the train and the evaluation set
    Parameters
    ----------
    X_train : pd.DataFrame
        Devolopment set dataframe
    X_test : pd.DataFrame
        Evaluation set dataframe
    min_df : float
        Minimum support for the tf-df. If is between (0, 1) is percentual, if it is >1 its absolute value is considered
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame])
        A tuple with X_train and X_test with the tf-df applied
    """
    
    vectorizer = TfidfVectorizer(strip_accents="ascii", stop_words = STOPWORDS, use_idf=False, min_df=min_df)
    vectorizer.fit(X_train["text"])
    train_tfdf = pd.DataFrame(vectorizer.transform(X_train["text"]).toarray(), columns=vectorizer.get_feature_names())
    test_tfdf = pd.DataFrame(vectorizer.transform(X_test["text"]).toarray(), columns=vectorizer.get_feature_names())

    # Concat features obtained by TF-DF with the original DataFrame
    X_train = pd.concat([X_train.reset_index(drop=True),train_tfdf.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True),test_tfdf.reset_index(drop=True)], axis = 1) 
    return X_train, X_test