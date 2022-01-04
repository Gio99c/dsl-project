from os import device_encoding, remove
import numpy as np
import pandas as pd
import regex as re
from scipy.sparse.construct import rand
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
import string
import fasttext
from emot.emo_unicode import EMOTICONS_EMO
import html


class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)
            lemmas.append(lemma)
        return lemmas


def convert_emoticons(text:str) -> str:
    "Convert the emoticons in text such as ':)' by using the EMOTICONS_EMO dict"
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), EMOTICONS_EMO[emot], text)
    return text


def load(filepath="./DSL2122_january_dataset/development.csv") -> pd.DataFrame:
    """Load the dataset
    Parameters
    ----------
    filepath: string
        filepath of the dataset to load
    Returns
    -------
    pd.DataFrame
        the loaded dataframe
    """

    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except FileNotFoundError:
        print('File not found')


def extract_features(tweets: pd.DataFrame) -> pd.DataFrame:
    """Extract different features from the existsting ones
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    Returns
    -------
    pd.DataFrame
        the same dataframe with the new features
    """

    new_cols_date = ["day_of_week", "month_of_year", "day_of_month", "time"]

    tweets[new_cols_date] = tweets['date'].str.split(' ', expand=True)[[0,1,2,3]]
    tweets["hour_of_day"] = tweets['time'].str.split(':', expand=True)[0].astype(int)

    # Cast int of day of the month
    #tweets['day_of_month']= pd.to_numeric(tweets['day_of_month'])

    # Drop columns that are no important 
    cols_to_drop = ["date", "time", "flag"]
    tweets.drop(columns= cols_to_drop, inplace=True)

    # Extract the numb of chars in the text
    tweets['char_count'] = list(map(lambda x: len(x), tweets['text'])) 

    return tweets


def drop_duplicates(tweets: pd.DataFrame, drop_long_text=False, k=140) -> pd.DataFrame:
    """!! Only appliable for train set !! - Drop the duplicated tweets and (optionally) the tweets that are longer than k characters
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    drop_long_text: bool
        Decide to drop or not the text longer than k
    k: integer
        Lenght of the tweets to be considered "long"
    Returns
    -------
    pd.DataFrame
        the same dataframe with the function applied
    """
    tweets.drop_duplicates(subset=['text', 'sentiment'], keep='first', inplace=True)
    tweets.drop_duplicates(subset=['text'], inplace=True, keep=False)

    if drop_long_text:
        tweets = tweets.loc[tweets['char_count'] <= k]

    return tweets


def clean_text(tweets: pd.DataFrame) -> pd.DataFrame:
    """Clean the text feature from unrelevant information and convert the emoticons into text

    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets

    Returns
    -------
    pd.DataFrame
        the same dataframe with the function applied
    """
    # Convert HTML entities into characters
    tweets["text"] = tweets["text"].apply(lambda x : html.unescape(x))

    # regex pattern for the site domain extraction
    urls = "(www\.)|(https?:\/\/)|(\.((com)|(ly)|(it)|(to)|(fm)|(co)|(me)|(gov)|(net)|(org)|(uk)|(im)|(gd)|(cc))[\/\w\d-~_\.]*)"
    tweets['text'] = tweets['text'].str.replace(pat=urls, repl="", regex=True)

    # convert emoticons into text
    tweets['text'].apply(lambda x: convert_emoticons(x))

    return tweets


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
    #tweets["text"] = list(map(lambda x: re.sub("(@[\d\w]+)|(https?://[\w\d]+)|((www\.)[\d\w]+)|", "", x), tweets["text"])) #text cleaning (urls and users @)
    lemmaTokenizer = LemmaTokenizer()                                                                      
    vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words('english'), strip_accents="ascii", use_idf=False, min_df=min_df)
    vectorizer.fit(X_train["text"])
    train_tfdf = pd.DataFrame(vectorizer.transform(X_train["text"]).toarray(), columns=vectorizer.get_feature_names())
    test_tfdf = pd.DataFrame(vectorizer.transform(X_test["text"]).toarray(), columns=vectorizer.get_feature_names())

    # we can use pd.concat it immediately cast to dataframe
    X_train = pd.DataFrame(np.column_stack([X_train,train_tfdf]), columns=X_train.columns.append(train_tfdf.columns)) #stack the two dataframes horizontally
    X_test = pd.DataFrame(np.column_stack([X_test,test_tfdf]), columns=X_test.columns.append(test_tfdf.columns))

    return X_train, X_test


def text_mining_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
    """Extract the sentiment from the text feature
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    Returns
    -------
    pd.DataFrame
        the same dataframe with six new features: neg, neu, pos, compound, polarity, subjectivity
    """
    #tweets["text"] = list(map(lambda x: re.sub("(@[\d\w]+)|(https?://[\w\d]+)|((www\.)[\d\w]+)|", "", x), tweets["text"])) #text cleaning (urls and users @)

    sentiment = np.array(list(map(lambda x: list(SentimentIntensityAnalyzer().polarity_scores(x).values()), tqdm(tweets["text"]))))
    tweets["neg"] = sentiment[:, 0]
    tweets["neu"] = sentiment[:, 1]
    tweets["pos"] = sentiment[:, 2]
    tweets["compound"] = sentiment[:, 3]
    tweets["polarity"] = list(map(lambda x: TextBlob(x).sentiment.polarity, tqdm(tweets["text"])))
    tweets["subjectivity"] = list(map(lambda x: TextBlob(x).sentiment.subjectivity, tqdm(tweets["text"])))
    return tweets

def add_user_text(tweets: pd.DataFrame) -> pd.DataFrame:
    """Add the username as the first word (with the @) in the text feature
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    Returns
    -------
    pd.DataFrame
        Same dataframe with the function applied
    """
    tweets["text"] = np.array("@") + tweets["user"].values + np.array(" ") + tweets["text"].values
    tweets.drop(columns=["user"], inplace=True)

    return tweets
    

def add_word_embeddings(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    """Add the word embeddings scores to the development and evaluation set, based on the vocabolury of the training set.
    Parameters
    ----------
    X_train : pd.DataFrame
        Devolopment set dataframe
    X_test : pd.DataFrame
        Evaluation set dataframe
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame, pd.Series])
        a tuple with X_train, X_test and y_train
    """
    print('Sto facendo autovalidation')
    
    
    X_valid_train, X_valid_test, y_valid_train, y_valid_test = train_test_split(X_train, X_train["sentiment"], test_size=0.2, stratify= X_train["sentiment"], random_state=42)
    text_train = np.array("__label__") + X_valid_train["sentiment"].astype("str").values + np.array(" ") + X_valid_train["text"].values
    text_valid = np.array("__label__") + X_valid_test["sentiment"].astype("str").values + np.array(" ") + X_valid_test["text"].values
    np.savetxt("train.txt", text_train, fmt="%s")
    np.savetxt("valid.txt", text_train, fmt="%s")
    model = fasttext.train_supervised("train.txt", autotuneValidationFile="valid.txt")
    remove("train.txt")
    remove("valid.txt")

    #lo so, questa cosa è poco leggibile, la migliorerò
    scores_dev = pd.DataFrame([map(lambda x : x[1], sorted(zip(model.predict(text, k=2)[0],model.predict(text, k=2)[1]), key=lambda x : x[0])) for text in X_train["text"]], columns=["embedding_negativity", "embedding_positivity"])
    scores_eval = pd.DataFrame([map(lambda x : x[1], sorted(zip(model.predict(text, k=2)[0],model.predict(text, k=2)[1]), key=lambda x : x[0])) for text in X_test["text"]], columns=["embedding_negativity", "embedding_positivity"])
    
    X_train = pd.DataFrame(np.column_stack([X_train, scores_dev]), columns=X_train.columns.append(scores_dev.columns))
    X_test = pd.DataFrame(np.column_stack([X_test, scores_eval]), columns=X_test.columns.append(scores_eval.columns))

    return X_train, X_test

def convert_categorical(tweets: pd.DataFrame) -> pd.DataFrame:
    """Converts the features that are not categorical
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    Returns
    -------
    pd.DataFrame
        the same dataframe with the function applied
    """
    tweets.drop(columns=["text"], inplace=True) #when the schema will be fixed, this will be moved in the last text processing step

    day_of_week_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    months_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    
    tweets["day_of_week"] = list(map(lambda x: day_of_week_dict[x], tweets["day_of_week"]))
    tweets["month_of_year"] = list(map(lambda x: months_dict[x], tweets["month_of_year"]))
    tweets["day_of_month"] = list(map(lambda x: int(x), tweets["day_of_month"]))
    tweets["hour_of_day"] = list(map(lambda x: int(x), tweets["hour_of_day"]))

    return tweets

    

def normalize(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame, pd.Series]):
    """Applies a MinMaxScaler to all the features of the dataframes and pops y_train from X_train
    Parameters
    ----------
    X_train : pd.DataFrame
        Devolopment set dataframe
    X_test : pd.DataFrame
        Evaluation set dataframe
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame, pd.Series])
        a tuple with X_train, X_test and y_train
    """

    y_train = X_train.pop("sentiment").astype('int')
    scaler = MinMaxScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train

def save_results(y_pred: list, fp: str):
    """Saves the prediction of the classification model in the sample_submission.csv file
    Parameters
    ----------
    y_pred : list
        The list of predictions made by the classifier
    """

    pd.Series(y_pred, name="Predicted").to_csv(fp, index_label="Id")