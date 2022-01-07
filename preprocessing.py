from os import remove
from fasttext.FastText import load_model
#from nltk.corpus.reader import util
import numpy as np
import pandas as pd
import regex as re
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
#from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm
import fasttext
from emot.emo_unicode import EMOTICONS_EMO
import html
import tldextract

from emot.emo_unicode import EMOTICONS_EMO
from utils import CONTRACTIONS, SLANGS, STOPWORDS



def expand_contraction_form(text:str) -> str:
    # Replace the contractions from a string to their equivalent by using CONTRCTIONS dic in utils
    for word in text.split(sep= " "):
        if word in CONTRACTIONS.keys():
            text = re.sub(word , CONTRACTIONS[word], text)
    return text


def convert_slangs(text:str) -> str:
    # Replace the slangs with their equivalent by using SLANG dic in utils
    for word in text.split(sep= " "):
        if word in SLANGS.keys():
            text = re.sub(word , SLANGS[word], text)
    return text


def convert_emoticons(text:str) -> str:
    # Replace the emoticons with their equivalent by using EMOTICONS_EMO dic in emo.unicode_emo
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), EMOTICONS_EMO[emot], text)
    return text


def word_lemmatizer(text: str) -> str:
    lem_text = [WordNetLemmatizer().lemmatize(i.strip()) for i in text]
    return lem_text


def load_dataset(filepath="./DSL2122_january_dataset/development.csv") -> pd.DataFrame:
    # Return the dataset
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except FileNotFoundError:
        print('File not found')


def extract_date_features(tweets: pd.DataFrame) -> pd.DataFrame:
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

    day_of_week_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    months_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    new_cols_date = ["day_of_week", "month_of_year", "day_of_month", "time"]

    tweets[new_cols_date] = tweets['date'].str.split(' ', expand=True)[[0,1,2,3]]
    tweets["hour_of_day"] = tweets['time'].str.split(':', expand=True)[0].astype(int) # Extract only hour from time
    tweets['day_of_month']= pd.to_numeric(tweets['day_of_month'])   # Cast int day_of_month

    # Convert to categorical
    tweets["day_of_week"] = list(map(lambda x: day_of_week_dict[x], tweets["day_of_week"]))
    tweets["month_of_year"] = list(map(lambda x: months_dict[x], tweets["month_of_year"]))

    
    return tweets


def count_characters(tweets: pd.DataFrame, trainset:bool) -> pd.DataFrame:
    # Extract the numb of chars in the text and drop too long tweets
    tweets['char_count'] = list(map(lambda x: len(html.unescape(x)), tweets['text']))
    if trainset:
        tweets = tweets[tweets['char_count'] <= 140]

    return tweets


# ONLY FOR TRAIN SET
def drop_duplicates(tweets: pd.DataFrame) -> pd.DataFrame:

    # Drop the copies tweet of the same user, keep only the first and 
    # Drop all the tweets with the same text but different sentiment
    tweets.drop_duplicates(subset=['text', "sentiment", "user"], inplace=True, keep="first") 
    tweets = tweets[~(tweets.duplicated(subset="text") & ~tweets.duplicated(subset=["text", "sentiment"]))]

    return tweets


def print_best_params(X_train: pd.DataFrame) -> None:
    
    print('Testing word embeddings')
    
    X_valid_train, X_valid_test, _, _ = train_test_split(X_train, X_train["sentiment"], test_size=0.2, stratify= X_train["sentiment"], random_state=42)
    
    text_train = np.array("__label__") + X_valid_train["sentiment"].astype("str").values + np.array(" ") + X_valid_train["text"].values
    text_valid = np.array("__label__") + X_valid_test["sentiment"].astype("str").values + np.array(" ") + X_valid_test["text"].values

    np.savetxt("train.txt", text_train, fmt="%s")
    np.savetxt("valid.txt", text_valid, fmt="%s")

    model = fasttext.train_supervised("train.txt", autotuneValidationFile="valid.txt")
    args_obj = model.f.getArgs()

    for hparam in dir(args_obj):
        if not hparam.startswith('__'):
            print(f"{hparam} -> {getattr(args_obj, hparam)}") 

    remove("train.txt")
    remove("valid.txt")


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
    
    
    
    text_train = np.array("__label__") + X_train["sentiment"].astype("str").values + np.array(" ") + X_train["text"].values

    np.savetxt("train.txt", text_train, fmt="%s")

    model = fasttext.train_supervised("train.txt", epoch =5, wordNgrams=2, wsbucket = 1003902, cutoff = 0, dim = 136, dsub= 2, label = "__label__", 
                                     lr =0.20784007534156418, lrUpdateRate = 100, maxn = 6, minCount = 1, minCountLabel = 0 ,minn = 3, neg = 5, qnorm = False, qout = False,
                                     retrain = False, saveOutput = False,seed = 0, t= 0.0001, thread = 15, verbose = 2, ws =5)
    remove("train.txt")

    scores_dev = []
    scores_eval = []
    for text in X_train["text"]:
        prediction = model.predict(text, k=2)
        scores_dev.append(map(lambda x : x[1], sorted(zip(prediction[0],prediction[1]), key=lambda x : x[0])))

    for text in X_test["text"]:
        prediction = model.predict(text, k=2)
        scores_eval.append(map(lambda x : x[1], sorted(zip(prediction[0],prediction[1]), key=lambda x : x[0])))
    
    new_features_names= ["embedding_negativity", "embedding_positivity"]
    scores_dev = pd.DataFrame(scores_dev, columns=new_features_names)
    scores_eval = pd.DataFrame(scores_eval, columns=new_features_names)
    
    
    X_train = pd.DataFrame(np.column_stack([X_train, scores_dev]), columns=X_train.columns.append(scores_dev.columns))
    X_test = pd.DataFrame(np.column_stack([X_test, scores_eval]), columns=X_test.columns.append(scores_eval.columns))

    return X_train, X_test


def clean_text(tweets: pd.DataFrame) -> pd.DataFrame:
    """Clean the text feature from irrelevant informations. We applied these changes:
       1) Expand contractions forms
       2) Convert HTML entities
       3) Convert text in loewr case
       4) Extract  URL domains
       5) Remove duplicated characters
       6) Remove numbers
       7) Replace emoticons
       8) Replace slangs
       9) Remove hashtags and mentioned users
       10) Lemmatize text
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    deep_clean: boo
        if True performs a deeper cleaning by removing number, hashtags and mentioned users. Default False
    Returns
    -------
    pd.DataFrame
        the same dataframe with the function applied
    """

    tweets["text"] = tweets['text'].apply(lambda x: expand_contraction_form(x)) #1

    tweets["text"] = tweets["text"].apply(lambda x : html.unescape(x)) #2

    tweets["text"] = tweets["text"].str.lower() #3

    tweets["text"] = tweets['text'].str.replace(pat="((https?:\/\/)?([w]+\.)?\S+)", repl=lambda x: tldextract.extract(x.group(1)).domain, regex=True) #4

    tweets["text"] = tweets["text"].apply(lambda x: re.sub("(\w)\\1{2,}", "\\1\\1", x)) #5

    tweets["text"] = tweets["text"].apply(lambda elem: re.sub(r"\d+", "", elem)) #6

    tweets["text"] = tweets['text'].apply(lambda x: convert_emoticons(x)) #7
   
    tweets["text"] = tweets['text'].apply(lambda x: convert_slangs(x)) #8

    tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"(@[\w\d]+)|#", "", x)) #9

    tweets["text"] = tweets["text"].apply(lambda x: " ".join(word_lemmatizer(x.split()))) #10


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
    
    vectorizer = TfidfVectorizer(strip_accents="ascii", stop_words = STOPWORDS , use_idf=False, min_df=min_df)
    vectorizer.fit(X_train["text"])

    train_tfdf = pd.DataFrame(vectorizer.transform(X_train["text"]).toarray(), columns=vectorizer.get_feature_names())
    test_tfdf = pd.DataFrame(vectorizer.transform(X_test["text"]).toarray(), columns=vectorizer.get_feature_names())

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

    # EXPLAIN 
    sentiment = np.array(list(map(lambda x: list(SentimentIntensityAnalyzer().polarity_scores(x).values()), tqdm(tweets["text"]))))
    tweets["neg"] = sentiment[:, 0]
    tweets["neu"] = sentiment[:, 1]
    tweets["pos"] = sentiment[:, 2]
    tweets["compound"] = sentiment[:, 3]
    tweets["polarity"] = list(map(lambda x: TextBlob(x).sentiment.polarity, tqdm(tweets["text"])))
    tweets["subjectivity"] = list(map(lambda x: TextBlob(x).sentiment.subjectivity, tqdm(tweets["text"])))
    return tweets

    
"""def add_word_embeddings(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
Add the word embeddings scores to the development and evaluation set, based on the vocabolury of the training set.
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
    

    print('Starting word embeddings')
    
    
    X_valid_train, X_valid_test, _, _ = train_test_split(X_train, X_train["sentiment"], test_size=0.2, stratify= X_train["sentiment"], random_state=42)

    text_train = np.array("__label__") + X_valid_train["sentiment"].astype("str").values + np.array(" ") + X_valid_train["text"].values
    text_valid = np.array("__label__") + X_valid_test["sentiment"].astype("str").values + np.array(" ") + X_valid_test["text"].values

    np.savetxt("train.txt", text_train, fmt="%s")
    np.savetxt("valid.txt", text_valid, fmt="%s")

    model = fasttext.train_supervised("train.txt", autotuneValidationFile="valid.txt")

    remove("train.txt")
    remove("valid.txt")



    new_features_names= ["embedding_negativity", "embedding_positivity"]

    #lo so, questa cosa è poco leggibile, la migliorerò
    scores_dev = pd.DataFrame([map(lambda x : x[1], sorted(zip(model.predict(text, k=2)[0],model.predict(text, k=2)[1]), key=lambda x : x[0])) for text in X_train["text"]], columns= new_features_names)
    scores_eval = pd.DataFrame([map(lambda x : x[1], sorted(zip(model.predict(text, k=2)[0],model.predict(text, k=2)[1]), key=lambda x : x[0])) for text in X_test["text"]], columns= new_features_names)
    
    #pd.concat
    X_train = pd.DataFrame(np.column_stack([X_train, scores_dev]), columns=X_train.columns.append(scores_dev.columns))
    X_test = pd.DataFrame(np.column_stack([X_test, scores_eval]), columns=X_test.columns.append(scores_eval.columns))

    return X_train, X_test"""


"""def convert_categorical(tweets: pd.DataFrame) -> pd.DataFrame:
    Converts the features that are not categorical
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets
    Returns
    -------
    pd.DataFrame
        the same dataframe with the function applied
    
   # tweets.drop(columns=["text"], inplace=True) #when the schema will be fixed, this will be moved in the last text processing step

    day_of_week_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    months_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    
    tweets["day_of_week"] = list(map(lambda x: day_of_week_dict[x], tweets["day_of_week"]))
    tweets["month_of_year"] = list(map(lambda x: months_dict[x], tweets["month_of_year"]))


    return tweets"""

    
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


def drop_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):
    columns_to_drop = ["text", "user", "flag", "date", "time"]

    X_train.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)

    return X_train,X_test


def save_results(y_pred: list, file_path: str) -> None:
    pd.Series(y_pred, name="Predicted").to_csv(file_path, index_label="Id")
    print(f"Resulted saved in {file_path}")