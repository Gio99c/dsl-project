from io import TextIOWrapper
from os import device_encoding, remove
from nltk.corpus.reader import util
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
import fasttext
from emot.emo_unicode import EMOTICONS_EMO
import html
import tldextract
from utils import CONTRACTIONS, SLANGS, STOPWORDS



"""class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, document):
        lemmas = []
        for t in word_tokenize(document):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)
            lemmas.append(lemma)
        return lemmas"""

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


def expand_contraction_form(text:str) -> str:
    # Replace the contractions from a string to their equivalent by using CONTRCTIONS dic in utils
    for word in text.split(sep= " "):
        if word in CONTRACTIONS.keys():
            
            text = re.sub(word , CONTRACTIONS[word], text)
        
    return text


def convert_slangs(text:str) -> str:
    # Replace the slangs in text by using the SLANGS dict
    for word in text.split(sep= " "):
        if word in SLANGS.keys():
            text = re.sub(word , SLANGS[word], text)
    return text


def word_lemmatizer(text: str) -> str:
    lem_text = [WordNetLemmatizer().lemmatize(i.strip()) for i in text]
    return lem_text


def convert_emoticons(text:str) -> str:
    # Replaxe the emoticons in text such as ':)' by using the EMOTICONS_EMO dict
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), EMOTICONS_EMO[emot], text)
    return text


def load_dataset(filepath="./DSL2122_january_dataset/development.csv") -> pd.DataFrame:
    # Load dataset
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except FileNotFoundError:
        print('File not found')


def extract_date_features(tweets: pd.DataFrame) -> pd.DataFrame:
    """Extract features from date column and convert them to categorical 
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
    # drop the copies tweet of the same user, keep only the first
    tweets.drop_duplicates(subset=['text', "sentiment", "user"], inplace=True, keep="first") 
    # drop all the tweets with the same text but diffent sentiment
    tweets = tweets[~(tweets.duplicated(subset="text") & ~tweets.duplicated(subset=["text", "sentiment"]))]

    return tweets


def clean_text(tweets: pd.DataFrame, deep_clean=False) -> pd.DataFrame:
    """Clean the text feature from unrelevant information and convert the emoticons into text
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
    tweets["text"] = tweets['text'].apply(lambda x: expand_contraction_form(x))
    # Convert HTML entities into characters
    tweets["text"] = tweets["text"].apply(lambda x : html.unescape(x))
    # Convert the text in lower case
    tweets["text"] = tweets["text"].str.lower()

    # extract the domain from the urls
    #urls = "(www\.)|(https?:\/\/)|(\.((com)|(ly)|(it)|(to)|(fm)|(co)|(me)|(gov)|(net)|(org)|(uk)|(im)|(gd)|(cc))[\/\w\d\-\~_\.]*)"
    tweets["text"] = tweets['text'].str.replace(pat="((https?:\/\/)?([w]+\.)?\S+)", repl=lambda x: tldextract.extract(x.group(1)).domain, regex=True)

    # removes duplicated letters
    tweets["text"] = tweets["text"].apply(lambda x: re.sub("(\w)\\1{2,}", "\\1\\1", x))

    # remove numbers
    tweets["text"] = tweets["text"].apply(lambda elem: re.sub(r"\d+", "", elem))

    # convert emoticons into text
    tweets["text"] = tweets['text'].apply(lambda x: convert_emoticons(x))
    #tweets["text"] = tweets['text'].apply(lambda x: expand_contraction_form(x))
    tweets["text"] = tweets['text'].apply(lambda x: convert_slangs(x))



    
    # remove hashtags and mentioned users
    #tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"(@|#[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "", x))  
    tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"(@[\w\d]+)|#", "", x)) 
    tweets["text"] = tweets["text"].apply(lambda x: re.sub("(\w)\\1{2,}", "\\1\\1", x))  
    # remove numbers
    tweets["text"] = tweets["text"].apply(lambda elem: re.sub(r"\d+", "", elem))
    #tweets["text"] = tweets["text"].apply(lambda x: re.sub(r"(@|#[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "", x))    

    # Lemmatize text
    tweets["text"] = tweets["text"].apply(lambda x: " ".join(word_lemmatizer(x.split())))
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
    
    vectorizer = TfidfVectorizer(strip_accents="ascii", stop_words = STOPWORDS, use_idf=False, min_df=min_df)
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
    
    
    
    text_train = np.array("__label__") + X_train["sentiment"].astype("str").values + np.array(" ") + X_train["text"].values
    
    np.savetxt("train.txt", text_train, fmt="%s")
        
    
    model = fasttext.train_supervised("train.txt", epoch = 3, wordNgrams=3, bucket =  1245274, dim = 118, 
                                     lr =0.14218312633399402, lrUpdateRate = 100, maxn = 6, minCount = 1, minCountLabel = 0 ,minn = 3, neg = 5,
                                   seed = 0, t= 0.0001, thread = 15, verbose = 2, ws =5)
    remove("train.txt")
    
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


def drop_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple([pd.DataFrame, pd.DataFrame]):

    # Drop useless columns 
    columns_to_drop = ["text", "user", "flag", "date", "time"]

    X_train.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)

    return X_train,X_test