from os import device_encoding, remove
import numpy as np
import pandas as pd
import regex as re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def add_word_embeddings(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    text = np.array("__label__") + y_train.astype("str").values + np.array(" ") + X_train["text"].values
    np.savetxt("tweets.txt", text, fmt="%s")
    model = fasttext.train_supervised("tweets.txt")
    remove("tweets.txt")

    #lo so, questa cosa è poco leggibile, la migliorerò
    scores_dev = pd.DataFrame([map(lambda x : x[1], sorted(zip(model.predict(text, k=2)[0],model.predict(text, k=2)[1]), key=lambda x : x[0])) for text in X_train["text"]], columns=["embedding_negativity", "embedding_positivity"])
    scores_eval = pd.DataFrame([map(lambda x : x[1], sorted(zip(model.predict(text, k=2)[0],model.predict(text, k=2)[1]), key=lambda x : x[0])) for text in X_test["text"]], columns=["embedding_negativity", "embedding_positivity"])

    X_train.drop(columns=["user", "text"], inplace=True) #For the moment
    X_test.drop(columns=["user", "text"], inplace=True)

    X_train = pd.DataFrame(np.column_stack([X_train, scores_dev]), columns=X_train.columns.append(scores_dev.columns))
    X_test = pd.DataFrame(np.column_stack([X_test, scores_eval]), columns=X_test.columns.append(scores_eval.columns))

    scaler = MinMaxScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test


def load(filepath="./DSL2122_january_dataset/development.csv") -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except FileNotFoundError:
        print('File not found')

def cleaning(tweets: pd.DataFrame) -> pd.DataFrame:
    new_cols_date = ["day_of_week", "month_of_year", "day_of_month", "time"]

    tweets[new_cols_date] = tweets['date'].str.split(' ', expand=True)[[0,1,2,3]]
    tweets["hour_of_day"] = tweets['time'].str.split(':', expand=True)[0].astype(int)

    # Drop columns that are no important 
    cols_to_drop = ["date", "time", "flag"]
    tweets.drop(columns= cols_to_drop, inplace=True)

    # Drop duplicates
    tweets.drop_duplicates(subset=['text', 'sentiment'], keep='first', inplace=True)
    tweets.drop_duplicates(subset=['text'], inplace=True)
    
    tweets["night"] = (tweets["hour_of_day"].astype("int") >= 18) | (tweets["hour_of_day"].astype("int") <= 5) # to verify with an histogram optimal times

    return tweets.sample(1000)

def hashtag_tfidf(tweets: pd.DataFrame) -> pd.DataFrame:
    tweets["hashtags"] = list(map(lambda t : " ".join(re.findall("#[\d\w]+", t)), tweets["text"]))
    lemmaTokenizer = LemmaTokenizer()                                                                      
    vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words('english'), strip_accents="ascii")
    tfidf = vectorizer.fit_transform(tweets["hashtags"])
    hashtags_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    tweets = pd.DataFrame(np.column_stack([tweets,hashtags_tfidf]), columns=tweets.columns.append(hashtags_tfidf.columns)) #stack the two dataframes horizontally

    return tweets

def text_mining_tfdf(tweets: pd.DataFrame, min_df=0.01) -> pd.DataFrame: #must work on this function to improve perfomance
    #tweets["text"] = list(map(lambda x: re.sub("(@[\d\w]+)|(https?://[\w\d]+)|((www\.)[\d\w]+)|", "", x), tweets["text"])) #text cleaning (urls and users @)
    lemmaTokenizer = LemmaTokenizer()                                                                      
    vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words('english'), strip_accents="ascii", use_idf=False, min_df=min_df)
    tfdf = vectorizer.fit_transform(tweets["text"])
    tweets_text_tfdf = pd.DataFrame(tfdf.toarray(), columns=vectorizer.get_feature_names())
    tweets = pd.DataFrame(np.column_stack([tweets,tweets_text_tfdf]), columns=tweets.columns.append(tweets_text_tfdf.columns)) #stack the two dataframes horizontally
    return tweets

def text_mining_sentiment(tweets: pd.DataFrame) -> pd.DataFrame:
    #tweets["text"] = list(map(lambda x: re.sub("(@[\d\w]+)|(https?://[\w\d]+)|((www\.)[\d\w]+)|", "", x), tweets["text"])) #text cleaning (urls and users @)

    sentiment = np.array(list(map(lambda x: list(SentimentIntensityAnalyzer().polarity_scores(x).values()), tqdm(tweets["text"]))))
    tweets["neg"] = sentiment[:, 0]
    tweets["neu"] = sentiment[:, 1]
    tweets["pos"] = sentiment[:, 2]
    tweets["compound"] = sentiment[:, 3]
    tweets["polarity"] = list(map(lambda x: TextBlob(x).sentiment.polarity, tqdm(tweets["text"])))
    tweets["subjectivity"] = list(map(lambda x: TextBlob(x).sentiment.subjectivity, tqdm(tweets["text"])))
    return tweets

def preprocessing(tweets: pd.DataFrame) -> tuple([pd.DataFrame, pd.Series]):
    day_of_week_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    months_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    
    tweets["day_of_week"] = list(map(lambda x: day_of_week_dict[x], tweets["day_of_week"]))
    tweets["month_of_year"] = list(map(lambda x: months_dict[x], tweets["month_of_year"]))
    tweets["day_of_month"] = list(map(lambda x: int(x), tweets["day_of_month"]))
    tweets["hour_of_day"] = list(map(lambda x: int(x), tweets["hour_of_day"]))

    y = tweets.pop("sentiment").astype('int')
    X = tweets

    return X, y

def replace_pattern(tweets: pd.DataFrame) -> pd.DataFrame:
    
    # regex pattern
    hashtags = "#[\d\w]+"
    mentioned = "@[\d\w]+"
    ampersand = "&[\d\w]+"
    urls = '((www.[^s]+)|(https?://[^s]+))'

    pat = hashtags + "|" + mentioned +  "|" + ampersand + "|" + urls
    
    repl = " "
    
    tweets['text'] = tweets['text'].str.replace(pat= pat, repl = repl, regex=True)

    return tweets