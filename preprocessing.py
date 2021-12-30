import numpy as np
import pandas as pd
#import regex as re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as sw

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


def load():
    return pd.read_csv("development.csv")

def cleaning(tweets):
    tweets[["day_of_week", "month_of_year", "day_of_month", "time", "tz", "year"]] = tweets['date'].str.split(' ', expand=True)
    tweets[["hour_of_day", "minute", "second"]] = tweets['time'].str.split(':', expand=True)
    tweets.drop(columns=["date", "time"], inplace=True)
    tweets.drop(columns=["tz", "year", "minute", "second", "flag"], inplace=True)
    #tweets["text"] = tweets["text"][tweets.duplicated(subset=["text"], keep="first")] #duplicated removal - work in progress
    tweets["night"] = (tweets["hour_of_day"].astype("int") >= 18) | (tweets["hour_of_day"].astype("int") <= 5) # to verify with an histogram optimal times
    return tweets

def text_mining(tweets): #must work on this function to improve perfomance
    lemmaTokenizer = LemmaTokenizer()                                                                      
    vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words('english'), strip_accents="ascii", use_idf=False, min_df=0.01)
    tfidf = vectorizer.fit_transform(tweets["text"])
    tweets_text_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    tweets = pd.concat((tweets, tweets_text_tfidf), axis=1)
    tweets.drop(columns=["user", "text"], inplace=True) #For the moment
    return tweets

def preprocessing1(tweets):
    day_of_week_dict = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    months_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    
    tweets["day_of_week"] = list(map(lambda x: day_of_week_dict[x], tweets["day_of_week"]))
    tweets["month_of_year"] = list(map(lambda x: months_dict[x], tweets["month_of_year"]))
    tweets["day_of_month"] = list(map(lambda x: int(x), tweets["day_of_month"]))
    tweets["hour_of_day"] = list(map(lambda x: int(x), tweets["hour_of_day"]))
    tweets["ids"] = ColumnTransformer([('somename', MinMaxScaler(), [1])], remainder='passthrough').fit_transform(tweets)

    #y = tweets.pop("sentiment")
    #X = tweets

    return tweets


def create_hours_bins(dataframe: pd.DataFrame) -> pd.DataFrame:

    bin_ranges = [0,2,  4, 8, 12, 16, 20, 24]
    bin_names = ['Notte', 'Notte FOnda', 'Mattina Presto','Tarda Mattinata', 'Primo pomeriggio', ' Tardo Pomeriggio', 'Sera']
    dataframe['hours_bin'] = pd.cut(np.array(dataframe['hour_of_day'].astype("int")), include_lowest = True, bins= bin_ranges, labels = bin_names)
    dataframe = pd.get_dummies(data= dataframe, columns= ['hours_bin'], drop_first=True)

    return dataframe


def create_pubblic_holiday(dataframe: pd.DataFrame) -> pd.DataFrame:

    holydays_dict ={"Mon": False, "Tue": False, "Wed": False, "Thu": False, "Fri": False, "Sat": True, "Sun": True}
    dataframe["pubblic_holyday"] = list(map(lambda x: holydays_dict[x], dataframe["day_of_week"]))
    
    return dataframe



def drop_dupl(dataframe: pd.DataFrame):
    dataframe.drop_duplicates(subset= ['text','sentiment'], keep = 'first', inplace=True)
    dataframe.drop_duplicates(subset=['text'],inplace=True )