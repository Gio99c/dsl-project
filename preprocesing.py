# Standard libraries
import numpy as np
import pandas as pd

# Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#NLTK
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# FastText
import fasttext
from os import remove

# Clean text
from emot.emo_unicode import EMOTICONS_EMO
import regex as re
import html
import tldextract
from utils import CONTRACTIONS, SLANGS, STOPWORDS

from tqdm import tqdm




# START TEXT CLEANING FUNCTIONS


def expand_contraction_form(text:str) -> str:
    # Replace the contractions from a string to their equivalent by using CONTRACTIONS dic in utils
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


def convert_emoticons(text:str) -> str:
    # Replaxe the emoticons in text such as ':)' by using the EMOTICONS_EMO dict
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), EMOTICONS_EMO[emot], text)
    return text


def word_lemmatizer(text: str) -> str:
    # Lemmatize the text
    lem_text = [WordNetLemmatizer().lemmatize(i.strip()) for i in text]
    return lem_text


def clean_text(tweets: pd.DataFrame) -> pd.DataFrame:
    """Clean the text feature from irrelevant information and convert the emoticons into text
    Parameters
    ----------
    tweets : pd.DataFrame
        Dataframe of tweets

    Returns
    -------
    pd.DataFrame
        Same DataFrame with text cleaned 
    """
    
    # Convert text in lower case
    tweets["text"] = tweets["text"].str.lower()

    #  Expand contracted form
    tweets["text"] = list(map(lambda x: expand_contraction_form(x),tweets['text'] ))

    # Convert HTML entities into characters
    tweets["text"] = list(map(lambda x: html.unescape(x), tweets['text'])) 

    # Extract the domain from the URLs
    tweets["text"] = tweets['text'].str.replace(pat="((https?:\/\/)?([w]+\.)?\S+)", repl=lambda x: tldextract.extract(x.group(1)).domain, regex=True)

    # Remove repeated characters
    tweets["text"] = list(map(lambda x: re.sub("(\w)\\1{2,}", "\\1\\1", x), tweets['text']))

    # Remove numbers 
    tweets["text"] = list(map(lambda elem: re.sub(r"\d+", "", elem), tweets['text']))

    # Replace emoticons into text
    tweets["text"] = list(map(lambda x: convert_emoticons(x), tweets["text"]))

    # Replace slangs
    tweets["text"] = list(map(lambda x: convert_slangs(x), tweets["text"]))
    
    # Lemmatize text
    tweets["text"] = list(map(lambda x: " ".join(word_lemmatizer(x.split())), tweets["text"]))

    return tweets


# END TEXT CLEANING FUNCTIONS




# START WORD EMBEDDINGS FUNCTIONS


def add_user_text(tweets: pd.DataFrame) -> pd.DataFrame:
    # Add the username as the first word (with the @) in the text feature
    tweets["text"] = np.array("@") + tweets["user"].values + np.array(" ") + tweets["text"].values

    return tweets
    

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
    X_train : pd.DataFrame
        Devolopment set dataframe
    X_test : pd.DataFrame
        Evaluation set dataframe
    Returns
    -------
    tuple([pd.DataFrame, pd.DataFrame, pd.Series])
        a tuple with X_train, X_test and y_train
    """

    print('Starting word embeddings')
    
    text_train = np.array("__label__") + X_valid_train["sentiment"].astype("str").values + np.array(" ") + X_valid_train["text"].values
    
    np.savetxt("train.txt", text_train, fmt="%s")   
    
    model = fasttext.train_supervised("train.txt", epoch = 4, wordNgrams=2, bucket =  966320, dim = 126, 
                                     lr =0.20158291921508406, lrUpdateRate = 100, maxn = 6, minCount = 1, minCountLabel = 0 ,minn = 3, neg = 5,
                                   seed = 0, t= 0.0001, thread = 15, verbose = 2, ws =5)
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


# END WORD EMBEDDINGS FUNCTIONS




# START NEW FEATURES EXTRACTION FUNCTIONS


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


# END NEW FEATURES EXTRACTION FUNCTIONS




# START TF-DF


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


# END TF-DF




# START SENTIMENT ANALYSIS


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

    sentiment = np.array(list(map(lambda x: list(SentimentIntensityAnalyzer().polarity_scores(x).values()), tqdm(tweets["text"]))))

    tweets["neg"] = sentiment[:, 0]
    tweets["neu"] = sentiment[:, 1]
    tweets["pos"] = sentiment[:, 2]
    tweets["compound"] = sentiment[:, 3]
    tweets["polarity"] = list(map(lambda x: TextBlob(x).sentiment.polarity, tqdm(tweets["text"])))
    tweets["subjectivity"] = list(map(lambda x: TextBlob(x).sentiment.subjectivity, tqdm(tweets["text"])))

    return tweets


# END SENTIMENT ANALYSIS




# START UTILS FUNCTIONS  


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


def drop_duplicates(tweets: pd.DataFrame) -> pd.DataFrame:
    # Drop the copies tweet of the same user, keep only the first
    tweets.drop_duplicates(subset=['text', "sentiment", "user"], inplace=True, keep="first") 

    # Drop all the tweets with the same text but diffent sentiment
    tweets = tweets[~(tweets.duplicated(subset="text") & ~tweets.duplicated(subset=["text", "sentiment"]))]

    return tweets


def load_dataset(filepath="./DSL2122_january_dataset/development.csv") -> pd.DataFrame:
    # Load dataset
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except FileNotFoundError:
        print('File not found')


# END UTILS FUNCTIONS  