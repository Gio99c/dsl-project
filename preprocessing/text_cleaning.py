import pandas as pd
import numpy as np

# Text libraries 
import regex as re
import html
import tldextract
from nltk.stem.wordnet import WordNetLemmatizer

# Import dictionaries 
from emot.emo_unicode import EMOTICONS_EMO
from preprocessing.utils import SLANGS, CONTRACTIONS



def add_user_text(tweets: pd.DataFrame) -> pd.DataFrame:
    # Add the username as the first word (with the @) in the text feature
    tweets["text"] = np.array("@") + tweets["user"].values + np.array(" ") + tweets["text"].values

    return tweets
 

def expand_contraction_form(text:str) -> str:
    # Replace contractions forms with a string that has the equivalent meaning by using CONTRACTIONS dic in utils
    for word in text.split(sep= " "):
        if word in CONTRACTIONS.keys():
            text = re.sub(word , CONTRACTIONS[word], text)
        
    return text


def convert_slangs(text:str) -> str:
    # Replace slangs in the text by using the SLANGS dict
    for word in text.split(sep= " "):
        if word in SLANGS.keys():
            text = re.sub(word , SLANGS[word], text)
    return text


def convert_emoticons(text:str) -> str:
    # Replace emoticons in the text such as ':)' by using the EMOTICONS_EMO dict
    for emot in EMOTICONS_EMO:
        text = re.sub(re.escape(emot), EMOTICONS_EMO[emot], text)
    return text


def word_lemmatizer(text: str) -> str:
    # Lemmatize the text
    lem_text = [WordNetLemmatizer().lemmatize(i.strip()) for i in text]
    return lem_text


def clean_text(tweets: pd.DataFrame) -> pd.DataFrame:
    """Clean the text format and remove irrelevant informations
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


def count_characters(tweets: pd.DataFrame, trainset:bool) -> pd.DataFrame:
    # Extract the numb of chars in the text and drop too long tweets
    tweets['char_count'] = list(map(lambda x: len(html.unescape(x)), tweets['text']))
    if trainset:
        tweets = tweets[tweets['char_count'] <= 140]

    return tweets