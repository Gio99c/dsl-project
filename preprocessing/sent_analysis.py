import pandas as pd
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from tqdm import tqdm


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
