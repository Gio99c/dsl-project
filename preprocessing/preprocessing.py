import pandas as pd

from sklearn.preprocessing import MinMaxScaler


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


# FEATURES EXTRACTION FUNCTION


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