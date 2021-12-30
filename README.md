# Twitter Sentiment Analysis Classification

## Libraries


```python
import numpy as np
import pandas as pd
import regex as re
```

## Dataset Load


```python
!curl "https://dbdmg.polito.it/dbdmg_web/wp-content/uploads/2021/12/DSL2122_january_dataset.zip" -Lo dataset.zip
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 17.7M  100 17.7M    0     0  16.0M      0  0:00:01  0:00:01 --:--:-- 16.0M



```python
!unzip -q dataset.zip; rm dataset.zip; rm -r __MACOSX/
```


```python
tweets = pd.read_csv("./DSL2122_january_dataset/development.csv")
tweets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>ids</th>
      <th>date</th>
      <th>flag</th>
      <th>user</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1833972543</td>
      <td>Mon May 18 01:08:27 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>Killandra</td>
      <td>@MissBianca76 Yes, talking helps a lot.. going...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1980318193</td>
      <td>Sun May 31 06:23:17 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>IMlisacowan</td>
      <td>SUNSHINE. livingg itttt. imma lie on the grass...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1994409198</td>
      <td>Mon Jun 01 11:52:54 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>yaseminx3</td>
      <td>@PleaseBeMine Something for your iphone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1824749377</td>
      <td>Sun May 17 02:45:34 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>no_surprises</td>
      <td>@GabrielSaporta couldn't get in to the after p...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2001199113</td>
      <td>Tue Jun 02 00:08:07 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>Rhi_ShortStack</td>
      <td>@bradiewebbstack awww is andy being mean again...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>224989</th>
      <td>0</td>
      <td>2261324310</td>
      <td>Sat Jun 20 20:36:48 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>CynthiaBuroughs</td>
      <td>@Dropsofreign yeah I hope Iran people reach fr...</td>
    </tr>
    <tr>
      <th>224990</th>
      <td>1</td>
      <td>1989408152</td>
      <td>Mon Jun 01 01:25:45 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>unitechy</td>
      <td>Trying the qwerty keypad</td>
    </tr>
    <tr>
      <th>224991</th>
      <td>0</td>
      <td>1991221316</td>
      <td>Mon Jun 01 06:38:10 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>Xaan</td>
      <td>I love Jasper &amp;amp; Jackson but that wig in th...</td>
    </tr>
    <tr>
      <th>224992</th>
      <td>0</td>
      <td>2239702807</td>
      <td>Fri Jun 19 08:51:56 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>Ginger_Billie</td>
      <td>I am really tired and bored and bleh! I feel c...</td>
    </tr>
    <tr>
      <th>224993</th>
      <td>1</td>
      <td>2016018811</td>
      <td>Wed Jun 03 06:00:29 PDT 2009</td>
      <td>NO_QUERY</td>
      <td>kevin_choo</td>
      <td>@alyshatan good luck!! It sounds interesting t...</td>
    </tr>
  </tbody>
</table>
<p>224994 rows × 6 columns</p>
</div>



## Data preprocessing


```python
tweets.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 224994 entries, 0 to 224993
    Data columns (total 6 columns):
     #   Column     Non-Null Count   Dtype 
    ---  ------     --------------   ----- 
     0   sentiment  224994 non-null  int64 
     1   ids        224994 non-null  int64 
     2   date       224994 non-null  object
     3   flag       224994 non-null  object
     4   user       224994 non-null  object
     5   text       224994 non-null  object
    dtypes: int64(2), object(4)
    memory usage: 10.3+ MB



```python
tweets["date"]
```




    0         Mon May 18 01:08:27 PDT 2009
    1         Sun May 31 06:23:17 PDT 2009
    2         Mon Jun 01 11:52:54 PDT 2009
    3         Sun May 17 02:45:34 PDT 2009
    4         Tue Jun 02 00:08:07 PDT 2009
                          ...             
    224989    Sat Jun 20 20:36:48 PDT 2009
    224990    Mon Jun 01 01:25:45 PDT 2009
    224991    Mon Jun 01 06:38:10 PDT 2009
    224992    Fri Jun 19 08:51:56 PDT 2009
    224993    Wed Jun 03 06:00:29 PDT 2009
    Name: date, Length: 224994, dtype: object



The _date_ feature contains several different information, these are retrived with the following lines of code


```python
tweets[["day_of_week", "month", "day", "time", "tz", "year"]] = tweets['date'].str.split(' ', expand=True)
```


```python
tweets[["hour", "minute", "second"]] = tweets['time'].str.split(':', expand=True)
```

At this point, the information whose information have been extracted, can be removed


```python
tweets.drop(columns=["date", "time"], inplace=True)
```


```python
tweets["flag"].unique()
```




    array(['NO_QUERY'], dtype=object)




```python
tweets["tz"].unique()
```




    array(['PDT'], dtype=object)




```python
tweets["year"].unique()
```




    array(['2009'], dtype=object)



Since the dataset containes only dates in Pacific Daylight Time (PDT) format and only for the year 2009, these features are not relevant and can be dropped.
The flag feature does not contain any useful info and minutes and seconds do not convey any information so they can be removed as well.


```python
tweets.drop(columns=["tz", "year", "minute", "second", "flag"], inplace=True)
```


```python
tweets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>ids</th>
      <th>user</th>
      <th>text</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1833972543</td>
      <td>Killandra</td>
      <td>@MissBianca76 Yes, talking helps a lot.. going...</td>
      <td>Mon</td>
      <td>May</td>
      <td>18</td>
      <td>01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1980318193</td>
      <td>IMlisacowan</td>
      <td>SUNSHINE. livingg itttt. imma lie on the grass...</td>
      <td>Sun</td>
      <td>May</td>
      <td>31</td>
      <td>06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1994409198</td>
      <td>yaseminx3</td>
      <td>@PleaseBeMine Something for your iphone</td>
      <td>Mon</td>
      <td>Jun</td>
      <td>01</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1824749377</td>
      <td>no_surprises</td>
      <td>@GabrielSaporta couldn't get in to the after p...</td>
      <td>Sun</td>
      <td>May</td>
      <td>17</td>
      <td>02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2001199113</td>
      <td>Rhi_ShortStack</td>
      <td>@bradiewebbstack awww is andy being mean again...</td>
      <td>Tue</td>
      <td>Jun</td>
      <td>02</td>
      <td>00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>224989</th>
      <td>0</td>
      <td>2261324310</td>
      <td>CynthiaBuroughs</td>
      <td>@Dropsofreign yeah I hope Iran people reach fr...</td>
      <td>Sat</td>
      <td>Jun</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>224990</th>
      <td>1</td>
      <td>1989408152</td>
      <td>unitechy</td>
      <td>Trying the qwerty keypad</td>
      <td>Mon</td>
      <td>Jun</td>
      <td>01</td>
      <td>01</td>
    </tr>
    <tr>
      <th>224991</th>
      <td>0</td>
      <td>1991221316</td>
      <td>Xaan</td>
      <td>I love Jasper &amp;amp; Jackson but that wig in th...</td>
      <td>Mon</td>
      <td>Jun</td>
      <td>01</td>
      <td>06</td>
    </tr>
    <tr>
      <th>224992</th>
      <td>0</td>
      <td>2239702807</td>
      <td>Ginger_Billie</td>
      <td>I am really tired and bored and bleh! I feel c...</td>
      <td>Fri</td>
      <td>Jun</td>
      <td>19</td>
      <td>08</td>
    </tr>
    <tr>
      <th>224993</th>
      <td>1</td>
      <td>2016018811</td>
      <td>kevin_choo</td>
      <td>@alyshatan good luck!! It sounds interesting t...</td>
      <td>Wed</td>
      <td>Jun</td>
      <td>03</td>
      <td>06</td>
    </tr>
  </tbody>
</table>
<p>224994 rows × 8 columns</p>
</div>



Instead of taking into account the specific hour, I decided that it is better to characterize the record by specifing if they were written in night hours (from 18 to 5) or in daylight hourse (from 6 to 17).

_night_ is a boolean feature


```python
tweets["night"] = (tweets["hour"].astype("int") >= 18) | (tweets["hour"].astype("int") <= 5)
```


```python
#tweets.drop(columns=["hour"], inplace=True) // Choose to remove it or not
```


```python
tweets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>ids</th>
      <th>user</th>
      <th>text</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>night</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1833972543</td>
      <td>Killandra</td>
      <td>@MissBianca76 Yes, talking helps a lot.. going...</td>
      <td>Mon</td>
      <td>May</td>
      <td>18</td>
      <td>01</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1980318193</td>
      <td>IMlisacowan</td>
      <td>SUNSHINE. livingg itttt. imma lie on the grass...</td>
      <td>Sun</td>
      <td>May</td>
      <td>31</td>
      <td>06</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1994409198</td>
      <td>yaseminx3</td>
      <td>@PleaseBeMine Something for your iphone</td>
      <td>Mon</td>
      <td>Jun</td>
      <td>01</td>
      <td>11</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1824749377</td>
      <td>no_surprises</td>
      <td>@GabrielSaporta couldn't get in to the after p...</td>
      <td>Sun</td>
      <td>May</td>
      <td>17</td>
      <td>02</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2001199113</td>
      <td>Rhi_ShortStack</td>
      <td>@bradiewebbstack awww is andy being mean again...</td>
      <td>Tue</td>
      <td>Jun</td>
      <td>02</td>
      <td>00</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>224989</th>
      <td>0</td>
      <td>2261324310</td>
      <td>CynthiaBuroughs</td>
      <td>@Dropsofreign yeah I hope Iran people reach fr...</td>
      <td>Sat</td>
      <td>Jun</td>
      <td>20</td>
      <td>20</td>
      <td>True</td>
    </tr>
    <tr>
      <th>224990</th>
      <td>1</td>
      <td>1989408152</td>
      <td>unitechy</td>
      <td>Trying the qwerty keypad</td>
      <td>Mon</td>
      <td>Jun</td>
      <td>01</td>
      <td>01</td>
      <td>True</td>
    </tr>
    <tr>
      <th>224991</th>
      <td>0</td>
      <td>1991221316</td>
      <td>Xaan</td>
      <td>I love Jasper &amp;amp; Jackson but that wig in th...</td>
      <td>Mon</td>
      <td>Jun</td>
      <td>01</td>
      <td>06</td>
      <td>False</td>
    </tr>
    <tr>
      <th>224992</th>
      <td>0</td>
      <td>2239702807</td>
      <td>Ginger_Billie</td>
      <td>I am really tired and bored and bleh! I feel c...</td>
      <td>Fri</td>
      <td>Jun</td>
      <td>19</td>
      <td>08</td>
      <td>False</td>
    </tr>
    <tr>
      <th>224993</th>
      <td>1</td>
      <td>2016018811</td>
      <td>kevin_choo</td>
      <td>@alyshatan good luck!! It sounds interesting t...</td>
      <td>Wed</td>
      <td>Jun</td>
      <td>03</td>
      <td>06</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>224994 rows × 9 columns</p>
</div>



### Hashtag and mentioned user extraction

---
Ho dubbi che questa cosa possa essere effettivamente efficace. Come facciamo a utilizzare queste informazioni
quando ci sono più utenti menzionati e quando ci sono diversi hashtag?
Per gli hashtag, si potrebbe fare qualcosa (tipo tf-idf o tf-df), mentre per gli utenti proprio non saprei
---


```python
tweets["hashtags"] = list(map(lambda t : re.findall("#[\d\w]+", t), tweets["text"]))
tweets["mentioned"] = list(map(lambda t : re.findall("@[\d\w]+", t), tweets["text"]))
```


```python
tweets["mentioned"]
```




    0            [@MissBianca76]
    1                         []
    2            [@PleaseBeMine]
    3          [@GabrielSaporta]
    4         [@bradiewebbstack]
                     ...        
    224989       [@Dropsofreign]
    224990                    []
    224991                    []
    224992                    []
    224993          [@alyshatan]
    Name: mentioned, Length: 224994, dtype: object




```python
tweets["hashtags"]
```




    0         []
    1         []
    2         []
    3         []
    4         []
              ..
    224989    []
    224990    []
    224991    []
    224992    []
    224993    []
    Name: hashtags, Length: 224994, dtype: object



## Text mining


```python
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

lemmaTokenizer = LemmaTokenizer()                                                                      
vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=sw.words('english'), strip_accents="unicode")
tfidf = vectorizer.fit_transform(tweets["text"])
```

    /Users/gio/opt/anaconda3/lib/python3.8/site-packages/sklearn/feature_extraction/text.py:388: UserWarning: Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would'] not in stop_words.
      warnings.warn('Your stop_words may be inconsistent with '



```python
pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>!</th>
      <th>#</th>
      <th>$</th>
      <th>%</th>
      <th>&amp;</th>
      <th>'</th>
      <th>''</th>
      <th>''and</th>
      <th>''la</th>
      <th>''photos</th>
      <th>...</th>
      <th>øaø3ø£u</th>
      <th>øoø</th>
      <th>ø£øμuø§</th>
      <th>ø§uuu</th>
      <th>ø§uø3ø1uø</th>
      <th>ø§uø§øaøμø§uø§øa</th>
      <th>ø§ø</th>
      <th>ø­uø</th>
      <th>μa</th>
      <th>μa1a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.250272</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>224989</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>224990</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>224991</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.159766</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>224992</th>
      <td>0.127275</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>224993</th>
      <td>0.229561</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>224994 rows × 164475 columns</p>
</div>




```python

```
