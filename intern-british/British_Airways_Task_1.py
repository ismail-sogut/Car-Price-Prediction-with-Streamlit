# TASK_1 : Web Scraping

#pip install beautifulsoup4
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("punkt")
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import textblob
from PIL import Image
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 37
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())

    print(f"   ---> {len(reviews)} total reviews")

df = pd.DataFrame()
df["reviews"] = reviews
df.head(20)


df.to_csv("BA_reviews.csv")

# "✅ Trip Verified"

#### My Study

# BA represents British Airways, and it should not be removed since it is important to analise
df["reviews"] = df["reviews"].str.replace("BA", "British Airways")
df.head()

# Normalizing Case Folding
###############################

df['reviews'] = df['reviews'].str.lower()

# Punctuations
###############################
df['reviews'] = df['reviews'].str.replace("[^\w\s]", '', regex=True)

# Numbers
###############################
df['reviews'] = df['reviews'].str.replace('\d', '', regex=True)

# Stopwords
###############################

sw = stopwords.words('english')

df['reviews'] = df['reviews'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords
###############################

temp_df = pd.Series(' '.join(df['reviews']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['reviews'] = df['reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Tokenization
###############################

df["reviews"].apply(lambda x: TextBlob(x).words).head()

# Lemmatization
###############################
df['reviews'] = df['reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# word frequency
###############################

tf = df["reviews"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)


# Barplot
###############################

tf[tf["tf"] > 100].plot.bar(x="words", y="tf")
plt.show(block=True)

# Wordcloud
###############################

text = " ".join(i for i in df.reviews)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure(block=True)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)


# 3. Sentiment Analysis
##################################################

df["reviews"].head()


sia = SentimentIntensityAnalyzer()

df["reviews"][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviews"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviews"].apply(lambda x: sia.polarity_scores(x)["compound"])

# 4. Feature Engineering
###############################

df["reviews"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["reviews"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviews"]