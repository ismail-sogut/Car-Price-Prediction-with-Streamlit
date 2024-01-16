# !pip install nltk
# !pip install textblob
# !pip install wordcloud

import nltk
nltk.download('stopwords')
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textblob
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Görev-1:


# Adım1: Metin ön işleme için clean_text adında fonksiyon oluşturunuz. Fonksiyon;
# • Büyük küçük harf dönüşümü,
# • Noktalama işaretlerini çıkarma,
# • Numerik ifadeleri çıkarma Işlemlerini gerçekleştirmeli.

df = pd.read_csv("5_NLP/DATASETS/wiki_data.csv")
df.head()
df.info()


# "text" değişkeni üzerinden tüm işlemleri yapcaağız
def clean_text(dataframe, col):
    # text_cols = [col for col in dataframe.columns if dataframe.columns.str.contains("text")]

    ###############################
    # Normalizing Case Folding
    ###############################

    dataframe[col] = dataframe[col].str.lower()

    ###############################
    # Punctuations
    ###############################

    dataframe[col] = dataframe[col].str.replace("[^\w\s]", '')

    # regular expression

    ###############################
    # Numbers
    ###############################

    dataframe[col] = dataframe[col].str.replace('\d', '')



# Adım2: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

clean_text(df, "text")
# df["text"] = df["text"].str.replace('\', " ")
df.head()

# Adım3: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleri çıkaracak remove_stopwords adında fonksiyon yazınız.

def remove_stopwords(dataframe, col):
    sw = stopwords.words('english')
    dataframe[col] = dataframe[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


# Adım4: Yazdığınız fonksiyonu veri seti içerisindeki tüm metinlere uygulayınız.

remove_stopwords(df, "text")
df.head()

# Adım5: Metinde az geçen (1000'den az, 2000'den az gibi) kelimeleri bulunuz. Ve bu kelimeleri metin içerisinden çıkartınız.

# rare işlemi yapılacak

temp_df = pd.Series(' '.join(df['text']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Adım6: Metinleri tokenize edip sonuçları gözlemleyiniz.

df["text"].apply(lambda x: TextBlob(x).words).head()

df.head()


# Adım7: Lemmatization işlemi yapınız.

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



# Görev 2: Veriyi Görselleştiriniz

# Adım1: Metindeki terimlerin frekanslarını hesaplayınız.

freq = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

freq.columns = ["words", "freq"]

freq.sort_values("freq", ascending=False)


# Adım2: Bir önceki adımda bulduğunuz terim frekanslarının Barplot grafiğini oluşturunuz.

freq[freq["freq"] > 2500].plot.bar(x="words", y="freq")
plt.show(block=True)



# Adım3: Kelimeleri WordCloud ile görselleştiriniz.


text = " ".join(i for i in df.text)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)
wordcloud.to_file("wordcloud222.png")