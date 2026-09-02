import numpy as np
import pandas as pd
import kagglehub
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import os

# Download latest version
path = kagglehub.dataset_download("algord/fake-news")
file_path=os.path.join(path, 'FakeNewsNet.csv')
print("Path to dataset files:", path)

# nltk.download('stopwords')
# print("Stopwords downloaded:", stopwords.words('english'))

#//////////////
news_dataset=pd.read_csv(file_path)
print(news_dataset.isnull().sum())
# replaing thenul values with empty strings
news_dataset=news_dataset.fillna('')
#merging the source domain name with the newstitle 
news_dataset['content']=news_dataset['source_domain']+ ': '+news_dataset['title']

# print(news_dataset.head())

X=news_dataset.drop(columns='real',axis=1)
Y=news_dataset['real']

#checking is any row has the vlaue of real as 0
# result=news_dataset.loc[news_dataset['real']==0, 'title']
# print(result)

port_stem=PorterStemmer()


#stemming involes reduing a word to its root word e.g Actor,actress,ating -> act
def stemming(content):
    stemmed_content= re.sub('[^a-zA-Z]',' ',content)
    stemmed_content= stemmed_content.lower()
    stemmed_content= stemmed_content.split()
    #removing stop words
    stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content= ' '.join(stemmed_content)
    return stemmed_content

# print(X.head())

news_dataset['content']=news_dataset['content'].apply(stemming)

print(news_dataset['content'].values)

































































































# print