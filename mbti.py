import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from emoji import UNICODE_EMO, EMOTICONS
from tqdm import tqdm
from collections import Counter
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D, Input, MaxPooling1D, Embedding
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 50

# Create the pandas DataFrame object
df = pd.read_csv("data/mbti_1.csv")

# Converting emojis into their meaning
def convert_emojis(data):
    for emoji, meaning in UNICODE_EMO.items():
        data = data.replace(emoji, meaning)
    return data


def convert_emoticons(data):
    for emoticon, meaning in EMOTICONS.items():
        data = data.replace(emoticon, meaning)
    return data


# Pre-processing
def preprocess_data(data):
    # The way to implement the replacement of "|||" using pandas dataframe method
    data = data.str.replace('|||', ' ', regex=False)
    # The way to replace links ending jpg|jpeg|gif|png with IMAGE
    data = data.str.replace(r'https?://\S+?/\S+?\.(?:jpg|jpeg|gif|png)', 'IMAGE', regex=True)
    # The way to replace the remaining links with URL
    data = data.str.replace(r'https?://[^\s<>"]+|www\.[^\s<>"]+', 'URL', regex=True)
    # Convert emojis
    data = convert_emojis(data)
    data = convert_emoticons(data)
    # Strip Punctuation
    data = data.str.replace(r'[\.+]', ".", regex=True)
    # Remove multiple fullstops
    data = data.str.replace(r'[^\w\s]', '', regex=True)
    # Remove Non-words
    data = data.str.replace(r'[^a-zA-Z\s]', '', regex=True)
    # Replace multiple spaces with one space
    data = data.str.replace(r'\s+', ' ', regex=True)
    # Convert posts to lowercase
    data = data.str.lower()
    return data


df.posts = preprocess_data(df.posts)
# print(df.head(10))

# Remove posts with number of words less than 20
min_words = 20
# print("Before : Number of posts", len(df))
df["total words"] = df["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
df = df[df["total words"] >= min_words]
# print("After : Number of posts", len(df))

# Remove special words (16 personality type abbreviations), because they are in posts, hence could bias the decision
# pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ',
#               'ISTJ', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
pers_types = df['type'].unique()
sub = '|'.join(r"\b{}\b".format(x.lower()) for x in pers_types)
df['posts'] = df['posts'].str.replace(sub, '')

# Encode labels with values between 0 and n_classes-1. If label repeats it assigns the same value to the same label.
df['type of encoding'] = LabelEncoder().fit_transform(df['type'])
target = df['type of encoding']

# Use one-hot encoder and tokenization for Sequential model
ohe = OneHotEncoder(sparse=False)
target_seq = ohe.fit_transform(target.values.reshape(-1, 1))
# Tokenize words
max_nb_words = 200000
tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(df["posts"])
# Creating dictionary of word indexes
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# Retokenize
max_nb_words = len(word_index)
tokenizer = Tokenizer(num_words=max_nb_words)
tokenizer.fit_on_texts(df["posts"])
sequences = tokenizer.texts_to_sequences(df["posts"])
print(sequences[0])
print(len(sequences))

# Constants
input_y_num = 16
max_post_len = np.max([len(x) for x in sequences])

# Pad Sequences
sequences = sequence.pad_sequences(sequences, maxlen=max_post_len)

# Split Train/Test
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(sequences, target_seq, test_size=0.1,
                                                                    stratify=target_seq, random_state=42)

# Bag of Words Model
# Vectorizing(converting posts into numerical form) the posts for the model and filtering Stop-words
train = CountVectorizer(stop_words='english').fit_transform(df["posts"])
# print(train, train.shape)
# print(target)

# Training & Evaluating : 70-30 split
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.15, stratify=target, random_state=42)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Logistic Regression
# fit model to training data
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# # make predictions for test data
# Y_test = logreg.predict(X_test)
# # evaluate predictions
# predictions = [round(value) for value in Y_test]
# accuracy = accuracy_score(y_test, predictions)
# # print the result as float number with 2 digits after the delimiter(%.2f%%)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))  # 49.79%

# XG boost Classifier
# xgb = XGBClassifier(use_label_encoder=False)
# xgb.fit(X_train, y_train)
# Y_test = xgb.predict(X_test)
# # evaluate predictions
# predictions = [round(value) for value in Y_test]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))  # 52.92%

# NaiveBayes
# nb = MultinomialNB()
# nb.fit(X_train, y_train)
# # Y_train = nb.predict(X_train)
# # print("Train Accuracy:", np.mean(Y_train == y_train))
# Y_test = nb.predict(X_test)
# acc = np.mean(Y_test == y_test)
# print("Test Accuracy: %.2f%%" % (acc * 100))  # 32.26%

# Sequential Models


# Split words. Tokenization. Retrieve all words of each post separated by coma.
# This function is necessary for many operations, e.g. for applying Removing stopwords operation
# df['posts'] = [text.split() for text in df['posts']]
# Strip/align text if there are any problem with this
# df['posts'] = [[word.strip() for word in text] for text in df['posts']]
# Removing stopwords
# stop_words = set(stopwords.words('english'))
# df['posts'] = [[word for word in text if word not in stop_words] for text in df['posts']]

# Finding the most common words in all posts.
# words = df['posts'].apply(lambda x: x)
# words = [x for y in words for x in y]
# print(Counter(words).most_common(20))

# TODO: Finish Keras LSTM model. Try transformers

# print(df.head(10))
