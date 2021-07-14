import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from emoji import UNICODE_EMO, EMOTICONS
from tqdm import tqdm
from collections import Counter

pd.options.display.max_colwidth = 200

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
    data = data.str.replace(r'([|])', ' ', regex=True)
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
    # Convert posts to lowercase
    data = [text.lower() for text in data]
    return data


df.posts = preprocess_data(df.posts)

# Remove special words (16 personality type abbreviations), because they are in posts, hence could bias the decision
# pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ',
#               'ISTJ', 'ENFJ', 'ISFJ', 'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
pers_types = np.unique(np.array(df['type']))
sub = '|'.join(r"\b{}\b".format(x.lower()) for x in pers_types)
df['posts'] = df['posts'].str.replace(sub, '')

# Split words. Retrieve all words of each post separated by coma. This function is necessary for many operations,
# e.g. for applying Removing stopwords operation
df['posts'] = [text.split() for text in df['posts']]
# Strip/align text if there are any problem with this
df['posts'] = [[word.strip() for word in text] for text in df['posts']]
# Removing stopwords
stop_words = set(stopwords.words('english'))
df['posts'] = [[word for word in text if word not in stop_words] for text in df['posts']]

# Finding the most common words in all posts.
words = df['posts'].apply(lambda x: x)
words = [x for y in words for x in y]
print(Counter(words).most_common(20))

print(df.head(20))
