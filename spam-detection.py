import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from wordcloud import WordCloud
from future.utils import iteritems
from builtins import range

# read the data from spam.csv with this encoding to fit
data_frame = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# remove unnecessary columns
data_frame = data_frame.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# change the column names
data_frame.columns = ["label", "message"]

data_frame['b_label'] = data_frame['label'].map({'ham': 0, 'spam': 1})
Y = data_frame['b_label'].values

data_frame_train, data_frame_test, Ytrain, Ytest = train_test_split(data_frame['message'], Y, test_size=0.3)

def visualize(label):
    words = ''
    for message in data_frame[data_frame['label'] == label]['message']:
        message = message.lower()
        words += message + ' '
    wordcloud = WordCloud(width=800, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize('spam')

visualize('ham')


