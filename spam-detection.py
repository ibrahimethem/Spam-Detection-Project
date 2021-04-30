import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from wordcloud import WordCloud

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

data_frame_train, data_frame_test, Ytrain, Ytest = train_test_split(data_frame['message'], Y, test_size=0.3)

# Transform the data with TF-IDF Vectorizer for featuring vectors to use as input to estimator.
# It also removes the stop words.
tfidf = TfidfVectorizer(decode_error='ignore')
Xtrain = tfidf.fit_transform(data_frame_train)
Xtest = tfidf.transform(data_frame_test)


# A function for reporting the wrong predictions which are:
#   spams that are predicted as ham,
#   and hams that are predicted as spams.
# I takes prediction_label for each model
def report_wrong_predictions(prediction_label):
    sneaky_spam = data_frame[(data_frame[prediction_label] == 0) & (data_frame['b_label'] == 1)]['message']
    print('Numbers of spams predicted as ham is ', sneaky_spam.count())
    print('\n')
    for msg in sneaky_spam[:3]:
        print(msg)
    not_actually_spam = data_frame[(data_frame[prediction_label] == 1) & (data_frame['b_label'] == 0)]['message']
    print('\n')
    print('Numbers of ham messages predicted as spam is ', not_actually_spam.count())

    for msg in not_actually_spam[:3]:
        print(msg)
    print('\n')


# Naive Bayes Model
NB_model = MultinomialNB()
NB_model.fit(Xtrain, Ytrain)
print('Naive Bayes Score: \n')
print("train score:", NB_model.score(Xtrain, Ytrain))
print("test score:", NB_model.score(Xtest, Ytest))

X = tfidf.transform(data_frame['message'])
data_frame['prediction_NB'] = NB_model.predict(X)
report_wrong_predictions('prediction_NB')


# AdaBoostClassifier algorithm
ABC_Model = AdaBoostClassifier()
ABC_Model.fit(Xtrain, Ytrain)
print('AdaBoostClassifier Score: \n')
print('train score:', ABC_Model.score(Xtrain,Ytrain))
print('test score:', ABC_Model.score(Xtest, Ytest))

data_frame['prediction_ABC'] = ABC_Model.predict(X)
report_wrong_predictions('prediction_ABC')


# Multi-Layer perception algorithm
mlp_classifier_model = MLPClassifier()
mlp_classifier_model.fit(Xtrain, Ytrain)
print('MLP Classifier Score: \n')
print('train score:', mlp_classifier_model.score(Xtrain,Ytrain))
print('test score:', mlp_classifier_model.score(Xtest, Ytest))

data_frame['prediction_MLP'] = mlp_classifier_model.predict(X)
report_wrong_predictions('prediction_MLP')

