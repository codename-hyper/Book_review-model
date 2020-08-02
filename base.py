import json
import random

file_name = './data/sentiment/Books_small_10000.json'


# using enums
class Sentiment():
    positive = 'POSITIVE'
    negative = 'NEGATIVE'


# class creation
class Review():
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score >= 3.0:
            return Sentiment.positive
        else:
            return Sentiment.negative


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.negative, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.positive, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)


reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append((Review(review['reviewText'], review['overall'])))

# importing needed feature from sci-kit learn
from sklearn.model_selection import train_test_split

# splitting the data for training and testing the model
training, test = train_test_split(reviews, test_size=0.33, random_state=42)

# evenly distribute
train_container = ReviewContainer(training)

test_container = ReviewContainer(test)

# splitting the data into 'what is input' and 'what is output' (x is input,y is output)
# train_x = [x.text for x in training]
# train_y = [y.sentiment for y in training]
#
# test_x = [x.text for x in training]
# test_y = [y.sentiment for y in training]

train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

# bags of words vectorization

# importing needed feature from sci-kit learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()

# converting review to matrix
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

# <editor-fold desc="classification">
# Linear SVM
from sklearn import svm

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors, train_y)

clf_svm.predict(test_x_vectors[0])

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

clf_dec.predict(test_x_vectors[0])

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vectors.toarray(), train_y)

clf_gnb.predict(test_x_vectors[0].toarray())

# Logistic Regression
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression(max_iter=10000)
clf_log.fit(train_x_vectors, train_y)

clf_log.predict(test_x_vectors[0])
# </editor-fold>

# Evaluation
# Mean Accuracy
# print(clf_svm.score(test_x_vectors, test_y))
# print(clf_dec.score(test_x_vectors, test_y))
# print(clf_gnb.score(test_x_vectors, test_y))
# print(clf_log.score(test_x_vectors, test_y))

# F1 Scores
from sklearn.metrics import f1_score

# print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.positive, Sentiment.negative]))
# print(f1_score(test_y, clf_gnb.predict(test_x_vectors), average=None, labels=[Sentiment.positive, Sentiment.negative]))
# print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.positive, Sentiment.negative]))
# print(f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.positive, Sentiment.negative]))

# Testing
test_set = ['very fun', "bad book do not buy", 'horrible waste of time']
new_test = vectorizer.transform(test_set)

# print(clf_svm.predict(new_test))

# using grid search to tune model
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 4, 8, 16, 32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

# import pickle to save models
import pickle

# saving
# with open('sentiment_classifier.pkl', 'wb') as f:
#     pickle.dump(clf, f)

# loading
# with open('sentiment_classifier.pkl', 'rb') as f:
#     loaded_clf = pickle.load(f)
