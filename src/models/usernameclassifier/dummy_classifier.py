import pandas as pd
from sklearn.dummy import DummyClassifier
# from sklearn import
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

SEED = 2021
BASE_DIR = '/shared/0/projects/reddit-political-affiliation/data/username-labels/'

train_data_source = BASE_DIR + 'user2label.flair.train.csv'
dev_data_source = BASE_DIR + 'user2label.flair.dev.csv'

print("Reading in data sources")
train_df = pd.read_csv(train_data_source)
dev_df = pd.read_csv(dev_data_source)

print("Training set length: {}".format(len(train_df)))
print("Dev set length: {}".format(len(dev_df)))

print("Running logistic regression model")
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 6), lowercase=False)
clf = LogisticRegression(max_iter=10000)
pipe = make_pipeline(vectorizer, clf)
pipe.fit(train_df.username, train_df.binary_label)

dev_preds = pipe.predict(dev_df.username)
print("F1 score for logistic regression: {}".format(f1_score(dev_df.binary_label, dev_preds)))

print("Running dummy classifier")
dummy = DummyClassifier()
dummy.fit(train_df.username, train_df.binary_label)
dummy_dev = dummy.predict(dev_df.username)
print("F1 score for dummy clf: {}".format(f1_score(dev_df.binary_label, dummy_dev)))
