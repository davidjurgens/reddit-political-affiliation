import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# SETTINGS
features_file = '/shared/0/projects/reddit-political-affiliation/data/psm/RC_2019-09.tsv'
test_split = 0.2

print("Reading in existing features")
df = pd.read_csv(features_file, sep='\t')

df = shuffle(df)
train, test = train_test_split(df, test_size=test_split)
print("Total number of users in train: {}. Total number of users in test: {}".format(len(train), len(test)))

print("Training logistic regression model")
print(train.columns)
y = train['political']
X = train.drop('political', axis=1)
clf = LogisticRegression(random_state=0).fit(X, y)

y_labels, X_test = test['political'], test.drop('political', axis=1)
y_preds = clf.predict(X_test)

print(classification_report(y_labels, y_preds))
