import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# SETTINGS
features_file = '/shared/0/projects/reddit-political-affiliation/data/psm/features/RC_2019-09.tsv'
test_split = 0.1

print("Reading in existing features")
df = pd.read_csv(features_file, sep='\t')
train, test = train_test_split(df, test_size=test_split)
print("Total number of users in train: {}. Total number of users in test: {}".format(len(train), len(test)))

print("Training logistic regression model")
print(train.columns)
y = train['is_political']
X = train.drop('is_political', axis=1)
clf = LogisticRegression(random_state=0).fit(X, y)

y_labels, X_test = test['is_political'], test.drop('is_political', axis=1)
y_preds = clf.predict(X_test)

print(classification_report(y_labels, y_preds))
print(accuracy_score(y_labels, y_preds))
