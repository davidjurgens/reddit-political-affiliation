import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# SETTINGS
features_file = ''
test_split = 0.2


print("Reading in existing features")
df = pd.read_csv('', sep='\t')
df = shuffle(df)
train, test = train_test_split(df, test_size=test_split)
print("Total number of users in train: {}. Total number of users in test: {}".format(len(train), len(test)))

print("Training logistic regression model")
X, y = train.drop('political'), train['political']
clf = LogisticRegression(random_state=0).fit(X, y)

X_test, y_labels = test.drop('political'), test['political']
y_preds = clf.predict(X_test)

print(classification_report(y_labels, y_preds))
