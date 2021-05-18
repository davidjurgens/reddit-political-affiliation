import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

TEST_SPLIT = 0.1
random_state=42
drop_source=False

def train_combined_model(df_train):
    df_train = set_is_either_flip(df_train)
    train, test = train_test_split(df_train, test_size=0.1,random_state=random_state)
    train, test = train.drop(['dem_flip', 'rep_flip'], axis=1), test.drop(['dem_flip', 'rep_flip'], axis=1)

    y, X = train['is_flip'], train.drop('is_flip', axis=1)
    clf = LogisticRegression(random_state=random_state).fit(X, y)

    y_labels, X_test = test['is_flip'], test.drop('is_flip', axis=1)
    y_preds = clf.predict(X_test)

    print("RESULTS FOR COMBINED MODEL")
    print(classification_report(y_labels, y_preds))
    print(classification_report(y_labels, y_preds,output_dict=True))
    print(accuracy_score(y_labels, y_preds))

    filename = ('drop_source' if drop_source else '')+'combined_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # TODO: Save predictions

    coef_dict = {}
    for coef, feat in zip(clf.coef_[0, :], X.columns):
        coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    #print(coef_dict)

    return clf


def train_rep_to_dem_model(df_train):
    train, test = train_test_split(df_train, test_size=0.1,random_state=random_state)
    train, test = train.drop('dem_flip', axis=1), test.drop('dem_flip', axis=1)

    y, X = train['rep_flip'], train.drop('rep_flip', axis=1)
    clf = LogisticRegression(random_state=random_state).fit(X, y)

    y_labels, X_test = test['rep_flip'], test.drop('rep_flip', axis=1)
    y_preds = clf.predict(X_test)

    print("RESULTS FOR REP TO DEM MODEL")
    print(classification_report(y_labels, y_preds))
    print(classification_report(y_labels, y_preds, output_dict=True))
    print(accuracy_score(y_labels, y_preds))

    filename = ('drop_source' if drop_source else '')+'rep2dem_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # TODO: Save predictions

    coef_dict = {}
    for coef, feat in zip(clf.coef_[0, :], X.columns):
        coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    #print(coef_dict)

    return clf


def train_dem_to_rep_model(df_train):
    train, test = train_test_split(df_train, test_size=0.1,random_state=random_state)

    train, test = train.drop('rep_flip', axis=1), test.drop('rep_flip', axis=1)
    print(len(train), len(test))

    y, X = train['dem_flip'], train.drop('dem_flip', axis=1)
    clf = LogisticRegression(random_state=random_state).fit(X, y)

    y_labels, X_test = test['dem_flip'], test.drop('dem_flip', axis=1)
    y_preds = clf.predict(X_test)


    print("RESULTS FOR DEM TO REP MODEL")
    print(classification_report(y_labels, y_preds))
    print(classification_report(y_labels, y_preds, output_dict=True))
    print(accuracy_score(y_labels, y_preds))

    filename = ('drop_source' if drop_source else '')+'dem2rep_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # TODO: Save predictions

    coef_dict = {}
    for coef, feat in zip(clf.coef_[0, :], X.columns):
        coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    #print(coef_dict)

    return clf


def train_dummy_clf(df_train):
    print("Training logistic regression model")
    y, X = df_train['is_flip'], df_train.drop('is_flip', axis=1)
    return DummyClassifier(strategy="most_frequent").fit(X, y)


def test_logistic_clf(clf, df_test):
    y_labels, X_test = df_test['is_flip'], df_test.drop('is_flip', axis=1)
    y_preds = clf.predict(X_test)
    print(classification_report(y_labels, y_preds))
    print(accuracy_score(y_labels, y_preds))


def set_is_either_flip(df):
    is_flip = []
    for row in df.itertuples():
        if row.rep_flip or row.dem_flip:
            is_flip.append(1)
        else:
            is_flip.append(0)

    df['is_flip'] = is_flip
    return df


if __name__ == '__main__':
    df = pd.read_csv("/shared/0/projects/reddit-political-affiliation/data/user-features/training_df.tsv", sep='\t')
    # Convert source from flair/community/gold into 0, 1, 2
    df['source'] = df['source'].astype('category').cat.codes

    if drop_source:
        df.drop(columns=['source'],inplace=True)
    print (df.head())
    train_combined_model(df)
    train_rep_to_dem_model(df)
    train_dem_to_rep_model(df)
