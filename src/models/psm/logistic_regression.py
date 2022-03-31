import pickle
import sys

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

TEST_SPLIT = 0.2
random_state=42
drop_source=False

def train_combined_model(df_train):
    df_train = set_is_either_flip(df_train)
    #print (df_train['is_flip'].value_counts())
    train_0,test_0=train_test_split(df_train[df_train['is_flip']==0], test_size=TEST_SPLIT,random_state=random_state)
    train_1, test_1 = train_test_split(df_train[df_train['is_flip'] == 1], test_size=TEST_SPLIT,random_state=random_state)
    train=pd.concat([train_0,train_1])
    test=pd.concat([test_0,test_1])
    #train, test = train_test_split(df_train, test_size=TEST_SPLIT,random_state=random_state)
    train, test = train.drop(['dem_flip', 'rep_flip'], axis=1), test.drop(['dem_flip', 'rep_flip'], axis=1)
    train.reset_index(drop=True).to_csv('train.tsv',sep='\t')
    test.reset_index(drop=True).to_csv('test.tsv', sep='\t')
    print(train,test)

    y, X = train['is_flip'], train.drop('is_flip', axis=1)
    clf = LogisticRegression(random_state=random_state).fit(X, y)

    y_labels, X_test = test['is_flip'], test.drop('is_flip', axis=1)

    y_0=y_labels[X_test['source']==0]
    X_0=X_test[X_test['source']==0]

    y_1 = y_labels[X_test['source'] == 1]
    X_1 = X_test[X_test['source'] == 1]

    y_2 = y_labels[X_test['source'] == 2]
    X_2 = X_test[X_test['source'] == 2]

    print("---------source 0-----------")
    y_preds_0 = clf.predict(X_0)
    print("RESULTS FOR COMBINED MODEL")
    print(classification_report(y_0, y_preds_0))
    print(classification_report(y_0, y_preds_0, output_dict=True))
    print(accuracy_score(y_0, y_preds_0))

    print("---------source 1-----------")
    y_preds_1 = clf.predict(X_1)
    print("RESULTS FOR COMBINED MODEL")
    print(classification_report(y_1, y_preds_1))
    print(classification_report(y_1, y_preds_1, output_dict=True))
    print(accuracy_score(y_1, y_preds_1))

    print("---------source 2-----------")
    y_preds_2 = clf.predict(X_2)
    print("RESULTS FOR COMBINED MODEL")
    print(classification_report(y_2, y_preds_2))
    print(classification_report(y_2, y_preds_2, output_dict=True))
    print(accuracy_score(y_2, y_preds_2))

    y_preds = clf.predict(X_test)
    y_preds=np.random.randint(2, size=len(y_preds))
    #y_preds=[1]*len(y_preds)
    print("RESULTS FOR COMBINED MODEL")
    print(classification_report(y_labels, y_preds))
    print(classification_report(y_labels, y_preds, output_dict=True))
    print(accuracy_score(y_labels, y_preds))

    filename = ('drop_source' if drop_source else '') + 'combined_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # TODO: Save predictions

    coef_dict = {}
    for coef, feat in zip(clf.coef_[0, :], X.columns):
        coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    # print(coef_dict)

    return clf


def train_rep_to_dem_model(df_train):
    train, test = train_test_split(df_train, test_size=TEST_SPLIT,random_state=random_state)
    train, test = train.drop('dem_flip', axis=1), test.drop('dem_flip', axis=1)

    y, X = train['rep_flip'], train.drop('rep_flip', axis=1)
    clf = LogisticRegression(random_state=random_state).fit(X, y)

    y_labels, X_test = test['rep_flip'], test.drop('rep_flip', axis=1)
    y_preds = clf.predict(X_test)
    #y_preds=[0]*len(y_preds)
    y_preds = np.random.randint(2, size=len(y_preds))

    print("RESULTS FOR REP TO DEM MODEL")
    print(classification_report(y_labels, y_preds))
    print(classification_report(y_labels, y_preds, output_dict=True))
    print(accuracy_score(y_labels, y_preds))

    filename = ('drop_source' if drop_source else '') + 'rep2dem_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # TODO: Save predictions

    coef_dict = {}
    for coef, feat in zip(clf.coef_[0, :], X.columns):
        coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    # print(coef_dict)

    return clf


def train_dem_to_rep_model(df_train):
    train, test = train_test_split(df_train, test_size=TEST_SPLIT,random_state=random_state)
    train, test = train.drop('rep_flip', axis=1), test.drop('rep_flip', axis=1)
    print(len(train), len(test))

    y, X = train['dem_flip'], train.drop('dem_flip', axis=1)
    clf = LogisticRegression(random_state=random_state).fit(X, y)


    y_labels, X_test = test['dem_flip'], test.drop('dem_flip', axis=1)
    y_preds = clf.predict(X_test)
    y_preds = np.random.randint(2, size=len(y_preds))
    #y_preds = [0] * len(y_preds)

    print("RESULTS FOR DEM TO REP MODEL")
    print(classification_report(y_labels, y_preds))
    print(classification_report(y_labels, y_preds, output_dict=True))
    print(accuracy_score(y_labels, y_preds))

    filename = ('drop_source' if drop_source else '') + 'dem2rep_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # TODO: Save predictions

    coef_dict = {}
    for coef, feat in zip(clf.coef_[0, :], X.columns):
        coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    # print(coef_dict)

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
    df = pd.read_csv("/shared/0/projects/reddit-political-affiliation/data/user-features/training_df.tsv", sep='\t',index_col=0)
    # Convert source from flair/community/gold into 0, 1, 2
    df['source'] = df['source'].astype('category').cat.codes
    #print (df.columns)

    if drop_source:
        df.drop(columns=['source'],inplace=True)
    #print (df.head())
    train_combined_model(df)
    train_rep_to_dem_model(df)
    train_dem_to_rep_model(df)
