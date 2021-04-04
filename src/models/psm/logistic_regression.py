import glob
import pandas as pd
import random
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.1


def read_in_features_df(in_files):
    print("Total # of months of data: {}".format(len(in_files)))
    return pd.concat((pd.read_csv(f, sep='\t') for f in in_files))


def train_logistic_clf(df_train):
    print("Training logistic regression model")
    y, X = df_train['is_political'], df_train.drop('is_political', axis=1)
    return LogisticRegression(random_state=0).fit(X, y)


def test_logistic_clf(clf, df_test):
    y_labels, X_test = df_test['is_political'], df_test.drop('is_political', axis=1)
    y_preds = clf.predict(X_test)
    print(classification_report(y_labels, y_preds))
    print(accuracy_score(y_labels, y_preds))


def plot_probability_score_distribution(clf, df_test):
    y_labels, X_test = df_test['is_political'], df_test.drop('is_political', axis=1)
    probabilities = clf.predict_proba(X_test)
    sns.displot(probabilities)
    plt.show()
    return


def get_prob_distribution(in_files):
    df = read_in_features_df(files[:10])
    train, test = train_test_split(df, test_size=TEST_SPLIT)
    print("Total number of users in train: {}. Total number of users in test: {}".format(len(train), len(test)))
    clf = train_logisitic_clf(train)
    y_labels, X_test = df_test['is_political'], df_test.drop('is_political', axis=1)
    probabilities = clf.predict_proba(X_test)
    test_logistic_clf(clf, test)
    plot_probability_score_distribution(clf, test)
    return


if __name__ == '__main__':
    files = glob.glob('/shared/0/projects/reddit-political-affiliation/data/psm/features/gold/*.tsv')
    random.shuffle(files)
    df = read_in_features_df(files[:10])
    train, test = train_test_split(df, test_size=TEST_SPLIT)
    print("Total number of users in train: {}. Total number of users in test: {}".format(len(train), len(test)))
    clf = train_logisitic_clf(train)
    test_logistic_clf(clf, test)
    plot_probability_score_distribution(clf, test)
