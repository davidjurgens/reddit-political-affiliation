import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

INPUT_BASE_PATH = "/shared/0/projects/reddit-political-affiliation/data/username-classifier/"
NON_POLITICAL_USERS_RATIO = 5


def get_non_political_users(n):
    """ Read in non-political usernames from sampled data """
    non_pol_users_file = INPUT_BASE_PATH + "non_political_usernames.tsv"
    non_pol_users = set()
    count = 0
    print("Reading in {} non political users".format(n))
    with open(non_pol_users_file, 'r') as f:
        for line in f:
            non_pol_users.add(line.strip())
            count += 1
            if count >= n:
                return non_pol_users

    return non_pol_users


def build_training_df(source):
    """ Source should be: flair, gold, or silver """
    print("Building training dataframe for source: {}".format(source))
    df = pd.read_csv(get_training_file_name(source), sep='\t', index_col=False)

    usernames, politics = df['username'].tolist(), convert_politics_to_binary(df['politics'].tolist())
    non_political_users = get_non_political_users(n=len(df) * NON_POLITICAL_USERS_RATIO)
    non_political_users_politics = [0] * len(non_political_users)

    print("Total political usernames in training data: {}".format(len(usernames)))
    print("Total non-political usernames in training data: {}".format(len(non_political_users)))

    usernames.extend(non_political_users)
    politics.extend(non_political_users_politics)

    train_df = pd.DataFrame()
    train_df['username'] = usernames
    train_df['politics'] = politics
    train_df = shuffle(train_df)
    return train_df


def get_test_df(source):
    """ Source should be: flair, gold, or silver """
    print("Reading in test dataframe for source: {}".format(source))
    return pd.read_csv(get_test_file_name(source), sep='\t', index_col=False)


def train_username_classifier(train_df):
    print("Training username classifier")
    X = np.array(train_df['username']).reshape(-1, 1)
    y = train_df['politics']
    # TODO: Put the username classifier HERE
    return


def run_clf_on_test(clf, source):
    print("Running classifier on test data for source: {}".format(source))
    test_df = get_test_df(source)
    X_test = np.array(test_df['username']).reshape(-1, 1)
    y_labels = test_df['politics']
    # TODO: Update this to run the predictions on the test data
    y_preds = clf.predict(X_test)
    print(classification_report(y_labels, y_preds))
    print(accuracy_score(y_labels, y_preds))


def get_training_file_name(source):
    if source == 'flair':
        return INPUT_BASE_PATH + "flair/train.tsv"
    elif source == 'gold':
        return INPUT_BASE_PATH + "gold/train.tsv"
    elif source == 'silver':
        return INPUT_BASE_PATH + "gold/train.tsv"
    else:
        raise AssertionError("Invalid source: " + source)


def get_test_file_name(source):
    if source == 'flair':
        return INPUT_BASE_PATH + "flair/test.tsv"
    elif source == 'gold':
        return INPUT_BASE_PATH + "gold/test.tsv"
    elif source == 'silver':
        return INPUT_BASE_PATH + "gold/test.tsv"
    else:
        raise AssertionError("Invalid source: " + source)


def convert_politics_to_binary(all_politics):
    values = []
    for politics in all_politics:
        if politics == 'Democrat':
            values.append(0)
        elif politics == "Republican":
            values.append(1)
        else:
            raise AssertionError("Invalid politics:" + politics)
    return values


if __name__ == '__main__':
    train = build_training_df("flair")
    lr_clf = train_username_classifier(train)
    run_clf_on_test(lr_clf, "flair")
