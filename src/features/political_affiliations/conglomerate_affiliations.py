import glob
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics
from src.features.political_affiliations.flair_political_affiliations import read_in_flair_affiliations

''' 
    Read in all of the political affiliations (silver, gold, flair) and create train dev and test sets with the joined
    data. Hardcoded directory paths because they are static 
'''


def grab_all_data_sources():
    silver_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/*.tsv")
    gold_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv")
    flair_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/*.tsv")

    print("Number of silver files: {}".format(silver_files))
    print("Number of gold files: {}".format(gold_files))
    print("Number of flair files: {}".format(flair_files))

    silver_data = read_in_user_politics(silver_files)
    gold_data = read_in_user_politics(gold_files)
    flair_data = read_in_flair_affiliations(flair_files)

    return silver_data, gold_data, flair_data


def build_df(silver, gold, flair):
    user_entries = defaultdict(list)

    print("Working on silver data")
    for user, user_politics in silver.items():
        for entry in user_politics:
            row = {'username': user, 'source': 'silver', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            if not value_already_exists(user_entries, row):
                user_entries[user].append(row)

    print("Working on gold data")
    for user, user_politics in gold.items():
        for entry in user_politics:
            row = {'username': user, 'source': 'gold', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            if not value_already_exists(user_entries, row):
                user_entries[user].append(row)

    print("Working on flair data")
    for user, flair_data in flair.items():
        for entry in flair_data:
            row = {'username': user, 'source': 'flair', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            if not value_already_exists(user_entries, row):
                user_entries[user].append(row)

    # All user entries are next to each other in the dataframe so we can partition without having
    # the same user in multiple datasets
    user_entries = shuffle_dict_keys(user_entries)
    rows = []
    for user, entries in user_entries.items():
        for entry in entries:
            rows.append(entry)

    return pd.DataFrame(rows)


def shuffle_dict_keys(user_entries):
    print("Shuffling usernames")
    users = list(user_entries.keys())
    random.shuffle(users)
    shuffled_dict = {}
    for user in users:
        shuffled_dict[user] = user_entries[user]
    return shuffled_dict


def value_already_exists(user_entries, row):
    for entry in user_entries[row['username']]:
        if row['politics'] == entry['politics'] and row['source'] == entry['source']:
            return True

    return False


def validate_no_overlap(train_df, dev_df, test_df):
    print("Checking overlap between train and dev users")
    overlap_1 = set(train_df['username'].tolist()).intersection(set(dev_df['username'].tolist()))
    print("Length of the overlap: {}".format(len(overlap_1)))

    print("Checking overlap between dev and test users")
    overlap_2 = set(dev_df['username'].tolist()).intersection(set(test_df['username'].tolist()))
    print("Length of the overlap: {}".format(len(overlap_2)))


def split_into_train_dev_test(df, train_size=0.8):
    split_index = int(len(df) * train_size)
    train = df[:split_index]
    # Remaining data is test and dev. Divided into equal parts
    test_dev = df[split_index + 1:]
    half_split_index = int(len(test_dev) * .5)
    test = test_dev[:half_split_index]
    dev = test_dev[half_split_index:]
    return train, test, dev


def get_train_political_affiliations():
    file_path = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/train.tsv"
    user_politics = defaultdict(list)
    print("Reading in user politics from file: {}".format(file_path))
    train_df = pd.read_csv(file_path, index_col=False, delimiter='\t')

    for index, row in train_df.iterrows():
        entry = {'politics': row['politics'], 'source': row['source'], 'subreddit': row['subreddit'],
                 'created': row['created']}
        user_politics[row['username']].append(entry)

    return user_politics


if __name__ == '__main__':
    out_directory = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/"
    silver, gold, flair = grab_all_data_sources()
    df = build_df(silver, gold, flair)
    print(df.head(10))
    train, dev, test = split_into_train_dev_test(df)

    print("Length of train data: {}".format(len(train)))
    print("Length of dev data: {}".format(len(dev)))
    print("Length of test data: {}".format(len(test)))

    validate_no_overlap(train, dev, test)

    train.to_csv(out_directory + "train.tsv", sep='\t')
    dev.to_csv(out_directory + "dev.tsv", sep='\t')
    test.to_csv(out_directory + "test.tsv", sep='\t')
