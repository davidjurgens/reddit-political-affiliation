import random
import sys
from collections import defaultdict
from glob import glob
from itertools import islice

import pandas as pd

from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics
from src.features.political_affiliations.flair_political_affiliations import read_in_flair_affiliations
from src.features.political_affiliations.community_labels import get_user_politics_for_community_labels

''' 
    Read in all of the political affiliations (silver, gold, flair) and create train dev and test sets with the joined
    data. Hardcoded directory paths because they are static 
'''

OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/"


def grab_all_data_sources():
    gold_files = glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv")
    flair_files = glob("/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/*.tsv")
    community_files = glob("/shared/0/projects/reddit-political-affiliation/data/community-affiliations/*.tsv")

    print("Number of gold files: {}".format(gold_files))
    print("Number of flair files: {}".format(flair_files))
    print("Number of community files: {}".format(community_files))

    gold_data = read_in_user_politics(gold_files)
    flair_data = read_in_flair_affiliations(flair_files)
    community_data = get_user_politics_for_community_labels()

    return {'gold': gold_data, 'flair': flair_data, 'community': community_data}


def build_df(data_by_source):
    user_entries = defaultdict(list)

    print("Working on gold data")
    for user, user_politics in data_by_source['gold'].items():
        for entry in user_politics:
            row = {'username': user, 'source': 'gold', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            if not value_already_exists(user_entries, row):
                user_entries[user].append(row)

    print("Working on flair data")
    for user, flair_data in data_by_source['flair'].items():
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

    df = pd.DataFrame(rows)
    return filter_bad_actors(df)


def shuffle_dict_keys(user_entries):
    print("Shuffling usernames")
    users = list(user_entries.keys())
    random.seed(42)
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


def read_in_political_affiliations(file_path):
    user_politics = defaultdict(list)
    print("Reading in user politics from file: {}".format(file_path))
    train_df = pd.read_csv(file_path, index_col=False, delimiter='\t')

    for index, row in train_df.iterrows():
        entry = {'politics': row['politics'], 'source': row['source'], 'subreddit': row['subreddit'],
                 'created': row['created']}
        user_politics[row['username']].append(entry)

    return user_politics


def filter_bad_actors(df):
    print("Filtering out bad actors")
    bad_actors = read_in_bad_actors_from_tsv(
        ['/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_365_days_1_flip_flop.tsv'])
    bad_actors = set(bad_actors.keys())
    print("Total number of bad actors: {}".format(len(bad_actors)))
    return df[~df['username'].isin(bad_actors)]


def get_political_affiliations(source):
    assert source == 'train' or source == 'dev' or source == 'test'
    return read_in_political_affiliations(
        "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/{}.tsv".format(source))


def get_df(name):
    assert name == 'train' or name == 'dev' or name == 'test'
    return pd.read_csv(OUTPUT_DIRECTORY + '/{}.tsv'.format(name), sep='\t', index_col=False)


def get_train_users(source):
    assert source == 'flair' or source == 'community' or source == 'gold'
    train_df = get_df('train')
    source_df = train_df[train_df['source'] == source]
    return set(source_df['username'].tolist())


def get_train_users_by_politics(source, politics):
    assert source == 'flair' or source == 'community' or source == 'gold'
    assert politics == "Democrat" or politics == "Republican"
    train_df = get_df('train')
    source_df = train_df[train_df['source'] == source]
    source_politics_df = source_df[source_df['politics'] == politics]
    return set(source_politics_df['username'].tolist())


def get_all_political_users():
    print("Grabbing all political users")
    train_df = get_df('train')
    dev_df = get_df('dev')
    test_df = get_df('test')

    train_users = set(train_df['username'].tolist())
    dev_users = set(dev_df['username'].tolist())
    test_users = set(test_df['username'].tolist())

    return train_users.union(dev_users).union(test_users)


def add_in_community_labels():
    """ Adding in community politics without changing the existing train/dev/test users """
    # Grab the community political affiliations and split them randomly 80/10/10
    community_politics = get_user_politics_for_community_labels()

    community_politics = shuffle_dict_keys(community_politics)
    train_length = int(len(community_politics) * 0.8)
    dev_length, test_length = int(len(community_politics) * .1), int(len(community_politics) * .1)

    train_users = dict(islice(community_politics.items(), train_length))
    dev_users = dict(islice(community_politics.items(), train_length + 1, train_length + dev_length))
    test_users = dict(
        islice(community_politics.items(), train_length + dev_length + 1, train_length + dev_length + test_length))

    print(len(train_users), len(dev_users), len(test_users))

    # Read in train/dev/test
    train_df = pd.read_csv(OUTPUT_DIRECTORY + 'train.tsv', sep='\t', index_col=False)
    dev_df = pd.read_csv(OUTPUT_DIRECTORY + 'dev.tsv', sep='\t', index_col=False)
    test_df = pd.read_csv(OUTPUT_DIRECTORY + 'test.tsv', sep='\t', index_col=False)

    add_users_into_existing_df_and_save(train_df, train_users, 'train.tsv')
    add_users_into_existing_df_and_save(dev_df, dev_users, 'dev.tsv')
    add_users_into_existing_df_and_save(test_df, test_users, 'test.tsv')


def add_users_into_existing_df_and_save(df, user_politics, fname):
    rows = []
    for user, politics in user_politics.items():
        entry = {'username': user, 'politics': politics, 'source': 'community', 'subreddit': '', 'created': ''}
        rows.append(entry)

    new_df = pd.DataFrame(rows)
    result = pd.concat([df, new_df], ignore_index=True)
    print(len(df), len(result), fname)
    result.to_csv(OUTPUT_DIRECTORY + fname, sep='\t', index=False)


if __name__ == '__main__':
    df = build_df(grab_all_data_sources())
    print(df.head(10))
    train, dev, test = split_into_train_dev_test(df)

    print("Length of train data: {}".format(len(train)))
    print("Length of dev data: {}".format(len(dev)))
    print("Length of test data: {}".format(len(test)))

    validate_no_overlap(train, dev, test)

    train.to_csv(OUTPUT_DIRECTORY + "train.tsv", sep='\t')
    dev.to_csv(OUTPUT_DIRECTORY + "dev.tsv", sep='\t')
    test.to_csv(OUTPUT_DIRECTORY + "test.tsv", sep='\t')
