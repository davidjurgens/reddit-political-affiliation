import glob
import sys

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions

CONGLOMERATE_DATA_PATH = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/"
OUTPUT_BASE_PATH = "/shared/0/projects/reddit-political-affiliation/data/username-classifier/"


def get_all_political_usernames():
    """ Grab all political users to cross-reference when sampling non-political users """
    print("Getting all political usernames")
    political_users = set()
    train_df = pd.read_csv(CONGLOMERATE_DATA_PATH + "train.tsv", sep='\t', index_col=False)
    dev_df = pd.read_csv(CONGLOMERATE_DATA_PATH + "dev.tsv", sep='\t', index_col=False)
    test_df = pd.read_csv(CONGLOMERATE_DATA_PATH + "test.tsv", sep='\t', index_col=False)

    train_users = set(train_df['username'].tolist())
    dev_users = set(dev_df['username'].tolist())
    test_users = set(test_df['username'].tolist())

    print("Total political users in train: {}".format(len(train_users)))
    print("Total political users in dev: {}".format(len(dev_users)))
    print("Total political users in test: {}".format(len(test_users)))

    political_users.update(train_users)
    political_users.update(dev_users)
    political_users.update(test_users)

    return political_users


def output_training_usernames_by_type():
    """ Separate flair/gold/silver training usernames and save them to a TSV"""
    train_df = pd.read_csv(CONGLOMERATE_DATA_PATH + "train.tsv", sep='\t', index_col=False)
    train_flair = train_df[train_df['source'] == 'flair']
    train_gold = train_df[train_df['source'] == 'gold']
    train_silver = train_df[train_df['source'] == 'silver']

    train_flair.to_csv(OUTPUT_BASE_PATH + "flair/train.tsv", sep='\t')
    train_gold.to_csv(OUTPUT_BASE_PATH + "gold/train.tsv", sep='\t')
    train_silver.to_csv(OUTPUT_BASE_PATH + "silver/train.tsv", sep='\t')


def output_testing_usernames_by_type():
    """ Separate flair/gold/silver training usernames and save them to a TSV"""
    test_df = pd.read_csv(CONGLOMERATE_DATA_PATH + "test.tsv", sep='\t', index_col=False)
    test_flair = test_df[test_df['source'] == 'flair']
    test_gold = test_df[test_df['source'] == 'gold']
    test_silver = test_df[test_df['source'] == 'silver']

    test_flair.to_csv(OUTPUT_BASE_PATH + "flair/test.tsv", sep='\t')
    test_gold.to_csv(OUTPUT_BASE_PATH + "gold/test.tsv", sep='\t')
    test_silver.to_csv(OUTPUT_BASE_PATH + "silver/test.tsv", sep='\t')


def sample_non_political_usernames(in_files, political_users, n):
    # Spread the sampling out evenly over all files
    usernames_per_file = int(n / len(in_files))
    print("Sampling {} random non-political users. A total of {} will be collected from {} files"
          .format(n, usernames_per_file, len(files)))
    non_political_users = set()

    for in_file in in_files:
        file_usernames_collected = 0
        for submission in read_submissions(in_file):
            try:
                username = submission['author']

                if username not in political_users and username not in non_political_users:
                    non_political_users.add(username)
                    file_usernames_collected += 1

                if file_usernames_collected >= usernames_per_file:
                    break
            except Exception:
                continue

    return non_political_users


def output_non_political_usernames_to_tsv(usernames):
    out_file = OUTPUT_BASE_PATH + "non_political_usernames.tsv"
    print("Saving non political usernames to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        for username in usernames:
            f.write("{}\n".format(username))


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    pol_users = get_all_political_usernames()
    non_pol_users = sample_non_political_usernames(files, pol_users, n=500000)
    output_non_political_usernames_to_tsv(non_pol_users)
    output_training_usernames_by_type()
    output_testing_usernames_by_type()
