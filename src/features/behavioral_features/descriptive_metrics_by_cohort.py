import gzip
import json
import random
import sys
from collections import defaultdict
from glob import glob

import numpy as np

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.conglomerate_affiliations import get_train_df

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/behavior/"
random.seed(2021)


def get_random_sample_of_each_data_source(n=10000):
    print("Getting random sample for each data source")
    df = get_train_df()

    flair_df = df[df['source'] == 'flair']
    gold_df = df[df['source'] == 'gold']
    silver_df = df[df['source'] == 'silver']
    community_df = df[df['source'] == 'community']

    flair_users = set(flair_df['username'].tolist())
    flair_users = set(random.sample(flair_users, n))

    gold_users = set(gold_df['username'].tolist())
    gold_users = set(random.sample(gold_users, n))

    silver_users = set(silver_df['username'].tolist())
    silver_users = set(random.sample(silver_users, n))

    community_users = set(community_df['username'].tolist())
    community_users = set(random.sample(community_users, n))

    return {'flair': flair_users, 'gold': gold_users, 'silver': silver_users, 'community': community_users}


def get_total_comment_counts(users, source_name):
    month_files = glob(DATA_DIRECTORY + "*.gz")
    user_comment_count = defaultdict(int)

    for month_file in month_files[:12]:
        print("Working on counting total comments for {} users on file: {}".format(source_name, month_file))
        with gzip.open(month_file, 'r') as f:
            for line in f:
                user_counts = json.loads(line.strip())
                user = user_counts['user']
                if user in users:
                    for sub, count in user_counts['counts'].items():
                        user_comment_count[user] += count

    return user_comment_count


def summarize_comment_counts(user_comment_count, source_name):
    comment_counts = np.array(list(user_comment_count.values()))
    print("Average comment counts for {}: {}".format(source_name, np.mean(comment_counts)))
    print("Median comment counts for {}: {}".format(source_name, np.median(comment_counts)))
    print("STD comment counts for {}: {}".format(source_name, np.std(comment_counts)))


def run(users, source_name):
    print("Running for: " + source_name)
    summarize_comment_counts(get_total_comment_counts(users, source_name), source_name)


if __name__ == '__main__':
    users_by_source = get_random_sample_of_each_data_source()

    for source, users in users_by_source.items():
        run(users, source)
