import gzip
import json
import random
import sys
from collections import defaultdict
from glob import glob

import numpy as np
from scipy.stats import entropy

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.behavioral_features.descriptive_metrics_by_cohort import get_random_sample_of_each_data_source

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/behavior/"
random.seed(2021)


def grab_subreddit_counts_for_users(users, source_name):
    """ Grab subreddit submission counts for a subset of users """
    month_files = glob(DATA_DIRECTORY + "*.gz")
    user_subreddit_counts = defaultdict(lambda: defaultdict(int))

    for month_file in month_files:
        print("Working on {} users on file: {}".format(source_name, month_file))
        with gzip.open(month_file, 'r') as f:
            for line in f:
                user_counts = json.loads(line.strip())
                user = user_counts['user']

                if user in users:
                    # Update their existing counts
                    for sub, count in user_counts['counts'].items():
                        sub = 'r/' + sub
                        user_subreddit_counts[user][sub] += count

    save_total_counts_for_users(user_subreddit_counts, source_name)
    return user_subreddit_counts


def save_total_counts_for_users(user_subreddit_counts, source_name):
    out_file = OUTPUT_DIRECTORY + source_name + '_sample_counts.tsv'
    print("Saving conglomerate counts for {} to {}".format(source_name, out_file))

    with open(out_file, 'w') as f:
        for user, sub_counts in user_subreddit_counts.items():
            for sub, count in sub_counts.items():
                f.write("{}\t{}\t{}\n".format(user, sub, count))


def read_total_counts_for_users(source_name):
    user_subreddit_counts = defaultdict(lambda: defaultdict(int))
    in_file = OUTPUT_DIRECTORY + source_name + '.tsv'

    with open(in_file, 'r') as f:
        for line in f:
            user, sub, count = line.split('\t')
            user_subreddit_counts[user][sub] = count

    return user_subreddit_counts


def compute_cohort_entropy(user_subreddit_counts):
    user_entropy = {}

    for user, sub_counts in user_subreddit_counts.items():
        subreddit_counts = list(sub_counts.values())
        result = entropy(subreddit_counts)
        user_entropy[user] = result

    return user_entropy


def summarize_and_plot(user_entropy):
    entropy_values = np.array(list(user_entropy.values()))
    avg_entropy = np.mean(entropy_values)
    median_entropy = np.median(entropy_values)

    print("Average entropy for cohort: {}".format(avg_entropy))
    print("Median entropy for cohort: {}".format(median_entropy))


def randomly_sample_users(users, n):
    return set(random.sample(users, n))


def run(users, source_name):
    print("Total # of {} users: {}".format(source_name, len(users)))
    subreddit_counts = grab_subreddit_counts_for_users(users, source_name)
    user_entropy = compute_cohort_entropy(subreddit_counts)
    summarize_and_plot(user_entropy)


if __name__ == '__main__':
    users_by_source = get_random_sample_of_each_data_source(n=10000)

    for source, users in users_by_source.items():
        run(users, source)
