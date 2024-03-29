import gzip
import json
import random
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.features.behavioral_features.collect_user_features import get_all_user_features

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.data_helper import get_all_raw_files
from src.features.political_affiliations.conglomerate_affiliations import get_train_users, get_train_users_by_politics, \
    get_all_political_users
from src.features.behavioral_features.collect_samples import collect_non_political_usernames

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/behavior/"
random.seed(2021)


def get_random_sample_of_each_data_source(n=10000):
    print("Getting random sample for each data source")

    flair_users = get_train_users('flair')
    flair_users = set(random.sample(flair_users, n))

    gold_users = get_train_users('gold')
    gold_users = set(random.sample(gold_users, n))

    community_users = get_train_users('community')
    community_users = set(random.sample(community_users, n))

    print("Getting random sample of non political users")
    all_political_users = get_all_political_users()
    non_political_usernames = collect_non_political_usernames(get_all_raw_files(), all_political_users, count=n)

    return {'flair': flair_users,
            'gold': gold_users,
            'community': community_users,
            'non-political': non_political_usernames}


def get_random_sample_of_each_data_source_and_politics(n=3000):
    flair_rep_users = get_train_users_by_politics('flair', 'Republican')
    flair_dem_users = get_train_users_by_politics('flair', 'Democrat')
    flair_rep_users = set(random.sample(flair_rep_users, n))
    flair_dem_users = set(random.sample(flair_dem_users, n))

    gold_rep_users = get_train_users_by_politics('gold', 'Republican')
    gold_dem_users = get_train_users_by_politics('gold', 'Democrat')
    gold_rep_users = set(random.sample(gold_rep_users, n))
    gold_dem_users = set(random.sample(gold_dem_users, n))

    community_rep_users = get_train_users_by_politics('community', 'Republican')
    community_dem_users = get_train_users_by_politics('community', 'Democrat')
    community_rep_users = set(random.sample(community_rep_users, n))
    community_dem_users = set(random.sample(community_dem_users, n))

    return {'flair_rep': flair_rep_users, 'flair_dem': flair_dem_users,
            'gold_rep': gold_rep_users, 'gold_dem': gold_dem_users,
            'community_rep_users': community_rep_users, 'community_dem_users': community_dem_users}


def get_total_comment_counts(users, source_name):
    month_files = glob(DATA_DIRECTORY + "*.gz")
    user_comment_count = defaultdict(int)

    for month_file in month_files[:20]:
        print("Working on counting total comments for {} users on file: {}".format(source_name, month_file))
        with gzip.open(month_file, 'r') as f:
            for line in f:
                user_counts = json.loads(line.strip())
                user = user_counts['user']
                if user in users:
                    for sub, count in user_counts['counts'].items():
                        user_comment_count[user] += count

    save_comment_counts(user_comment_count, source_name)
    return user_comment_count


def get_total_karma_and_controversiality_counts(users, source_name):
    print("Grabbing user karma and controversiality for " + source_name)
    all_user_features = get_all_user_features()
    user_features = all_user_features[all_user_features['username'].isin(users)]

    user_karma, user_controversiality = {}, {}

    for row in user_features.itertuples():
        user, karma, controversiality = row.username, row.total_score, row.total_controversiality
        user_karma[user] = int(karma)
        user_controversiality[user] = int(controversiality)

    return user_karma, user_controversiality


def save_comment_counts(user_comment_count, source_name):
    out_file = OUTPUT_DIRECTORY + source_name + '_comment_counts.tsv'
    print("Saving comment counts to file: {}".format(out_file))

    with open(out_file, 'w') as f:
        for user, comment_count in user_comment_count.items():
            f.write("{}\t{}\n".format(user, comment_count))


def load_comment_counts(source_name):
    in_file = OUTPUT_DIRECTORY + source_name + '_comment_counts.tsv'
    print("Loading comment counts from file: {}".format(in_file))

    user_comment_count = {}
    with open(in_file, 'r') as f:
        for line in f:
            user, comment_count = line.split('\t')
            user_comment_count[user] = comment_count

    return user_comment_count


def summarize_comment_counts(user_comment_count, source_name):
    comment_counts = np.array(list(user_comment_count.values()))
    print("Average comment counts for {}: {}".format(source_name, np.mean(comment_counts)))
    print("Median comment counts for {}: {}".format(source_name, np.median(comment_counts)))
    print("STD comment counts for {}: {}".format(source_name, np.std(comment_counts)))


def plot_log_comment_counts(user_comment_counts_by_source):
    # Convert to dataframe to make plotting easier
    rows = []
    for source, user_comment_counts in user_comment_counts_by_source.items():
        for user, comment_count in user_comment_counts.items():
            entry = {'source': source, 'user': user, 'comment_count': comment_count}
            rows.append(entry)
    df = pd.DataFrame(rows)

    ax = sns.displot(df, x="comment_count", hue="source", kind="kde", log_scale=True)
    plt.show()


def run(users, source_name):
    print("Running for: " + source_name)
    total_comment_counts = get_total_comment_counts(users, source_name)
    summarize_comment_counts(total_comment_counts, source_name)
    return total_comment_counts


if __name__ == '__main__':

    users_by_source = get_random_sample_of_each_data_source()
    user_comment_counts_by_source = {}

    for source, users in users_by_source.items():
        comment_counts = run(users, source)
        user_comment_counts_by_source[source] = comment_counts

    plot_log_comment_counts(user_comment_counts_by_source)
