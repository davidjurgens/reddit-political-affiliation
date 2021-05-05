import sys
from glob import glob
from collections import Counter, defaultdict

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.word2vec.build_bipartite_network import filter_down_by_source
from src.data.word2vec.subreddit_political_labels import subreddit_politics

"""
    Script to build variations of the bipartite network. Example use cases
        - Only Flair users
        - Assign politics based on users posting in political subreddits
        - Change the min posts needed
        - Change the number of subreddits used
"""

OUT_DIR = "/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/"


def save_network_w_subreddit_political_labels(network_df, out_file):
    """ Assign political labels based on what subreddits users post in """
    user_politics = dict()
    user_sub_counts = defaultdict(Counter)

    print("Counting subreddit frequencies by user")
    for row in network_df.itertuples():
        user_sub_counts[row.username][row.subreddit] += 1

    print("Computing user politics based on subreddit visits")
    for user, sub_counts in user_sub_counts.items():
        for sub, count in sub_counts.most_common(n=25):
            subreddit = 'r/' + sub
            if subreddit in subreddit_politics:
                politics = subreddit_politics[subreddit]
                user_politics[user] = get_political_label(politics)
                continue

    print("Looping through the network and assigning the new political labels")
    for i, row in enumerate(network_df.itertuples()):
        if row.username in user_politics:
            network_df.at[i, 'politics'] = user_politics[row.username]

    out = OUT_DIR + 'subreddit_labels/' + out_file
    print("Saving network with subreddit labels to: {}".format(out))
    network_df.to_csv(out, sep='\t', index=False)


def get_political_label(politics):
    if politics.lower() == "democrat":
        return 0
    elif politics.lower() == "republican":
        return 1

    return -1


def combine_multiple_months(month_files):
    frames = []
    for m_file in month_files:
        df = pd.read_csv(m_file, sep='\t', index_col=False)
        frames.append(df)

    return pd.concat(frames)


def save_flair_users_only(network_df, out_file):
    print("Generating flair only network for file: {}".format(out_file))
    filtered_df = filter_down_by_source(network_df, sources=['flair'])
    print("Starting network size: {}. Filtered flair network size: {}".format(len(network_df), len(filtered_df)))
    filtered_df.to_csv(OUT_DIR + 'flair/' + out_file, sep='\t', index=False)


def save_gold_users_only(network_df, out_file):
    print("Generating gold only network for file: {}".format(out_file))
    filtered_df = filter_down_by_source(network_df, sources=['gold'])
    print("Starting network size: {}. Filtered gold network size: {}".format(len(network_df), len(filtered_df)))
    filtered_df.to_csv(OUT_DIR + 'gold/' + out_file, sep='\t', index=False)


def save_silver_users_only(network_df, out_file):
    print("Generating silver only network for file: {}".format(out_file))
    filtered_df = filter_down_by_source(network_df, sources=['silver'])
    print("Starting network size: {}. Filtered silver network size: {}".format(len(network_df), len(filtered_df)))
    filtered_df.to_csv(OUT_DIR + 'silver/' + out_file, sep='\t', index=False)


if __name__ == '__main__':
    network_files = glob(OUT_DIR + 'general/*_1000subs_10posts.tsv')
    print(len(network_files))

    for network_file in network_files:
        out_file = network_file.rsplit('/', 1)[-1]
        print("Working on: {}".format(out_file))
        network_df = pd.read_csv(network_file, sep='\t', index_col=False)
        save_network_w_subreddit_political_labels(network_df, out_file)
        save_flair_users_only(network_df, out_file)
        save_gold_users_only(network_df, out_file)
        save_silver_users_only(network_df, out_file)
