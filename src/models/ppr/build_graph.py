import glob
import pickle
from collections import *
from collections import Counter

import networkx as nx
from tqdm.notebook import tqdm


def filter_subreddit_and_users(files, super_user_cutoff=10000):
    """
        Top 10% of subreddits
        Users who have contributed to 3+ subreddits
        Remove bots
        Remove super users
    """
    subreddit_submissions = Counter()
    user_subreddits = defaultdict(set)
    user_post_totals = Counter()

    for fname in tqdm(files, desc='Processing all files'):
        with open(fname, 'rt') as f:
            lines = f.readlines()

        for line in tqdm(lines, position=1, desc='Counting subreddit and user frequency'):
            user, subreddit, freq = line[:-1].split('\t')
            freq = int(freq)
            subreddit = 'r/' + subreddit
            subreddit_submissions[subreddit] += freq
            user_subreddits[user].add(subreddit)
            user_post_totals[user] += freq

    # Grab top 10% of subreddits
    total_subreddits = len(subreddit_submissions)
    top_subreddits = subreddit_submissions.most_common(int(total_subreddits * .1))
    print("Total # of subreddits %d" % total_subreddits)
    print("Ten percent of subreddits %d" % len(top_subreddits))

    # Grab users who post in 3+ subreddits
    print("Total # of users %d" % len(user_subreddits))
    user_subreddits = {k: v for k, v in user_subreddits.items() if len(v) >= 3}
    print("Users who post in 3+ subreddits %d" % len(user_subreddits))

    # Remove super_users
    user_subreddits = {k: v for k, v in user_subreddits.items() if user_post_totals[k] < super_user_cutoff}
    print("Total # of users %d" % len(user_subreddits))
    return list(user_subreddits.keys()), top_subreddits


def build_graph(files):
    g = nx.Graph()

    for fname in tqdm(files, desc='Processing all files'):
        with open(fname, 'rt') as f:
            lines = f.readlines()

        for line in tqdm(lines, position=1, desc='Build graph from file'):
            user, subreddit, freq = line[:-1].split('\t')
            freq = int(freq)

            if not g.has_node(f):
                g.add_node(user)
            if not g.has_node(subreddit):
                g.add_node(subreddit)

            if g.has_edge(user, subreddit):
                g[user][subreddit]['weight'] += freq
            else:
                g.add_edge(user, subreddit, weight=freq)
    return g


if __name__ == '__main__':
    year = '2017'
    directory = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year + '*_filtered.tsv'
    files = glob.glob(directory)

    # users, subreddits = filter_subreddit_and_users(files)
    g = build_graph(files)

    # Dump for later use
    out_dir = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/'
    with open(out_dir + year + '_graph_filtered.pickle', 'wb') as handle:
        pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)
