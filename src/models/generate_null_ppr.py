import argparse
import csv
import glob
import pickle
import random
from collections import Counter

import networkx as nx
from tqdm.notebook import tqdm


def build_bipartite_graph(directory, min_sub_threshold=500):
    files = glob.glob(directory)

    filtered_subreddits = get_subreddits_above_threshold(files, min_sub_threshold)
    g = nx.Graph()

    users = set()
    subreddits = set()

    for fname in tqdm(files, desc='Processing all files'):
        with open(fname, 'rt') as f:
            lines = f.readlines()

        for line in tqdm(lines, position=1, desc='Build graph from file'):
            user, subreddit, freq = line[:-1].split('\t')
            freq = int(freq)

            # distinguish users from subreddits
            subreddit = 'r/' + subreddit

            users.add(user)
            subreddits.add(subreddit)

            if subreddit in filtered_subreddits:
                if not g.has_node(user):
                    g.add_node(user)
                if not g.has_node(subreddit):
                    g.add_node(subreddit)

                if g.has_edge(user, subreddit):
                    g[user][subreddit]['weight'] += freq
                else:
                    g.add_edge(user, subreddit, weight=freq)

    return filter_single_edge_nodes(g)


def get_subreddits_above_threshold(files, min_sub_threshold):
    """ Get subreddits that have a given number of submissions """

    sub_frequencies = Counter()
    for fname in tqdm(files, desc='Processing all files'):
        with open(fname, 'rt') as f:
            lines = f.readlines()

        for line in tqdm(lines, position=1, desc='Build graph from file'):
            user, subreddit, freq = line[:-1].split('\t')
            subreddit = 'r/' + subreddit
            sub_frequencies[subreddit] += int(freq)

    return {sub: count for sub, count in sub_frequencies.items() if count >= min_sub_threshold}


def filter_single_edge_nodes(g):
    """ Get rid of users who only have an edge to one subreddit """
    to_remove = [n for (n, deg) in g.degree() if deg == 1]
    g.remove_nodes_from(to_remove)
    return g


def get_personalization(g, sample):
    n = g.number_of_nodes()
    personalization = {}

    for n in tqdm(g.nodes, total=n):
        if n in sample:
            personalization[n] = 1
        else:
            personalization[n] = 0.00001
    return personalization


def compute_ppr_random_sample(g, sample_size, repeat=3):
    for i in range(repeat):
        print("Computing pagerank with random seed of size: {}".format(sample_size))
        sample = random.sample(g.nodes(), sample_size)
        ppr = nx.pagerank(g, personalization=get_personalization(g, sample))
        yield ppr


def output_to_tsv(fname, ppr_result):
    print("Saving PPR result to file: {}".format(fname))
    with open(fname, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for user, score in ppr_result.items():
            tsv_writer.writerow([user, score])
    print("{} successfully completed".format(fname))


def load_existing_graph(input_path):
    print("Loading in graph from {}".format(input_path))
    with open(input_path, 'rb') as handle:
        g = pickle.load(handle)
        return g


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate null PPR models for normalization')
    parser.add_argument('seed_size', type=int)
    parser.add_argument('year', type=str)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--g', help='Path to existing graph', type=str)
    args = parser.parse_args()

    if args.g:
        g = load_existing_graph(args.g)
    else:
        directory = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + args.year + '*.tsv'
        g = build_bipartite_graph(directory)

    out_directory = '/shared/0/projects/reddit-political-affiliation/data/ppr-scores/'

    for result in compute_ppr_random_sample(g, args.seed_size, repeat=args.repeat):
        # Generate a random 'id' for the file. Just need a way to save under different names
        fid = random.randrange(1, 10 ** 6)
        fname = out_directory + args.year + "_" + str(fid) + '.tsv'
        print("{} is complete".format(fname))
        output_to_tsv(fname, result)
