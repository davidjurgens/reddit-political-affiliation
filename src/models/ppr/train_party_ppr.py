import argparse
import glob
import pickle
import random
from collections import *
from collections import Counter

import networkx as nx
from tqdm.notebook import tqdm


def load_user_flair_labels(directory):
    files = glob.glob(directory)

    user_to_politic_counts = defaultdict(Counter)

    for fname in tqdm(files):
        with open(fname, 'rt') as f:
            for line in f:
                user, politics, freq = line.split('\t')
                user_to_politic_counts[user][politics] += int(freq)

    print("User to politic counts: " + str(len(user_to_politic_counts)))
    return filter_multiparty_users(user_to_politic_counts)


def filter_multiparty_users(user_to_politic_counts):
    user_to_politics = {}
    for u, pc in user_to_politic_counts.items():
        if len(pc) > 1:
            continue
        user_to_politics[u] = list(pc.keys())[0]
    print('Saw political affiliations for %d users' % len(user_to_politics))
    return user_to_politics


def split_test_train(user_to_politics):
    all_identified_users = list(user_to_politics.keys())
    random.seed(42)
    random.shuffle(all_identified_users)

    train_users = all_identified_users[:int(0.9 * len(all_identified_users))]
    test_users = all_identified_users[int(0.9 * len(all_identified_users)):]
    print("Number of train users: " + str(len(train_users)))
    print("Number of test users: " + str(len(test_users)))
    print(len(train_users), len(test_users))
    return train_users, test_users


def get_personalization(g, users, epsilon=0.00001):
    print("Generation personalization for page rank")
    n = g.number_of_nodes()
    personalization = {}

    for n in tqdm(g.nodes, total=n):
        if n in users:
            personalization[n] = 1
        else:
            personalization[n] = epsilon

    return personalization


def train_party_ppr(g, user_to_politics, out_directory, year):
    train, test = split_test_train(user_to_politics)
    rep_users = set([k for k in train if user_to_politics[k] == 'Republican'])
    dem_users = set([k for k in train if user_to_politics[k] == 'Democrat'])
    print("Number of Republican users: " + str(len(rep_users)))
    print("Number of Democrat users: " + str(len(dem_users)))

    rep_personalization = get_personalization(g, rep_users)
    dem_personalization = get_personalization(g, dem_users)

    rep_ppr = nx.pagerank(g, personalization=rep_personalization)
    print("Completed rep pagerank. Saving to file: " + out_directory + str(year) + "_rep_ppr.pickle")

    with open(out_directory + str(year) + "_rep_ppr.pickle", 'wb') as handle:
        pickle.dump(rep_ppr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dem_ppr = nx.pagerank(g, personalization=dem_personalization)
    print("Completed dem pagerank. Saving to file: " + out_directory + str(year) + "_dem_ppr.pickle")

    with open(out_directory + str(year) + "_dem_ppr.pickle", 'wb') as handle:
        pickle.dump(dem_ppr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training complete for both parties")


def load_existing_graph(input_path):
    print("Loading in graph from {}".format(input_path))
    with open(input_path, 'rb') as handle:
        g = pickle.load(handle)
        return g


if __name__ == '__main__':
    user_flair_labels = load_user_flair_labels(
        '/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/20*.tsv')

    parser = argparse.ArgumentParser(description='Train personalized page rank by party')
    parser.add_argument('g', help='Path to existing graph', type=str)
    parser.add_argument('o', help='Output path to store trained page rank models', type=str)
    parser.add_argument('y', help="Year of data", type=int)
    args = parser.parse_args()

    g = load_existing_graph(args.g)
    train_party_ppr(g, user_flair_labels, args.o, args.y)
