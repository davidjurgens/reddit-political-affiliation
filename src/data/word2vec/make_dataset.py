import random
import sys
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import random_split

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.word2vec.SubredditUserDataset import SubredditUserDataset


def build_dataset(network_path, validation_split=0.1, max_users=-1):
    print("Building dataset for network: {}".format(network_path))
    user_subreddits, vocab, all_subreddits, user_to_politics = build_user_to_subreddits(network_path)

    # Create validation dataset for political flairs
    pol_validation, pol_training = dict_random_split(user_to_politics, split_size=validation_split)
    print("User to politics training size: {}: " + str(len(pol_training)))
    print("User to politics validation size: {}: " + str(len(pol_validation)))

    # Create validation data for training data
    dataset = SubredditUserDataset(user_subreddits, all_subreddits, user_to_politics=pol_training, max_users=max_users)

    # Reset what are the actual subreddits
    all_subreddits = set(dataset.subreddit_to_idx.keys())
    user_subreddits = dict((k, user_subreddits[k]) for k in user_subreddits if k in dataset.user_to_idx)
    vocab = all_subreddits | set(user_subreddits)

    validation_size = int(validation_split * len(dataset))
    train_size = len(dataset) - validation_size

    # Fix the seed size for reproducibility
    torch.manual_seed(42)
    training, validation = random_split(dataset, [train_size, validation_size])
    print("Train size: {} Validation size: {}".format(train_size, validation_size))
    return dataset, training, validation, pol_validation, vocab, all_subreddits


def build_user_to_subreddits(bipartite_network):
    vocab = set()
    # Using a list so repeat subreddits are weighted!
    user_subreddits = defaultdict(list)
    all_subreddits = set()
    user_to_politics = dict()
    print("Reading in bipartite network")
    network_df = pd.read_csv(bipartite_network, index_col=False, delimiter='\t')
    print("Total rows in the network file: {}".format(len(network_df)))

    count = 0
    for row in network_df.itertuples():
        count += 1
        username, subreddit, politics = row.username, row.subreddit, row.politics
        vocab.add(username)
        vocab.add(subreddit)
        user_subreddits[username].append(subreddit)
        all_subreddits.add(subreddit)
        user_to_politics[username] = row.politics

        if count % 1000000 == 0:
            print("Completed reading {} rows from the bipartite network".format(count))

    all_subreddits = list(all_subreddits)
    print("Length of vocab: " + str(len(vocab)))
    print("User count: " + str(len(user_subreddits)))
    print("Subreddit count: " + str(len(all_subreddits)))

    return user_subreddits, vocab, all_subreddits, user_to_politics


def dict_random_split(d, split_size):
    # Convert into a list and shuffle
    item_list = list(d.items())
    random.seed(42)
    random.shuffle(item_list)

    # Split the list
    split = int(len(item_list) * split_size)
    split_a, split_b = item_list[:split], item_list[split:]

    return dict(split_a), dict(split_b)
