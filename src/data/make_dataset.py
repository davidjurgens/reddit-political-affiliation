import glob
import random
from collections import defaultdict, Counter

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from src.data.SubredditUserDataset import SubredditUserDataset


def build_dataset(network_path, flair_directory, comment_directory, validation_split=0.1, max_users=-1):
    user_subreddits, vocab, all_subreddits = build_user_to_subreddits(network_path)
    flair_files = glob.glob(flair_directory)
    comment_files = glob.glob(comment_directory)
    flair_politics = read_flair_political_affiliations(flair_files)
    comment_politics = read_comment_political_affiliations(comment_files)
    user_to_politics = {**flair_politics, **comment_politics}

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
    return dataset, training, validation, pol_validation, vocab


def build_user_to_subreddits(bipartite_network):
    vocab = set()
    user_subreddits = defaultdict(set)
    all_subreddits = set()

    with open(bipartite_network, 'rt') as f:
        lines = f.readlines()

    for line in tqdm(lines, position=1, desc='Building vocab from file'):
        user, subreddit, freq = line[:-1].split('\t')
        vocab.add(user)
        vocab.add(subreddit)
        user_subreddits[user].add(subreddit)
        all_subreddits.add(subreddit)

    all_subreddits = list(all_subreddits)
    print("Length of vocab: " + str(len(vocab)))
    print("User count: " + str(len(user_subreddits)))
    print("Subreddit count: " + str(len(all_subreddits)))

    return user_subreddits, vocab, all_subreddits


def read_flair_political_affiliations(files):
    user_to_politic_counts = defaultdict(Counter)

    for fname in tqdm(files, desc="Loading flair politics"):
        with open(fname, 'rt') as f:
            for line in f:
                user, politics, freq = line.split('\t')
                politics = politics.strip().lower()
                user_to_politic_counts[user][politics] += int(freq)

    print("User to politic counts: " + str(len(user_to_politic_counts)))
    print(list(user_to_politic_counts.items())[:10])

    user_to_politics = {}
    for u, pc in user_to_politic_counts.items():
        if len(pc) > 1:
            continue
        user_to_politics[u] = list(pc.keys())[0]

    print('Saw political affiliations for %d users from flairs' % len(user_to_politics))
    return convert_affiliations_to_binary(user_to_politics)


def read_comment_political_affiliations(files):
    user_to_politics = {}

    for fname in tqdm(files, desc="Loading politics from comment affiliations"):
        with open(fname, 'r') as f:
            for line in f:
                user, politics = line.split('\t')
                user_to_politics[user] = politics.strip().lower()

    print('Saw political affiliations for %d users from comments' % len(user_to_politics))
    return convert_affiliations_to_binary(user_to_politics)


def convert_affiliations_to_binary(user_to_politics):
    dems, reps = 0, 0

    for user, politics in user_to_politics.items():
        if politics == "democrat":
            user_to_politics[user] = 0
        elif politics == "republican":
            user_to_politics[user] = 1

    print("Number of democrats: {}".format(dems))
    print("Number of republicans: {}".format(reps))
    return user_to_politics


def dict_random_split(d, split_size):
    # Convert into a list and shuffle
    item_list = list(d.items())
    random.seed(42)
    random.shuffle(item_list)

    # Split the list
    split = int(len(item_list) * split_size)
    split_a, split_b = item_list[:split], item_list[split:]

    return dict(split_a), dict(split_b)
