import glob
from collections import defaultdict, Counter

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from src.data.SubredditUserDataset import SubredditUserDataset


def build_dataset(network_directory, flair_directory, validation_split=0.1, months=-1):
    network_files = glob.glob(network_directory)[:months]
    user_subreddits, vocab, all_subreddits = build_user_to_subreddits(network_files)
    flair_files = glob.glob(flair_directory)
    user_to_politics = read_political_affiliations(flair_files)
    training_dataset = SubredditUserDataset(user_subreddits, all_subreddits, user_to_politics)

    validation_size = int(validation_split * len(training_dataset))
    train_size = len(training_dataset) - validation_size

    # Fix the seed size for reproducibility
    torch.manual_seed(42)
    training, validation = random_split(training_dataset, [train_size, validation_size])
    print("Train size: {} Validation size: {}".format(train_size, validation_size))
    return training_dataset, training, validation, vocab


def build_user_to_subreddits(files):
    vocab = set()
    user_subreddits = defaultdict(set)
    all_subreddits = set()

    for fname in tqdm(files, desc="Processing all files"):
        print(fname)
        with open(fname, 'rt') as f:
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


def read_political_affiliations(files):
    user_to_politic_counts = defaultdict(Counter)

    for fname in tqdm(files):
        with open(fname, 'rt') as f:
            for line in f:
                user, politics, freq = line.split('\t')
                user_to_politic_counts[user][politics] += int(freq)

    print("User to politic counts: " + str(len(user_to_politic_counts)))
    print(list(user_to_politic_counts.items())[:10])

    user_to_politics = {}
    for u, pc in user_to_politic_counts.items():
        if len(pc) > 1:
            continue
        user_to_politics[u] = list(pc.keys())[0]

    print('Saw political affiliations for %d users' % len(user_to_politics))
    return convert_affiliations_to_binary(user_to_politics)


def convert_affiliations_to_binary(user_to_politics):
    for user, politics in user_to_politics.items():
        if politics == "Democrat":
            user_to_politics[user] = 0
        else:
            user_to_politics[user] = 1

    return user_to_politics
