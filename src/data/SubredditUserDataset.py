import random
import pickle
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class SubredditUserDataset(Dataset):

    def __init__(self, user_subreddits, all_subreddits, user_to_politics, num_negative_samples=5, max_users=-1):
        self.pos_and_neg_samples = []
        # Mappings to the embedding dimensions
        self.user_to_idx = {}
        self.subreddit_to_idx = {}
        self.user_subreddits = user_subreddits
        self.user_to_politics = user_to_politics

        def get_sub_idx(subreddit):
            if subreddit in self.subreddit_to_idx:
                sub_idx = self.subreddit_to_idx[subreddit]
            else:
                sub_idx = len(self.subreddit_to_idx)
                self.subreddit_to_idx[subreddit] = len(self.subreddit_to_idx)
            return sub_idx

        num_users = len(user_subreddits) if max_users < 0 else max_users

        for i, (user, subreddits) in enumerate(tqdm(user_subreddits.items(), total=num_users,
                                                    desc='Converting data to PyTorch')):
            if i >= num_users:
                break

            if user in user_to_politics:
                politics = user_to_politics[user]
            else:
                politics = -1

            self.user_to_idx[user] = len(self.user_to_idx)
            user_idx = self.user_to_idx[user]

            # Add all the positive samples
            for subreddit in subreddits:
                sub_idx = get_sub_idx(subreddit)
                self.pos_and_neg_samples.append((np.array([user_idx, sub_idx]), politics, 1))

            # Choose fixed negative samples
            neg = []
            num_neg = len(subreddits) * num_negative_samples
            # guard against super active users?
            num_neg = min(num_neg, len(all_subreddits) - num_neg)
            while len(neg) < num_neg:
                sub = all_subreddits[random.randint(0, len(all_subreddits) - 1)]
                if sub not in subreddits:  # Check if also in neg?
                    neg.append(sub)
            for _ in neg:
                sub_idx = get_sub_idx(subreddit)
                self.pos_and_neg_samples.append((np.array([user_idx, sub_idx]), politics, 0))

    def num_users(self):
        return len(self.user_to_idx)

    def num_subreddits(self):
        return len(self.subreddit_to_idx)

    def __len__(self):
        return len(self.pos_and_neg_samples)

    def __getitem__(self, idx):
        return self.pos_and_neg_samples[idx]

    def id_mappings_to_tsv(self, path):
        print("Saving user id mappings")
        dict_to_tsv(file_path=path + '_user_ids.tsv', dictionary=self.user_to_idx)

        print("Saving subreddit id mappings")
        dict_to_tsv(file_path=path + '_subreddit_ids.tsv', dictionary=self.subreddit_to_idx)

        print("Saving user subreddits")
        with open(path + '_user_subreddits.tsv', 'w') as f:
            for user, subreddits in self.user_subreddits.items():
                for sub in subreddits:
                    f.write("{}\t{}\n".format(user, sub))

    def load_id_mappings(self, path):
        print("Loading user and subreddit id mappings")
        self.user_to_idx = dict_from_tsv(path + '_user_ids.tsv')
        self.subreddit_to_idx = dict_from_tsv(path + '_subreddit_ids.tsv')

        print("Loading user subreddits")
        user_subreddits = defaultdict(list)
        with open(path + 'user_subreddits.tsv', 'r') as f:
            for line in f:
                user, subreddit = line.strip().split('\t')
                user_subreddits[user].append(subreddit)
        self.user_subreddits = user_subreddits


def dict_to_tsv(file_path, dictionary):
    with open(file_path, 'w') as f:
        for k, v in dictionary.items():
            f.write("{}\t{}\n".format(k, v))


def dict_from_tsv(file_path):
    dictionary = {}
    with open(file_path, 'r') as f:
        for line in f:
            k, v = line.strip().split('\t')
            dictionary[k] = int(v)
    return dictionary
