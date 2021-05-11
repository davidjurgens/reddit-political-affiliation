import sys
import gzip
import json
from glob import glob
import pandas as pd
from collections import defaultdict

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/behavior/"


def grab_subreddit_counts_for_users(users):
    """ Grab subreddit submission counts for a subset of users """
    month_files = glob(DATA_DIRECTORY + "*.gz")
    user_subreddit_counts = defaultdict(list)

    for month_file in month_files:
        with gzip.open(month_file, 'r') as f:
            for line in f:
                user_counts = json.loads(line.strip())
                user = user_counts['user']
                subreddit_counts = user_counts['counts']
                if user in users:
                    pass


if __name__ == '__main__':
    pass
