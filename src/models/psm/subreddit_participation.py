import gzip
import json
import random
import sys
from collections import defaultdict
from glob import glob

import pandas as pd

from src.data.data_helper import get_all_raw_files
from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv
from src.features.behavioral_features.collect_samples import collect_non_political_usernames
from src.models.psm.subreddit_constants import top_100_subreddits, political_subreddits

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/behavior/subreddit_participation/"
random.seed(2021)

""" Grab user participation for select subreddits by month """

all_subreddits = top_100_subreddits.union(political_subreddits)


def get_subreddit_participation_for_all_months(users):
    month_files = glob(DATA_DIRECTORY + "*.gz")

    for month_file in month_files:
        year_month = parse_year_month_from_filename(month_file)
        user_subreddit_participation = defaultdict(lambda: dict.fromkeys(all_subreddits, 0))
        print("Working on year month: " + year_month)
        with gzip.open(month_file, 'r') as f:
            for line in f:
                user_counts = json.loads(line.strip())
                user = user_counts['user']
                if user in users:
                    for sub, count in user_counts['counts'].items():
                        if sub in all_subreddits:
                            user_subreddit_participation[user][sub] = count

        save_month_participation_to_tsv(user_subreddit_participation, year_month)


def save_month_participation_to_tsv(user_subreddit_participation, year_month):
    print("Save user participation for year month: {}".format(year_month))
    # Format as df
    rows = []
    for user, subreddit_participation in user_subreddit_participation.items():
        entry = {'username': user, **subreddit_participation}
        rows.append(entry)

    if rows:
        out_file = OUTPUT_DIRECTORY + str(year_month) + '.tsv'
        print("Saving data to: {}".format(out_file))
        df = pd.DataFrame(rows)
        df.to_csv(out_file, sep='\t')


def parse_year_month_from_filename(file_name):
    return file_name.rsplit('/', 1)[-1].replace(".gz", "").replace("RC_", "")


def load_subreddit_participation(year, month):
    year_month = str(year) + '-' + str(month)
    in_file = OUTPUT_DIRECTORY + str(year_month) + '.tsv'
    return pd.read_csv(in_file, sep='\t')


if __name__ == '__main__':
    all_users = get_all_political_users()
    bad_actors = set(read_in_bad_actors_from_tsv(
        ['/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_365_days_1_flip_flop.tsv']).keys())

    non_political_usernames = collect_non_political_usernames(get_all_raw_files(), all_users, count=10000)
    all_users.update(bad_actors)
    all_users.update(non_political_usernames)
    print("Total number of users: {}".format(len(all_users)))

    get_subreddit_participation_for_all_months(all_users)
