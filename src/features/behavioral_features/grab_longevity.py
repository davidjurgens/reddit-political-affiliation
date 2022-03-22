import datetime
import gzip
import json
import random
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users
from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/behavior/"
random.seed(2021)


def get_months_user_posted_in(users):
    month_files = glob(DATA_DIRECTORY + "*.gz")

    user_month_totals = defaultdict(lambda: defaultdict(int))

    for month_file in month_files:
        year_month = parse_year_month_from_filename(month_file)
        print("Working on year month: " + year_month)
        with gzip.open(month_file, 'r') as f:
            for line in f:
                user_counts = json.loads(line.strip())
                user = user_counts['user']
                if user in users:
                    # Sum up all their contributions that month
                    total = 0
                    for sub, count in user_counts['counts'].items():
                        total += count
                    user_month_totals[user][year_month] = total

    return user_month_totals


def parse_year_month_from_filename(file_name):
    return file_name.rsplit('/', 1)[-1].replace(".gz", "").replace("RC_", "")


def compute_longevity_and_avg_posts_per_month(user_month_totals):
    print("Computing longevity and avg posts per month for all {} users".format(len(user_month_totals.keys())))
    user_avg_per_month = {}
    user_longevity = {}

    for user, month_counts in user_month_totals.items():
        all_year_months = list(month_counts.keys())
        user_longevity[user] = compute_longevity_from_months(all_year_months)
        # Compute the user's avg posts per month (active months only)
        avg_per_month = np.mean(list(month_counts.values()))
        user_avg_per_month[user] = avg_per_month

    return user_avg_per_month, user_longevity


def compute_longevity_from_months(year_months):
    if len(year_months) == 1:
        return 1

    # Convert the list to date times
    year_months_dates = []
    for year_month in year_months:
        year, month = year_month.split('-')
        d = datetime.date(int(year), int(month), 1)
        year_months_dates.append(d)

    year_months_dates.sort()

    min_date, max_date = year_months_dates[0], year_months_dates[-1]

    # Return the difference in months
    return (max_date.year - min_date.year) * 12 + max_date.month - min_date.month


def save_user_longevity_and_avg_posts_per_month(user_avg_per_month, user_longevity):
    users = list(user_avg_per_month.keys())
    rows = []

    for user in users:
        longevity = user_longevity[user]
        avg_per_month = user_avg_per_month[user]
        entry = {'username': user, 'longevity_months': int(longevity), 'avg_per_active_months': avg_per_month}
        rows.append(entry)

    df = pd.DataFrame(rows)
    print(df.head())
    print("Saving file to: {}".format(OUTPUT_DIRECTORY + "longevity.tsv"))
    df.to_csv(OUTPUT_DIRECTORY + "longevity.tsv", sep='\t', index=False)


def load_user_longevity_df():
    return pd.read_csv(OUTPUT_DIRECTORY + "longevity.tsv", sep='\t', index_col=False)


if __name__ == '__main__':
    all_political_users = get_all_political_users()
    bad_actors = set(read_in_bad_actors_from_tsv(
        ['/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_365_days_1_flip_flop.tsv']).keys())

    all_political_users.update(bad_actors)
    print("Total number of users: {}".format(len(all_political_users)))
    user_month_totals = get_months_user_posted_in(all_political_users)
    user_avg_per_month, user_longevity = compute_longevity_and_avg_posts_per_month(user_month_totals)
    save_user_longevity_and_avg_posts_per_month(user_avg_per_month, user_longevity)
