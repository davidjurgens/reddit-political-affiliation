import sys
import gzip
from glob import glob
import json
from collections import defaultdict

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.data_helper import grab_bot_accounts
from src.features.political_affiliations.subreddit_political_labels import subreddit_politics

DATA_DIRECTORY = "/shared/2/datasets/reddit-dump-all/user-subreddit-counts/"
OUTPUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/community-affiliations/"


def get_user_politics(month_file):
    print("Getting user politics by community participation for file: " + month_file)
    user_politics = defaultdict(list)
    bots = grab_bot_accounts()

    with gzip.open(month_file, 'r') as f:
        for line in f:
            user_counts = json.loads(line.strip())
            user = user_counts['user']
            for subreddit, count in user_counts['counts'].items():
                sub = 'r/' + subreddit
                if sub in subreddit_politics and user not in bots:
                    user_politics[user].append(subreddit_politics[sub])

    return user_politics


def filter_users_who_participate_in_both_politics(user_politics):
    print("Starting length of political users: {}".format(len(user_politics)))
    filtered_user_politics = {}
    for user, political_affiliations in user_politics.items():
        if len(set(political_affiliations)) == 1:
            filtered_user_politics[user] = political_affiliations[0]

    print("Length of filtered political users: {}".format(len(filtered_user_politics)))
    return filtered_user_politics


def output_to_tsv(month_file, user_politics):
    # Format the data as a dataframe
    rows = []
    for user, politics in user_politics.items():
        rows.append({'username': user, 'politics': politics, 'source': 'community'})

    df = pd.DataFrame(rows)
    # Grab everything after the last slash and change the extension
    out_file = month_file.rsplit('/', 1)[-1].replace(".gz", ".tsv")

    print("Saving user politics to file: " + OUTPUT_DIRECTORY + out_file)
    df.to_csv(OUTPUT_DIRECTORY + out_file, sep='\t', index=False)


def get_user_politics_for_community_labels(in_tsv):
    print("Reading in political affiliations for community labels from: " + in_tsv)
    user_politics = {}
    df = pd.read_csv(in_tsv, sep='\t', index_col=False)
    for row in df.iterrows():
        user_politics[row['username']] = row['politics']

    return user_politics


if __name__ == '__main__':
    month_files = glob(DATA_DIRECTORY + "*.gz")

    for m_file in month_files:
        print("Starting work on file: " + m_file)
        political_affiliations = get_user_politics(m_file)
        filtered_politics = filter_users_who_participate_in_both_politics(political_affiliations)
        output_to_tsv(m_file, user_politics=filtered_politics)
