import glob
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.conglomerate_affiliations import get_train_political_affiliations

# Settings
NON_POLITICAL_USERS_MULTIPLE = 5
TOP_SUBREDDIT_COUNT = 1000


def build_bipartite_network(month_file):
    print("Building network for file: {}".format(month_file))
    political_user_data = get_train_political_affiliations()
    political_user_data = filter_training_data_to_a_month(political_user_data, month_file)
    non_political_users = sample_non_political_users(month_file, political_user_data)
    print("Total political users: {}".format(len(political_user_data)))
    print("Total non-political users: {}".format(len(non_political_users)))
    user_to_politics, network_df = build_network(month_file, political_user_data, non_political_users)
    network_df = filter_to_top_subreddits(network_df)
    return user_to_politics, network_df


def build_network(month_file, political_user_data, non_political_users):
    rows = []
    political_users = set(political_user_data.keys())
    user_to_politics = dict()

    for submission in read_submissions(month_file):
        username, subreddit = submission['author'], submission['subreddit']
        if username in political_users:
            political_label = grab_prevailing_politics(political_user_data[username])
            user_to_politics[username] = political_label
            entry = {'username': username, 'subreddit': subreddit, 'politics': political_label}
            rows.append(entry)
        elif username in non_political_users:
            user_to_politics[username] = 0
            entry = {'username': username, 'subreddit': subreddit, 'politics': -1}
            rows.append(entry)

    return user_to_politics, pd.DataFrame(rows, columns=['username', 'subreddit', 'politics'])


def grab_prevailing_politics(user_political_entries):
    politics = [e['politics'] for e in user_political_entries]
    politics_count = Counter(politics)
    most_frequent = politics_count.most_common(1)[0][0]

    # TODO: Do we want to handle the bad actors edge case here (very small minority)
    # The vast majority of the data has only one political party
    if most_frequent == "Democrat":
        return 0
    elif most_frequent == "Republican":
        return 1

    return -1


def filter_to_top_subreddits(network_df):
    print("Filtering network to top {} subreddits".format(TOP_SUBREDDIT_COUNT))
    # Grab the top subreddits
    top_subs = network_df['subreddit'].value_counts()[:TOP_SUBREDDIT_COUNT].index.tolist()

    # Only include top subs
    return network_df[network_df['subreddit'].isin(top_subs)]


def sample_non_political_users(month_file, political_users):
    users = set()
    count = len(political_users) * NON_POLITICAL_USERS_MULTIPLE
    print("Sampling {} non political users from: {}".format(count, month_file))

    for submission in read_submissions(month_file):
        username = submission['author']

        if username not in political_users:
            users.add(username)
            if len(users) % 100000 == 0 and len(users) > 0:
                print("Completed sampling {} non political users from file: {}".format(len(users), month_file))

        if len(users) >= count:
            return users

    # If we run out of usernames
    return users


# TODO: Store the year-month in the conglomerate political data instead of filtering afterwards...
def filter_training_data_to_a_month(political_user_data, file_path):
    year_month = get_year_month(file_path)
    start_ts, end_ts = get_timestamp_range(year_month)

    political_user_data_filtered = defaultdict(list)

    for user, political_data in political_user_data.items():
        has_ts_in_month = False
        for entry in political_data:
            created_utc = entry['created']
            if start_ts <= created_utc <= end_ts:
                has_ts_in_month = True
        if has_ts_in_month:
            political_user_data_filtered[user] = political_data

    return political_user_data_filtered


def get_year_month(file_path):
    month_file = parse_name_from_filepath(file_path)
    year_month_w_extension = month_file.rsplit("_")[1]
    year_month = year_month_w_extension.rsplit('.', 1)[0]
    return year_month


def get_timestamp_range(year_month):
    year, month = year_month.split('-')
    # Timestamps are all UTC
    first_day_date = datetime(int(year), int(month), 1)
    last_day_date = datetime(int(year), int(month) + 1, 1) - timedelta(days=1)
    start_ts = first_day_date.replace(tzinfo=timezone.utc).timestamp()
    end_ts = last_day_date.replace(tzinfo=timezone.utc).timestamp()
    return start_ts, end_ts


def parse_name_from_filepath(filepath):
    name = filepath.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[0]


if __name__ == '__main__':
    train_tsv = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/train.tsv"
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    m_file = files[0]
    print("Using month file: {}".format(m_file))

    bipartite_network = build_bipartite_network(m_file)
    print(bipartite_network.head())
    print(len(bipartite_network))
    out_dir = "/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/"
    out_file = out_dir + get_year_month(m_file) + '.tsv'
    print("Saving network: {}".format(out_file))
    bipartite_network.to_csv(out_file, sep='\t', index=False)
