import glob
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.data_helper import read_submissions, grab_bot_accounts
from src.features.collect_samples import sample_non_political_users
from src.features.political_affiliations.conglomerate_affiliations import get_train_political_affiliations
from src.features.political_affiliations.flair_political_affiliations import get_all_flair_users
from src.features.political_affiliations.comment_political_affiliations import get_all_gold_users, get_all_silver_users

NON_POLITICAL_USERS_MULTIPLE = 5
OUT_DIR = "/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/general/"


def build_bipartite_network(month_file, n_subreddits, min_posts, out_file, sources=None):
    print("Building network for file: {}".format(month_file))
    political_user_data = get_train_political_affiliations()
    political_user_data = filter_training_data_to_a_month(political_user_data, month_file)
    non_political_users = sample_non_political_users(month_file, political_user_data,
                                                     count=len(political_user_data) * NON_POLITICAL_USERS_MULTIPLE)
    print("Total political users: {}".format(len(political_user_data)))
    print("Total non-political users: {}".format(len(non_political_users)))
    user_to_politics, network_df = build_network(month_file, political_user_data, non_political_users)
    return filter_network(network_df, n_subreddits, min_posts, out_file, sources)


def build_network(month_file, political_user_data, non_political_users):
    print("Building bipartite network for file: {}".format(month_file))
    rows = []
    political_users = set(political_user_data.keys())
    user_to_politics = dict()

    for submission in read_submissions(month_file):
        username, subreddit = submission.username, submission.subreddit
        if username in political_users:
            political_label = grab_political_label(political_user_data[username])
            user_to_politics[username] = political_label
            entry = {'username': username, 'subreddit': subreddit, 'politics': political_label}
            rows.append(entry)
        elif username in non_political_users:
            entry = {'username': username, 'subreddit': subreddit, 'politics': -1}
            rows.append(entry)

    return user_to_politics, pd.DataFrame(rows, columns=['username', 'subreddit', 'politics'])


def filter_network(network_df, n_subreddits, min_posts, out_file, sources=None):
    if sources:
        network_df = filter_down_by_source(network_df, sources)
    network_df = filter_to_top_subreddits(network_df, n_subreddits)
    network_df = filter_to_min_posts(network_df, min_posts)

    print("Saving network to file: {}".format(out_file))
    network_df.to_csv(out_file, sep='\t', index=False)


def filter_down_by_source(network_df, sources):
    print("Filtering network down to sources: {}".format(sources))
    users = set()

    if "flair" in sources:
        users.update(get_all_flair_users())

    if "gold" in sources:
        users.update(get_all_gold_users())

    if "silver" in sources:
        users.update(get_all_silver_users())

    return network_df[network_df['username'].isin(users)]


def filter_to_min_posts(network_df, min_posts):
    print("Filtering network down to min posts: {}".format(min_posts))
    user_submission_counts = network_df['username'].value_counts()
    return network_df[
        network_df['username'].isin(user_submission_counts.index[user_submission_counts.gt(min_posts - 1)])]


def filter_to_top_subreddits(network_df, n_subreddits):
    print("Filtering network to top {} subreddits".format(n_subreddits))
    # Grab the top subreddits
    top_subs = network_df['subreddit'].value_counts()[:n_subreddits].index.tolist()

    # Only include top subs
    return network_df[network_df['subreddit'].isin(top_subs)]


def grab_political_label(user_political_entries):
    if user_political_entries[0]['politics'] == "Democrat":
        return 0
    elif user_political_entries[0]['politics'] == "Republican":
        return 1

    print("Failed to grab prevailing politics for data: {}".format(user_political_entries))
    return -1


# TODO: Store the year-month in the conglomerate political data instead of filtering afterwards...
def filter_training_data_to_a_month(political_user_data, file_path):
    year_month = get_year_month(file_path)
    print("Filtering data down to month: {}".format(year_month))
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

    print("Completed filtering data down to month: {}".format(year_month))
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
    if int(month) == 12:
        next_month = 1
    else:
        next_month = int(month) + 1
    last_day_date = datetime(int(year), next_month, 1) - timedelta(days=1)
    start_ts = first_day_date.replace(tzinfo=timezone.utc).timestamp()
    end_ts = last_day_date.replace(tzinfo=timezone.utc).timestamp()
    return start_ts, end_ts


def parse_name_from_filepath(filepath):
    name = filepath.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[0]


def remove_bots(network_df, out_file):
    """ Should not be necessary but just in case """
    print("Removing bots for : {}".format(out_file))
    bots = grab_bot_accounts()
    filtered_df = network_df[~network_df['username'].isin(bots)]
    print("Starting network size: {}. Filtered network size: {}".format(len(network_df), len(filtered_df)))
    filtered_df.to_csv(OUT_DIR + out_file, sep='\t', index=False)


if __name__ == '__main__':
    train_tsv = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/train.tsv"
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    min_posts = 5
    n_subreddits = 25000

    for m_file in files[::-1]:
        print("Using month file: {}".format(m_file))
        out_file = OUT_DIR + get_year_month(m_file) + "_{}subs_{}posts.tsv".format(n_subreddits, min_posts)

        build_bipartite_network(m_file, n_subreddits, min_posts, out_file)
