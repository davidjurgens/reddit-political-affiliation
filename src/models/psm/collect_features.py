import glob
import sys
from collections import defaultdict, OrderedDict
from sklearn.utils import shuffle
import os.path
import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics
from src.models.psm.features import *

# SETTINGS
user_politics_dir = '/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/'
user_data_dir = '/shared/0/projects/reddit-political-affiliation/data/psm/user-data/'
user_features_dir = '/shared/0/projects/reddit-political-affiliation/data/psm/features/'


def initial_collect_features(month_file, politics_in_file, control_group_multiple):
    """
        Grab all submissions/metadata for political users and save the results to a TSV. Repeat for a control group
        month_file: Compressed raw data file for a single month of data
        politics_in_file: File path to TSV of users and their political affiliations
        control_group_multiple: Size of the control group in relation to the # of political users
    """
    fname = parse_name_from_filepath(month_file)

    # Collect all post/comment metadata for political users
    political_users = read_in_user_politics([politics_in_file])
    print("Total number of political users: {}".format(len(political_users)))
    pol_user_features = collect_submission_data(month_file, users=political_users)
    out_pol_tsv = user_data_dir + fname + '_political.tsv'
    save_user_features(pol_user_features, out_pol_tsv)

    # Collect all post/comment metadata for non-political users
    non_political_users = get_random_users_non_political(m_file, len(political_users) * control_group_multiple,
                                                         political_users)
    print("Total number of non-political users: {}".format(len(non_political_users)))
    non_pol_user_features = collect_submission_data(month_file, users=non_political_users)
    out_non_pol_tsv = user_data_dir + fname + '_non_political.tsv'
    save_user_features(non_pol_user_features, out_non_pol_tsv)

    return pol_user_features, non_pol_user_features


def collect_submission_data(month_file, users):
    """
        Grab all submission metadata for a subset of users
        month_file: Compressed raw data file for a single month of data
        users: should be a set or dict for O(1) lookups
    """
    user_features = defaultdict(list)
    print("Collecting user features for month: {}".format(month_file))
    for submission in read_submissions(month_file):
        username, subreddit, created = submission['author'], submission['subreddit'], submission['created_utc']
        if username in users:
            submission_type = 'post' if 'num_comments' in submission else 'comment'
            entry = {'username': username, 'type': submission_type, 'subreddit': subreddit, 'created': created}
            user_features[username].append(entry)
    return user_features


def save_user_features(user_features, out_tsv):
    print("Saving user features to file: {}".format(out_tsv))
    with open(out_tsv, 'w') as f:
        for user, features in user_features.items():
            for entry in features:
                f.write(
                    "{}\t{}\t{}\t{}\n".format(entry['username'], entry['type'], entry['subreddit'], entry['created']))


def load_user_features(in_tsv):
    print("Loading user features for file: {}".format(in_tsv))
    user_features = defaultdict(list)
    with open(in_tsv, 'r') as f:
        for line in f:
            username, sub_type, subreddit, created = line.split('\t')
            entry = {'username': username, 'type': sub_type, 'subreddit': subreddit, 'created': created}
            user_features[username].append(entry)

    return user_features


def get_random_users_non_political(month_file, count, political_users):
    users = set()
    for submission in read_submissions(month_file):
        username = submission['author']

        if username not in political_users:
            users.add(username)

        if len(users) >= count:
            return users

    return users


def get_post_and_comment_counts(user_features, user):
    post_counts, comment_counts = 0, 0
    for entry in user_features[user]:
        if entry['type'] == 'post':
            post_counts += 1
        else:
            comment_counts += 1

    return post_counts, comment_counts


def get_participation_in_subreddits(user_features, subreddits, user):
    submission_count = 0

    for entry in user_features[user]:
        if entry['subreddit'] in subreddits:
            submission_count += 1

    return submission_count


def get_time_of_day_counts(user_features, user):
    counts = OrderedDict({'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0})

    for entry in user_features[user]:
        result = get_time_of_day(entry['created'])
        counts[result] += 1

    return counts.values()


def build_features_df(political_user_features, non_political_user_features, top_subreddits, political_subreddits):
    print("Building features DataFrame")
    rows = []

    for user, features in political_user_features.items():
        post_counts, comment_counts = get_post_and_comment_counts(features, user)
        top_subreddit_participation = get_participation_in_subreddits(features, top_subreddits, user)
        pol_subreddit_participation = get_participation_in_subreddits(features, political_subreddits, user)
        morning_post_count, afternoon_post_count, evening_post_count, night_post_count = get_time_of_day_counts(
            features, user)
        row = {'post_counts': post_counts, 'comment_counts': comment_counts,
               'top_subreddit_participation': top_subreddit_participation,
               'political_subreddit_participation': pol_subreddit_participation,
               'morning_post_count': morning_post_count, 'afternoon_post_count': afternoon_post_count,
               'evening_post_count': evening_post_count, 'night_post_count': night_post_count, 'is_political': 1}
        rows.append(row)

    for user, features in non_political_user_features.items():
        post_counts, comment_counts = get_post_and_comment_counts(features, user)
        top_subreddit_participation = get_participation_in_subreddits(features, top_subreddits, user)
        pol_subreddit_participation = get_participation_in_subreddits(features, political_subreddits, user)
        morning_post_count, afternoon_post_count, evening_post_count, night_post_count = get_time_of_day_counts(
            features, user)
        row = {'post_counts': post_counts, 'comment_counts': comment_counts,
               'top_subreddit_participation': top_subreddit_participation,
               'political_subreddit_participation': pol_subreddit_participation,
               'morning_post_count': morning_post_count, 'afternoon_post_count': afternoon_post_count,
               'evening_post_count': evening_post_count, 'night_post_count': night_post_count, 'is_political': 0}
        rows.append(row)

    df = pd.DataFrame(rows)
    return shuffle(df)


def parse_name_from_filepath(filepath):
    name = filepath.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[0]


def user_features_already_exist(month_file):
    fname = parse_name_from_filepath(month_file)
    pol_file = user_data_dir + fname + '_political.tsv'
    non_pol_file = user_data_dir + fname + '_non_political.tsv'
    print(pol_file, non_pol_file)
    return os.path.isfile(pol_file) and os.path.isfile(non_pol_file)


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    m_file = files[0]
    file_name = parse_name_from_filepath(m_file)
    print(m_file, file_name)

    if user_features_already_exist(m_file):
        print("User features already exist. Loading them in")
        pol_user_features = load_user_features(user_data_dir + file_name + '_political.tsv')
        non_pol_user_features = load_user_features(user_data_dir + file_name + '_non_political.tsv')
    else:
        print("Collecting user features for the first time. Parsing through the entire month of the data")
        pol_user_features, non_pol_user_features = initial_collect_features(m_file,
                                                                            user_politics_dir + file_name + '.tsv',
                                                                            control_group_multiple=5)

    df_features = build_features_df(pol_user_features, non_pol_user_features, top_subreddits, political_subreddits)
    print(df_features.head(10))
    print("Saving features DataFrame to TSV")
    df_features.to_csv('/shared/0/projects/reddit-political-affiliation/data/psm/features/{}'
                       .format(file_name + '.tsv'), index=False, sep='\t')
