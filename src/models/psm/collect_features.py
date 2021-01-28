import glob
import sys
from collections import defaultdict

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics


def collect_user_features(month_file, users):
    """
        Grab all submission metadata for a subset of users
        month_file: Compressed data file for a single month of data
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


def get_random_users_non_political(month_file, count, political_users):
    users = set()
    for submission in read_submissions(month_file):
        username = submission['author']

        if username not in political_users:
            users.add(political_users)

        if len(users) >= count:
            return users

    return users


def get_post_and_comment_counts(user_features, user):
    post_counts, comment_counts = 0, 0
    for features in user_features[user]:
        for entry in features:
            if entry['type'] == 'post':
                post_counts += 1
            else:
                comment_counts += 1
    return post_counts, comment_counts


def get_participation_in_subreddits(user_features, subreddits, user):
    submission_count = 0

    for features in user_features[user]:
        for entry in features:
            if entry['subreddit'] in subreddits:
                submission_count += 1
    return submission_count


def create_features_df(month_file, users, featured_subreddits):
    user_features = collect_user_features(month_file, users)
    rows = []
    for user in users:
        post_counts, comment_counts = get_post_and_comment_counts(user_features, user)
        subreddit_participation = get_participation_in_subreddits(user_features, featured_subreddits, user)
        row = {'post_counts': post_counts, 'comment_counts': comment_counts,
               'subreddit_participation': subreddit_participation}
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    m_file = files[-1]
    featured_subreddits = []
    political_users = read_in_user_politics([m_file])

    df_political_users = create_features_df(m_file, users=political_users, featured_subreddits=featured_subreddits)
    df_political_users['political'] = 1

    non_political_users = get_random_users_non_political(m_file, len(political_users) * 4, political_users)
    df_non_political_users = create_features_df(m_file, non_political_users, featured_subreddits=featured_subreddits)
    df_non_political_users['political'] = 0

    df = pd.concat([df_political_users, df_non_political_users])
    df.to_csv('/shared/0/projects/reddit-political-affiliation/data/psm/{}.tsv'.format(m_file), index=False, sep='\t')
