import sys
from collections import defaultdict, OrderedDict
from datetime import datetime

import pandas as pd
import pytz

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions, get_all_raw_files
from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users
from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv


def collect_user_submission_data(month_files, users):
    user_features = defaultdict(list)
    for m_file in month_files:
        print("Collecting user features for month: {}".format(m_file))

        for submission in read_submissions(m_file):
            if submission.username in users:
                if submission.is_post():
                    submission_type = 'post'
                else:
                    submission_type = 'comment'
                entry = {'username': submission.username, 'subreddit': submission.subreddit, 'score': submission.score,
                         'submission_type': submission_type, 'gilded': submission.gilded,
                         'total_awards': submission.total_awards, 'controversiality': submission.controversiality}
                user_features[submission.username].append(entry)

    return user_features


def build_user_features_df(user_features):
    print("Building dataframe for user features")
    print("Total of {} users".format(len(user_features)))
    rows = []

    # Loop through the user posts and tally everything up
    for user, features in user_features.items():
        post_count, comment_count, total_awards, total_gilded, \
        total_controversiality, total_score = 0, 0, 0, 0, 0, 0
        morning_post_count, afternoon_post_count, evening_post_count, night_post_count = 0, 0, 0, 0
        for entry in features:
            total_controversiality += int(entry['controversiality'])
            total_awards += int(entry['total_awards'])
            total_score += int(entry['score'])
            total_gilded += int(entry['gilded'])
            morning_post_count, afternoon_post_count, evening_post_count, night_post_count = get_time_of_day_counts(
                features)
            if entry['submission_type'] == 'post':
                post_count += 1
            else:
                comment_count += 1

        row = {'username': user, 'total_controversiality': total_controversiality,
               'total_awards': total_awards, 'total_score': total_score, 'total_gilded': total_gilded,
               'morning_post_count': morning_post_count, 'afternoon_post_count': afternoon_post_count,
               'evening_post_count': evening_post_count, 'night_post_count': night_post_count}
        rows.append(row)

    return pd.DataFrame(rows)


def get_time_of_day_counts(features):
    counts = OrderedDict({'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0})

    for entry in features:
        result = get_time_of_day(entry['created'])
        counts[result] += 1

    return counts.values()


def get_time_of_day(created_utc):
    hour = datetime.fromtimestamp(int(created_utc), pytz.UTC).hour

    # Arbitrary cutoffs...
    if 5 <= hour <= 11:
        return 'morning'
    if 12 <= hour <= 16:
        return 'afternoon'
    if 17 <= hour <= 20:
        return 'evening'

    return 'night'


def run_collect(users, files, out_tsv):
    print(
        "Collecting user features for {} users over {} files and saving to {}".format(len(users), len(files), out_tsv))
    user_features = collect_user_submission_data(files, users)
    df = build_user_features_df(user_features)
    print("Saving features dataframe to TSV")
    df.to_csv(out_tsv, sep='\t')


if __name__ == '__main__':
    files = get_all_raw_files()
    users = get_all_political_users()
    print("Total number of users: {}".format(len(users)))
    run_collect(users, files,
                out_tsv='/shared/0/projects/reddit-political-affiliation/data/user-features/all_political.tsv')

    # Repeat for bad actors
    bad_actors = read_in_bad_actors_from_tsv(
        ['/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_365_days_1_flip_flop.tsv'])
    run_collect(bad_actors, files,
                out_tsv='/shared/0/projects/reddit-political-affiliation/data/user-features/bad_actors.tsv')
