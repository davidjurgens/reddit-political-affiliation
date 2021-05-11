import sys
from collections import defaultdict, OrderedDict
from datetime import datetime

import pandas as pd
import pytz

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.data_helper import read_submissions, get_all_raw_files
from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users
from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv
from src.features.behavioral_features.collect_samples import read_usernames_from_tsv
from src.features.psm.features import top_subreddits, political_subreddits

""" Script to collect behavioral user features like number of posts or total karma """


def collect_user_submission_data(month_file, pol_users, non_pol_users, bad_actors, pol_user_features,
                                 non_pol_user_features, bad_actor_features):
    print("Collecting user features for month: {}".format(month_file))

    for submission in read_submissions(month_file):
        if submission.username in pol_users:
            pol_user_features[submission.username].append(get_submission_metadata(submission))
        elif submission.username in non_pol_users:
            non_pol_user_features[submission.username].append(get_submission_metadata(submission))
        elif submission.username in bad_actors:
            bad_actor_features[submission.username].append(get_submission_metadata(submission))

    return pol_user_features, non_pol_user_features, bad_actor_features


def get_submission_metadata(submission):
    submission_type = 'post' if submission.is_post() else 'comment'
    top_subreddit = 1 if submission.subreddit in top_subreddits else 0
    pol_subreddit = 1 if submission.subreddit in political_subreddits else 0

    return {'username': submission.username, 'subreddit': submission.subreddit, 'score': submission.score,
            'submission_type': submission_type, 'gilded': submission.gilded, 'created': submission.created,
            'total_awards': submission.total_awards, 'controversiality': submission.controversiality,
            'top_subreddit': top_subreddit, 'political_subreddit': pol_subreddit}


def build_user_features_df(user_features):
    print("Building dataframe for user features")
    print("Total of {} users".format(len(user_features)))
    rows = []

    # Loop through the user posts and tally everything up
    for user, features in user_features.items():
        post_count, comment_count, total_awards, total_gilded, \
        total_controversiality, total_score = 0, 0, 0, 0, 0, 0
        top_subreddit_participation, pol_subreddit_participation = 0, 0
        morning_post_count, afternoon_post_count, evening_post_count, night_post_count = 0, 0, 0, 0
        for entry in features:
            total_controversiality += int(entry['controversiality'])
            total_awards += int(entry['total_awards'])
            total_score += int(entry['score'])
            total_gilded += int(entry['gilded'])
            top_subreddit_participation += entry['top_subreddit']
            pol_subreddit_participation += entry['political_subreddit']
            morning_post_count, afternoon_post_count, evening_post_count, night_post_count = get_time_of_day_counts(
                features)
            if entry['submission_type'] == 'post':
                post_count += 1
            else:
                comment_count += 1

        row = {'username': user, 'total_controversiality': total_controversiality, 'total_posts': post_count,
               'total_comments': comment_count, 'total_awards': total_awards, 'total_score': total_score,
               'total_gilded': total_gilded, 'morning_post_count': morning_post_count, 'afternoon_post_count':
                   afternoon_post_count, 'evening_post_count': evening_post_count, 'night_post_count': night_post_count,
               'top_subreddit_participation': top_subreddit_participation,
               'political_subreddit_participation': pol_subreddit_participation}
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


def run_collect(pol_users, non_pol_users, bad_actors, files, out_dir):
    pol_user_features, non_pol_user_features, bad_actor_features = defaultdict(list), defaultdict(list), defaultdict(
        list)
    for m_file in files[::-1]:
        pol_user_features, non_pol_user_features, bad_actor_features = \
            collect_user_submission_data(m_file, pol_users, non_pol_users, bad_actors, pol_user_features,
                                         non_pol_user_features, bad_actor_features)
        pol_df = build_user_features_df(pol_user_features)
        non_pol_df = build_user_features_df(non_pol_user_features)
        bad_actors_df = build_user_features_df(bad_actor_features)
        print("Saving feature dataframes to TSV")
        pol_df.to_csv(out_dir + 'all_political.tsv', sep='\t', index=False)
        non_pol_df.to_csv(out_dir + 'all_non_political.tsv', sep='\t', index=False)
        bad_actors_df.to_csv(out_dir + 'bad_actors.tsv', sep='\t', index=False)


def read_in_non_pol_user_features():
    in_file = '/shared/0/projects/reddit-political-affiliation/data/user-features/all_non_political.tsv'
    df = pd.read_csv(in_file, sep='\t', index_col=0)
    df = df.fillna(0)
    return df


def read_in_pol_user_features():
    in_file = '/shared/0/projects/reddit-political-affiliation/data/user-features/all_political.tsv'
    df = pd.read_csv(in_file, sep='\t', index_col=0)
    df = df.fillna(0)
    return df


def read_in_bad_actor_features():
    in_file = '/shared/0/projects/reddit-political-affiliation/data/user-features/bad_actors.tsv'
    df = pd.read_csv(in_file, sep='\t', index_col=0)
    df = df.fillna(0)
    return df


if __name__ == '__main__':
    files = get_all_raw_files()

    political_users = get_all_political_users()
    print("Total number of political users: {}".format(len(political_users)))

    bad_actors = read_in_bad_actors_from_tsv(
        ['/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_90_days_1_flip_flop.tsv'])

    non_political_users = read_usernames_from_tsv(
        '/shared/0/projects/reddit-political-affiliation/data/sample-submissions/non_political_usernames.tsv')
    print("Total of {} non political users".format(len(non_political_users)))

    run_collect(political_users, non_political_users, bad_actors, files,
                out_dir='/shared/0/projects/reddit-political-affiliation/data/user-features/')
