import sys
from collections import defaultdict, OrderedDict
from datetime import datetime

import pandas as pd
import pytz

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.data_helper import read_submissions, get_all_raw_files
from src.features.behavioral_features.collect_samples import collect_non_political_usernames
from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users
from src.features.political_affiliations.flair_political_affiliations import get_all_flair_users
from src.features.political_affiliations.comment_political_affiliations import get_all_gold_users
from src.features.political_affiliations.community_labels import get_all_community_users
from src.features.behavioral_features.subreddit_constants import top_subreddits, political_subreddits
from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv

OUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/user-features/"


def collect_user_submission_data(month_file, political_users, bad_actors, non_political_users, pol_user_features,
                                 bad_actor_features, non_pol_user_features):
    print("Collecting user features for month: {}".format(month_file))
    # Otherwise lookups take forever....
    assert type(political_users) == set and type(bad_actors) == set and type(non_political_users) == set

    for submission in read_submissions(month_file):
        # Check from smallest to largest
        if submission.username in bad_actors:
            bad_actor_features[submission.username].append(get_submission_metadata(submission))
        elif submission.username in non_political_users:
            non_pol_user_features[submission.username].append(get_submission_metadata(submission))
        elif submission.username in political_users:
            pol_user_features[submission.username].append(get_submission_metadata(submission))

    return pol_user_features, bad_actor_features, non_pol_user_features


def get_submission_metadata(submission):
    submission_type = 'post' if submission.is_post() else 'comment'
    top_subreddit = 1 if submission.subreddit in top_subreddits else 0
    pol_subreddit = 1 if submission.subreddit in political_subreddits else 0

    return {'username': submission.username, 'subreddit': submission.subreddit, 'score': submission.score,
            'submission_type': submission_type, 'gilded': submission.gilded, 'created': submission.created,
            'total_awards': submission.total_awards, 'controversiality': submission.controversiality,
            'top_subreddit': top_subreddit, 'political_subreddit': pol_subreddit}


def build_user_features_df(user_features, source):
    print("Building dataframe for {} user features".format(source))
    print("Total of {} users".format(len(user_features)))
    rows = []

    # Loop through the user posts and tally everything up
    for user, features in user_features.items():
        post_count, comment_count, total_awards, total_gilded, \
        total_controversiality, total_score = 0, 0, 0, 0, 0, 0
        top_subreddit_participation, pol_subreddit_participation = 0, 0
        time_of_day_counts = OrderedDict({'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0})

        for entry in features:
            total_controversiality += int(entry['controversiality'])
            total_awards += int(entry['total_awards'])
            total_score += int(entry['score'])
            total_gilded += int(entry['gilded'])
            time_of_day_counts[get_time_of_day(entry['created'])] += 1
            top_subreddit_participation += entry['top_subreddit']
            pol_subreddit_participation += entry['political_subreddit']

            if entry['submission_type'] == 'post':
                post_count += 1
            else:
                comment_count += 1

        row = {'username': user, 'total_controversiality': total_controversiality,
               'total_awards': total_awards, 'total_score': total_score, 'total_gilded': total_gilded,
               'morning_post_count': time_of_day_counts['morning'],
               'afternoon_post_count': time_of_day_counts['afternoon'],
               'evening_post_count': time_of_day_counts['evening'], 'night_post_count': time_of_day_counts['night'],
               'post_count': post_count, 'comment_count': comment_count,
               'top_subreddit_participation': top_subreddit_participation,
               'political_subreddit_participation': pol_subreddit_participation}
        rows.append(row)

    return pd.DataFrame(rows)


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


def run_collect(political_users, bad_actors, non_political_users, files):
    print("Total number of political users: {}".format(len(political_users)))
    print("Total number of bad actor users: {}".format(len(bad_actors)))
    print("Total number of non political users: {}".format(len(non_political_users)))

    pol_user_features, bad_actor_features, non_pol_user_features = defaultdict(list), defaultdict(list), defaultdict(
        list)

    for m_file in files:
        pol_user_features, bad_actor_features, non_pol_user_features = \
            collect_user_submission_data(m_file, political_users, bad_actors, non_political_users, pol_user_features,
                                         bad_actor_features, non_pol_user_features)

        pol_df = build_user_features_df(pol_user_features, 'political')
        bad_actors_df = build_user_features_df(bad_actor_features, 'bad actor')
        non_pol_df = build_user_features_df(non_pol_user_features, 'non-political')

        print("Saving political user features dataframe to TSV")
        pol_df.to_csv(OUT_DIRECTORY + 'all_political.tsv', sep='\t', index=False)

        print("Saving bad actor user features dataframe to TSV")
        bad_actors_df.to_csv(OUT_DIRECTORY + 'bad_actors.tsv', sep='\t', index=False)

        print("Saving non political user features dataframe to TSV")
        non_pol_df.to_csv(OUT_DIRECTORY + 'non_political.tsv', sep='\t', index=False)


def get_political_user_features(source):
    assert source == 'flair' or source == 'gold' or source == 'community'
    df = pd.read_csv(OUT_DIRECTORY + 'all_political.tsv', sep='\t')

    if source == 'flair':
        flair_users = get_all_flair_users()
        return df[df['username'].isin(flair_users)]
    elif source == 'gold':
        gold_users = get_all_gold_users()
        return df[df['username'].isin(gold_users)]
    else:
        community_users = get_all_community_users()
        return df[df['username'].isin(community_users)]


def read_in_non_pol_user_features():
    in_file = '/shared/0/projects/reddit-political-affiliation/data/user-features/all_non_political.tsv'
    return pd.read_csv(in_file, sep='\t', index_col=0)


def read_in_pol_user_features():
    in_file = '/shared/0/projects/reddit-political-affiliation/data/user-features/all_political.tsv'
    return pd.read_csv(in_file, sep='\t', index_col=0)


def read_in_bad_actor_features():
    in_file = '/shared/0/projects/reddit-political-affiliation/data/user-features/bad_actors.tsv'
    return pd.read_csv(in_file, sep='\t', index_col=0)


if __name__ == '__main__':
    files = get_all_raw_files()

    print("Reading in political users")
    all_political_users = get_all_political_users()
    print("Total number of political users: {}".format(len(all_political_users)))

    print("Reading in bad actors")
    bad_actors = set(read_in_bad_actors_from_tsv(
        ['/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_90_days_1_flip_flop.tsv']).keys())
    print("Total number of bad actors: {}".format(len(bad_actors)))

    print("Collecting random sample of non political users")
    non_political_usernames = collect_non_political_usernames(get_all_raw_files(), all_political_users, count=10000)
    print("Total of {} non political users".format(len(non_political_usernames)))

    run_collect(all_political_users, bad_actors, non_political_usernames, files)
