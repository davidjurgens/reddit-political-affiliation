import sys
import glob
from collections import Counter, defaultdict

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

    for submission in read_submissions(month_file):
        username, subreddit, created = submission['author'], submission['subreddit'], submission['created_utc']
        if username in users:
            submission_type = 'post' if 'num_comments' in submission else 'comment'
            entry = {'username': username, 'type': submission_type, 'subreddit': subreddit, 'created': created}
            user_features[username].append(entry)
    return user_features


def get_post_counts(user_features, user):
    post_counts = 0
    for features in user_features[user]:
        for entry in features:
            if entry['type'] == 'post':
                post_counts += 1
    return post_counts


def get_comment_counts(user_features, user):
    comment_counts = 0
    for features in user_features[user]:
        for entry in features:
            if entry['type'] == 'comment':
                comment_counts += 1
    return comment_counts


def get_participation_in_top_subreddits(user_features, top_subreddits, user):
    submission_count = 0

    for features in user_features[user]:
        for entry in features:
            if entry['subreddit'] in top_subreddits:
                submission_count += 1
    return submission_count


def output_counts_to_tsv(user_counts, out_file):
    pass


def create_features():
    """
        0 - non-political
        1 - political
    """
    pass


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))
    collect_user_features(files[0], [])
    collect_user_features(files[-1], [])
