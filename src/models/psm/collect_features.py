import sys
from collections import Counter

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics


def count_user_posts(month_file, users):
    """
        month_file: Compressed data file for a single month of data
        users: should be a set or dict for O(1) lookups
    """
    user_counts = Counter()

    for submission in read_submissions(month_file):
        username = submission['author']
        if username in users:
            user_counts[username] += 1

    return user_counts


def get_political_user_counts(month_file):
    user_politics = read_in_user_politics([month_file])
    return count_user_posts(month_file, user_politics)


def output_counts_to_tsv(user_counts, out_file):
    pass


def create_features():
    """
        0 - non-political
        1 - political
    """
    pass


if __name__ == '__main__':
    pass
