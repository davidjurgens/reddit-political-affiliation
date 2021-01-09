import glob
import sys
from collections import defaultdict

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.interactions.political_comment import PoliticalComment

""" 1st pass through to get comments of political users and associated metadata """


def get_political_user_comment_ids(files, political_users):
    for file in files:
        for submission in read_submissions(file):
            author = submission['author']
            if author in political_users and author != "[deleted]" and author != 'AutoModerator':
                comment_id, parent_id, subreddit, created_utc = submission['id'], submission['parent_id'], submission[
                    'subreddit'], submission['created_utc']
                text = submission['body']
                text = " ".join(text.split())
                politics = political_users[author]
                political_comment = PoliticalComment(comment_id, parent_id, author, subreddit, created_utc, politics, text)
                yield political_comment


def read_in_existing_politics(in_files):
    user_politics = defaultdict(list)

    for in_file in in_files:
        print("Reading in existing user politics from file: {}".format(in_file))
        with open(in_file, 'r') as f:
            for line in f:
                username, politics = line.split('\t')
                user_politics[username] = politics

    print("Total number of existing user politics is: {}".format(len(user_politics)))
    return user_politics


if __name__ == '__main__':
    # Read in user politics found from the comments
    political_affiliations_dir = '/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/*.tsv'
    files = glob.glob(political_affiliations_dir)
    pol_users = read_in_existing_politics(files)

    out_tsv = '/shared/0/projects/reddit-political-affiliation/data/interactions/comment_ids.tsv'

    raw_files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    raw_files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    raw_files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    # Run through all of the data and collect comments
    with open(out_tsv, 'w') as out_f:
        for comment in get_political_user_comment_ids(raw_files, pol_users):
            out_f.write(comment.to_tsv_row())
