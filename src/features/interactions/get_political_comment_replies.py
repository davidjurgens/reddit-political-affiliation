import glob
import sys

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.interactions.political_comment import PoliticalComment

""" 2nd pass through. Find child or parent comments of comments made by political users """


def read_in_comment_data(in_file):
    comments = {}
    with open(in_file, 'r') as f:
        for line in f:
            comment_id, parent_id, username, subreddit, created, politics = line.split('\t')
            political_comment = PoliticalComment(comment_id, parent_id, username, subreddit, created, politics)
            comments[comment_id] = political_comment
    return comments


def get_political_comment_replies(raw_files, political_comments):
    for file in raw_files:
        for submission in read_submissions(file):
            comment_id, parent_id, subreddit, created_utc = submission['id'], submission['parent_id'], submission[
                'subreddit'], submission['created_utc']

            # Check if the parent of the comment is from a political user (i.e. a reply to a political comment)
            # t1 means it's a comment
            if parent_id[:2] == 't1':
                parent_comment_id = parent_id[3:]
                if parent_comment_id in political_comments:
                    author = submission['author']
                    politics = political_comments[parent_comment_id].political_affiliation
                    political_comment = PoliticalComment(comment_id, parent_id, author, subreddit, created_utc,
                                                         politics)
                    yield political_comment


if __name__ == '__main__':
    comment_file = '/shared/0/projects/reddit-political-affiliation/data/interactions/comment_ids.tsv'
    comment_data = read_in_comment_data(comment_file)

    out_tsv = '/shared/0/projects/reddit-political-affiliation/data/interactions/child_comments.tsv'
    raw_files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    raw_files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    raw_files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    with open(out_tsv, 'w') as out_f:
        for pol_comment in get_political_comment_replies(raw_files, comment_data):
            out_f.write(pol_comment.to_tsv_row())
