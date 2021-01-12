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
            line = line.strip()
            comment_id, parent_id, username, subreddit, created, politics = line.split('\t')
            political_comment = PoliticalComment(comment_id, parent_id, username, subreddit, created, politics)
            comments[comment_id] = political_comment

    print("Total number of political comments: {}".format(len(comments)))
    return comments


def get_political_comment_replies(raw_files, political_comments):
    for file in raw_files:
        for submission in read_submissions(file):
            comment_id, parent_id, subreddit, created_utc = submission['id'], submission['parent_id'], submission[
                'subreddit'], submission['created_utc']
            author, text = submission['author'], submission['body']
            text = " ".join(text.split())
            if author == '[deleted]' or author == 'AutoModerator':
                continue
            # Check if the parent of the comment is from a political user (i.e. a reply to a political comment)
            # t1 means it's a comment
            if parent_id[:2] == 't1':
                parent_comment_id = parent_id[3:]
                if parent_comment_id in political_comments:
                    if comment_id in political_comments:
                        politics = political_comments[comment_id].political_affiliation
                    else:
                        politics = "Unknown"
                    political_comment = PoliticalComment(comment_id, parent_id, author, subreddit, created_utc,
                                                         politics, text)
                    yield political_comment


def get_all_comments(raw_files):
    pol_comments = read_in_comment_data(
        '/shared/0/projects/reddit-political-affiliation/data/interactions/comment_ids.tsv')
    pol_reply_comments = read_in_comment_data(
        '/shared/0/projects/reddit-political-affiliation/data/interactions/child_comments.tsv')

    for file in raw_files:
        for submission in read_submissions(file):
            comment_id, parent_id, subreddit, created_utc, author, text = submission['id'], submission['parent_id'], \
                                                                    submission['subreddit'], submission['created_utc'], \
                                                                    submission['author'], submission['body']
            try:
                text = submission['body']
                text = " ".join(text.split())
                # Forgot to remove t1_ from some comments...

                if comment_id in pol_comments:
                    politics = pol_comments[comment_id].political_affiliation
                    yield PoliticalComment(comment_id, parent_id, author, subreddit, created_utc,
                                           politics, text)
                elif comment_id in pol_reply_comments:
                    politics = pol_reply_comments[comment_id].political_affiliation
                    yield PoliticalComment(comment_id, parent_id, author, subreddit, created_utc,
                                           politics, text)
            except Exception:
                pass

def remove_id_prefix_and_bots(in_file, bots):
    """ Go back through and do some post collection cleaning """
    comments = 0
    with open(in_file, 'r') as f:
        for line in f:
            comment_id, parent_id, username, subreddit, created, politics, text = line.split('\t')
            if username not in bots:
                if comment_id[:2] == 't1':
                    comment_id = comment_id[3:]
                if parent_id[:2] == 't1':
                    parent_id = parent_id[3:]

                comments += 1
                political_comment = PoliticalComment(comment_id, parent_id, username, subreddit, created, politics, text)
                yield political_comment

    print("Total number of political comments: {}".format(comments))


if __name__ == '__main__':
    comment_file = '/shared/0/projects/reddit-political-affiliation/data/interactions/comment_ids.tsv'
    comment_data = read_in_comment_data(comment_file)

    out_child_tsv = '/shared/0/projects/reddit-political-affiliation/data/interactions/child_comments.tsv'
    out_all_comments = '/shared/0/projects/reddit-political-affiliation/data/interactions/all_comments.tsv'
    raw_files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    raw_files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    raw_files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    # with open(out_child_tsv, 'w') as out_f:
    #     for pol_comment in get_political_comment_replies(raw_files, comment_data):
    #         out_f.write(pol_comment.to_tsv_row())

    with open(out_all_comments, 'w') as out_f:
        for pol_comment in get_all_comments(raw_files):
            out_f.write(pol_comment.to_tsv_row())
