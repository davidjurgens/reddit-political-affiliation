import csv
import sys
from collections import defaultdict

from src.data.data_helper import read_submissions

"""  Script to parse out user flair information from Reddit comments and posts 
     
     Sample row of data
        - username: {subreddit1: [flair1, flair2], subreddit2: [flair3]}
"""


def parse_submissions(fname):
    """ Return a users subreddits with their associated flair(s) """

    user_flairs = defaultdict(lambda: defaultdict(list))
    for submission in read_submissions(fname):
        username, flair, subreddit = submission.username, submission.flair, submission.subreddit

        if flair and flair not in user_flairs[username][subreddit]:
            user_flairs[username][subreddit].append(flair)

    return user_flairs


def output_to_tsv(out_file, user_flairs):
    with open(out_file, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for username, subreddit_flairs in user_flairs.items():
            tsv_writer.writerow([username, dict(subreddit_flairs)])


if __name__ == '__main__':
    assert len(sys.argv) == 3, '2 input arguments required. (1) path to the input file (2) output directory to save TSV'

    in_file_path, out_dir = sys.argv[1], sys.argv[2]
    flair_data = parse_submissions(in_file_path)

    # Parse the file name from the full path and switch the extension to .tsv
    file_name = in_file_path.split('/')[-1]
    out = out_dir + file_name.split('.')[0] + '.tsv'

    print("Writing output for file {} to {}".format(in_file_path, out))
    output_to_tsv(out, flair_data)
