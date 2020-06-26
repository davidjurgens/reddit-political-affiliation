import sys
import csv
import bz2
import lzma
import json
from collections import defaultdict

"""  Script to parse out user flair information from Reddit comments and posts """


def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


def parse_submissions(file_handle):
    """ Return a users subreddits with their associated flair(s) """

    flair_count = 0
    user_flairs = defaultdict(lambda: defaultdict(set))

    for count, line in enumerate(file_handle):
        submission = json.loads(f.readline())
        username, flair, subreddit = submission['author'], submission['author_flair_text'], submission['subreddit']

        if flair:
            user_flairs[username][subreddit].add(flair)
            flair_count += 1

        if count % 1000000 == 0 and count > 0:
            print("Completed %d lines for file %s" % (count, file_handle))

    flair_percent = flair_count / float(count)
    return user_flairs, flair_percent


def output_to_tsv(out_file, user_flairs):
    with open(out_file, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for username, subreddit_flairs in user_flairs.items():
            tsv_writer.writerow([username, dict(subreddit_flairs)])


if __name__ == '__main__':
    assert len(sys.argv) == 3, '2 input arguments required. (1) path to the input file (2) output directory to save TSV'

    in_file_path, out_dir = sys.argv[1], sys.argv[2]

    print("Starting parse of file {}".format(in_file_path))
    f = get_file_handle(in_file_path)
    flair_data, flair_percent = parse_submissions(f)
    f.close()

    print("Percentage of users with a flair {} for file {}".format(flair_percent, in_file_path))
    print("Number of unique users for file {} is {}".format(in_file_path, len(list(flair_data.keys()))))

    file_name = in_file_path.split('.')[0]
    out = out_dir + file_name + ".tsv"
    print("Writing output for file {} to {}".format(in_file_path, out))
    output_to_tsv(out, flair_data)
