import sys
import csv
import bz2
import lzma
import json
from collections import defaultdict

"""  Script to parse out user flair information from Reddit comments and posts """

assert len(sys.argv) == 3, '2 input arguments required. (1) path to the input file (2) output directory to save TSV'

in_file, out_path = sys.argv[1], sys.argv[2]
print("Starting parse of file {}".format(in_file))
flair_count = 0
user_flairs = defaultdict(lambda: defaultdict(set))

if in_file[-3:] == "bz2":
    f = bz2.open(in_file)
elif in_file[-2:] == "xz":
    f = lzma.open(in_file)
else:
    raise AssertionError("Invalid extension for " + in_file + ". Expecting bz2 or xz file")

for count, line in enumerate(f):
    submission = json.loads(f.readline())
    username, flair, subreddit = submission['author'], submission['author_flair_text'], submission['subreddit']

    if flair:
        user_flairs[username][subreddit].add(flair)
        flair_count += 1

    if count % 1000000 == 0:
        print("Completed %d lines for file %s" % (count, in_file))

f.close()

flair_percent = flair_count / float(count)
print("Percentage of users with a flair {} for file {}".format(flair_percent, in_file))
print("Number of unique users for file {} is {}".format(in_file, len(list(user_flairs.keys()))))

with open(out_path + in_file, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for username, subreddit_flairs in user_flairs.items():
        tsv_writer.writerow([username, dict(subreddit_flairs)])
