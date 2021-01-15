import argparse
import glob
import re
import sys
from collections import *

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions

DEM_PATTERN = "((i am|i'm) a (democrat|liberal)|i vote[d]?( for| for a)? (democrat|hillary|biden|obama|blue))"

ANTI_REP_PATTERN = "(i (hate|despise|loathe) (conservatives|republicans|trump|donald trump|mcconell|mitch mcconell)|" \
                   "(i am|i'm) a (former|ex) (conservative|republican)|(i am|i'm) an ex-(conservative|republican)|" \
                   "i (was|used to be|used to vote)( a| as a)? (conservative|republican)|" \
                   "fuck (conservatives|republicans|donald trump|trump|mcconell|mitch mcconell))"

REP_PATTERN = "((i am|i'm) a (conservative|republican)|i vote[d]?( for| for a)? (" \
              "republican|conservative|trump|romney|mcconell))"

ANTI_DEM_PATTERN = "(i (hate|despise) (liberals|progressives|democrats|left-wing|biden|hillary|obama)|(i am|i'm) a (" \
                   "former|ex) (liberal|democrat|progressive)|(i am|i'm) an ex-(liberal|democrat|progressive)|i (" \
                   "was|used to be|used to vote)( a| as a)? (liberal|democrat|progressive)|fuck (" \
                   "liberals|progressives|democrats|biden|hillary|obama))"


def parse_comment_affiliations(file_path):
    user_politics = defaultdict(list)
    for submission in read_submissions(file_path):
        username, subreddit, created = submission['author'], submission['subreddit'], submission['created_utc']
        if username == '[deleted]':  # Can clean out the bots later ...
            continue

        text = get_submission_text(submission)

        if re.findall(DEM_PATTERN, text):
            entry = {'politics': 'Democrat', 'regex_match': 'dem', 'subreddit': subreddit, 'created': created,
                     'text': text}
            user_politics[username].append(entry)
        elif re.findall(ANTI_REP_PATTERN, text):
            entry = {'politics': 'Democrat', 'regex_match': 'anti_rep', 'subreddit': subreddit, 'created': created,
                     'text': text}
            user_politics[username].append(entry)
        elif re.findall(REP_PATTERN, text):
            entry = {'politics': 'Republican', 'regex_match': 'rep', 'subreddit': subreddit, 'created': created,
                     'text': text}
            user_politics[username].append(entry)
        elif re.findall(ANTI_DEM_PATTERN, text):
            entry = {'politics': 'Republican', 'regex_match': 'anti_dem', 'subreddit': subreddit, 'created': created,
                     'text': text}
            user_politics[username].append(entry)

    print("File completed! Total political users found: {}".format(len(user_politics)))
    return user_politics


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


def user_politics_to_tsv(user_politics, out_file):
    print("Saving user politics to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        for user, user_politics in user_politics.items():
            for entry in user_politics:
                f.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(user, entry['politics'], entry['regex_match'], entry['subreddit'],
                                                      entry['created'], entry['text']))


def read_in_user_politics(in_files):
    user_politics = defaultdict(list)

    for in_file in in_files:
        print("Reading in user politics from file: {}".format(in_file))
        with open(in_file, 'r') as f:
            for line in f:
                user, politics, regex_match, subreddit, created, text = line.split('\t')
                entry = {'politics': 'Republican', 'regex_match': 'anti_dem', 'subreddit': subreddit, 'created': created,
                         'text': text}
                user_politics[user].append(entry)

    return user_politics


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return " ".join(text.split())


def filter_out_quote_text():
    """
        If a user is quoting another comment we should not consider that a political declaration
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get political affiliations from comments')
    # parser.add_argument('--dir', type=str, help="The directory of the raw/compressed reddit files to run on")
    parser.add_argument('--out_politics', type=str,
                        help="Output directory for the political affiliations and bad actors",
                        default="/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/")
    args = parser.parse_args()
    # files = glob.glob(args.dir)

    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    for file in files:
        print("Starting on file: {}".format(file))
        user_politics = parse_comment_affiliations(file)
        fname = parse_name_from_filepath(file)
        out_file = args.out_politics + fname + ".tsv"
        user_politics_to_tsv(user_politics, out_file)
