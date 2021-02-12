import argparse
import glob
import re
import sys
from collections import *

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_regexes import *
from src.data.date_helper import read_submissions


def parse_comment_affiliations_silver_standard(file_path):
    user_politics = defaultdict(list)
    for submission in read_submissions(file_path):
        username, subreddit, created = submission['author'], submission['subreddit'], submission['created_utc']

        if username == '[deleted]':  # Can clean out the bots later ...
            continue

        text = get_submission_text(submission)
        # Ignore comments with quote replies
        if "&gt;" in text:
            continue

        dem_match, rep_match = False, False

        if re.findall(DEM_PATTERN_SILVER, text):
            entry = {'politics': 'Democrat', 'regex_match': 'dem', 'subreddit': subreddit, 'created': created,
                     'text': text}
            dem_match = True
        if re.findall(ANTI_REP_PATTERN_SILVER, text):
            entry = {'politics': 'Democrat', 'regex_match': 'anti_rep', 'subreddit': subreddit, 'created': created,
                     'text': text}
            dem_match = True
        if re.findall(REP_PATTERN_SILVER, text):
            entry = {'politics': 'Republican', 'regex_match': 'rep', 'subreddit': subreddit, 'created': created,
                     'text': text}
            rep_match = True
        if re.findall(ANTI_DEM_PATTERN_SILVER, text):
            entry = {'politics': 'Republican', 'regex_match': 'anti_dem', 'subreddit': subreddit, 'created': created,
                     'text': text}
            rep_match = True

        # Ignore comments that match both patterns
        if dem_match or rep_match and not (dem_match and rep_match):
            user_politics[username].append(entry)

    print("File completed! Total political users found: {}".format(len(user_politics)))
    return user_politics


def parse_comment_affiliations_gold_standard(file_path):
    user_politics = defaultdict(list)
    for submission in read_submissions(file_path):
        username, subreddit, created = submission['author'], submission['subreddit'], submission['created_utc']
        if username == '[deleted]':  # Can clean out the bots later ...
            continue

        text = get_submission_text(submission)
        # Ignore comments with quote replies
        if "&gt;" in text:
            continue

        dem_match, rep_match = False, False

        if re.findall(DEM_PATTERN_GOLD, text):
            entry = {'politics': 'Democrat', 'regex_match': 'dem', 'subreddit': subreddit, 'created': created,
                     'text': text}
            dem_match = True
        if re.findall(ANTI_REP_PATTERN_GOLD, text):
            entry = {'politics': 'Democrat', 'regex_match': 'anti_rep', 'subreddit': subreddit, 'created': created,
                     'text': text}
            dem_match = True
        if re.findall(REP_PATTERN_GOLD, text):
            entry = {'politics': 'Republican', 'regex_match': 'rep', 'subreddit': subreddit, 'created': created,
                     'text': text}
            rep_match = True
        if re.findall(ANTI_DEM_PATTERN_GOLD, text):
            entry = {'politics': 'Republican', 'regex_match': 'anti_dem', 'subreddit': subreddit, 'created': created,
                     'text': text}
            rep_match = True

        # Ignore comments that match both patterns
        if dem_match or rep_match and not (dem_match and rep_match):
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
                try:
                    user, politics, regex_match, subreddit, created, text = line.split('\t')
                    entry = {'politics': politics, 'regex_match': regex_match, 'subreddit': subreddit,
                             'created': created,
                             'text': text}
                    user_politics[user].append(entry)
                except Exception:
                    pass
    return user_politics


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return " ".join(text.split())


def count_regex_matches(in_files):
    match_counter = Counter()
    user_politics = read_in_user_politics(in_files)
    for user, user_politics in user_politics.items():
        for entry in user_politics:
            match_counter[entry['regex_match']] += 1

    for k, v in match_counter.items():
        print("Regex pattern {} has {} matches".format(k, v))


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
        user_politics = parse_comment_affiliations_gold_standard(file)
        fname = parse_name_from_filepath(file)
        out_file = args.out + 'gold/' + fname + ".tsv"
        user_politics_to_tsv(user_politics, out_file)

    in_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv")
    count_regex_matches(in_files)
