import re
import sys
from collections import defaultdict
from glob import glob

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics, \
    parse_name_from_filepath, user_politics_to_tsv

""" Script to filter down existing comment matches (from comment_political_affiliations.py) with new criteria """


def filter_matches(comment_politics_files, exclude_matches, out_dir):
    for file in comment_politics_files:
        user_politics = read_in_user_politics([file])
        print("Starting number of political users: {} in file: {}".format(len(user_politics), file))
        user_politics_filtered = defaultdict(list)

        for user, political_entries in user_politics.items():
            filtered_entries = []
            for entry in political_entries:
                if not re.findall(exclude_matches, entry['text']):
                    filtered_entries.append(entry)

            if len(filtered_entries) > 0:
                user_politics_filtered[user] = filtered_entries

        print("Filtered number of political users: {} in file: {}".format(len(user_politics_filtered), file))
        out_file = out_dir + parse_name_from_filepath(file) + ".tsv"
        print("Saving file to: {}".format(out_file))
        user_politics_to_tsv(user_politics_filtered, out_file)


if __name__ == '__main__':
    gold_comment_files = glob('/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv')
    silver_comment_files = glob(
        '/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/*.tsv')

    # Remove was/former declarations
    matches_to_exclude = "i (was|used to be|used to vote)( a| as a)? (conservative|republican)|i (" \
                         "was|used to be|used to vote)( a| as a)? (liberal|democrat|progressive)|liberal arts"
    filter_matches(gold_comment_files, matches_to_exclude,
                   out_dir='/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/')
    filter_matches(silver_comment_files, matches_to_exclude,
                   out_dir='/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/')
