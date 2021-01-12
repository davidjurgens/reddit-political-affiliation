import sys
import argparse
import glob
import re
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
    dem_count, anti_rep_count, rep_count, anti_dem_count = 0, 0, 0, 0
    for submission in read_submissions(file_path):
        username = submission['author']
        text = get_submission_text(submission)

        if re.findall(DEM_PATTERN, text):
            dem_count += 1
            user_politics[username].append("Democrat")
        elif re.findall(ANTI_REP_PATTERN, text):
            anti_rep_count += 1
            user_politics[username].append("Democrat")
        elif re.findall(REP_PATTERN, text):
            rep_count += 1
            user_politics[username].append("Republican")
        elif re.findall(ANTI_DEM_PATTERN, text):
            anti_dem_count += 1
            user_politics[username].append("Republican")

    print("File completed! Total political users found: {}".format(len(user_politics)))
    print("Democrat matches: {}. Anti-Republican matches: {}. Republican matches: {}. Anti-Democrat matches: {}"
          .format(dem_count, anti_rep_count, rep_count, anti_dem_count))
    return user_politics


def handle_bad_actors(user_politics, out_file):
    """ Find users who claim to be both democrats and republicans """
    good_actors, bad_actors = {}, []

    # Find the users who claim to be both
    for user, politics_list in user_politics.items():
        if len(set(politics_list)) == 1:
            good_actors[user] = politics_list[0]
        else:
            bad_actors.append(user)

    print("Total # of bad actors: {}".format(len(bad_actors)))

    # Save the bad actors to a file to a file
    with open(out_file, 'w') as f:
        for actor in bad_actors:
            f.write("{}\n".format(actor))

    # Return the filtered data
    return good_actors


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


def user_politics_to_tsv(user_politics, out_file):
    print("Saving user politics to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        for user, political_party in user_politics.items():
            f.write("{}\t{}\n".format(user, political_party))


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get political affiliations from comments')
    # parser.add_argument('--dir', type=str, help="The directory of the raw/compressed reddit files to run on")
    parser.add_argument('--out_politics', type=str,
                        help="Output directory for the political affiliations and bad actors",
                        default="/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/")
    parser.add_argument('--out_bad_actors', type=str,
                        help="Output directory for the political affiliations and bad actors",
                        default="/shared/0/projects/reddit-political-affiliation/data/bad-actors/")
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
        out_file_actors = args.out_bad_actors + fname + ".tsv"
        out_file = args.out_politics + fname + ".tsv"
        user_politics = handle_bad_actors(user_politics, out_file_actors)
        user_politics_to_tsv(user_politics, out_file)
