import glob
import json
import re
from collections import defaultdict

from src.data.date_helper import read_submissions

DEM_PATTERN = "(i am|i'm) a (democrat|liberal)|i vote[d]?( for| for a)? (democrat|hillary|biden|obama|blue)|i (" \
              "hate|despise) (conservatives|republicans|trump|donald trump|mcconell|mitch mcconell)|(i am|i'm) a (" \
              "former|ex) (conservative|republican)|(i am|i'm) an ex-(conservative|republican)|i (was|used to be|used " \
              "to vote)( a| as a)? (conservative|republican)|fuck (conservatives|republicans|donald " \
              "trump|trump|mcconell|mitch mcconell)"

REP_PATTERN = "((i am|i'm) a (conservative|republican)|i vote[d]?( for| for a)? (" \
              "republican|conservative|trump|romney|mcconell)|i (hate|despise) (" \
              "liberals|progressives|democrats|left-wing|biden|hillary obama)|(i am|i'm) a (former|ex) (" \
              "liberal|democrat|progressive)|(i am|i'm) an ex-(liberal|democrat|progressive)|i (was|used to be|used " \
              "to vote)( a| as a)? (liberal|democrat|progressive)|fuck (" \
              "liberals|progressives|democrats|biden|hillary|obama))"


def parse_submissions(fname, user_politics):
    """ Return a users subreddits with their associated politics(s) and timestamps """
    for submission in read_submissions(fname):
        text = get_submission_text(submission)
        username, created_utc = submission['author'], submission['created_utc']
        political_party = get_user_political_party(text)
        if political_party:
            user_politics[username].append((political_party, created_utc))

    return user_politics


def get_user_political_party(text):
    if re.findall(DEM_PATTERN, text.lower()):
        return "Democrat"
    elif re.findall(REP_PATTERN, text.lower()):
        return "Republican"
    return ""


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return text


def read_in_existing_politics(user_politics, in_file):
    print("Reading in existing user politics from file: {}".format(in_file))
    with open(in_file) as json_file:
        results = json.load(json_file)

    for user, political_data in results.items():
        user_politics[user] = political_data

    print("Total number of existing user politics is: {}".format(len(user_politics)))
    return user_politics


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    user_politics = defaultdict(list)
    out_path = '/shared/0/projects/reddit-political-affiliation/data/bad-actors/politics.json'
    user_politics = read_in_existing_politics(user_politics, out_path)

    for fname in files[19:]:
        print("Starting on file: {}".format(fname))
        user_politics = parse_submissions(fname, user_politics)

        # Save after every file just in case
        with open(out_path, 'w') as fp:
            json.dump(user_politics, fp)
