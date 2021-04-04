import glob
import sys
from collections import defaultdict

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.bad_actors.bad_actors import read_in_bad_actors_from_tsv


def get_bad_actor_submissions(file_path, bad_actors):
    bad_actor_comments = defaultdict(list)
    for submission in read_submissions(file_path):
        username = submission['author']
        if username in bad_actors:
            text = get_submission_text(submission)
            bad_actor_comments[username].append(text)

    return bad_actor_comments


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return " ".join(text.split())


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


def user_submissions_to_tsv(bad_actor_submissions, out_file):
    print("Saving user comments to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        for user, submissions in bad_actor_submissions.items():
            for submission in submissions:
                f.write("{}\t{}\n".format(user, submission))


def read_in_bad_actor_submissions(in_files):
    bad_actor_comments = defaultdict(list)

    for in_file in in_files:
        print("Reading in bad actor submissions from file: {}".format(in_file))
        with open(in_file, 'r') as f:
            for line in f:
                user, text = line.split('\t')
                bad_actor_comments[user].append(text)

    return bad_actor_comments


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    bad_actors_file = '/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors_365_days_1_' \
                      'flip_flop.tsv'
    submissions_out_dir = '/shared/0/projects/reddit-political-affiliation/data/bad-actors/submissions/'
    bad_actors = read_in_bad_actors_from_tsv([bad_actors_file])
    bad_actors = set(bad_actors.keys())
    print("Total number of bad actors: {}".format(len(bad_actors)))

    for file in files:
        print("Starting on file: {}".format(file))
        bad_actor_submissions = get_bad_actor_submissions(file, bad_actors)
        fname = parse_name_from_filepath(file)
        out_file = submissions_out_dir + fname + ".tsv"
        user_submissions_to_tsv(bad_actor_submissions, out_file)
