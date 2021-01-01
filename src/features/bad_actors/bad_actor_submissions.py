import argparse
import glob
import json
from collections import defaultdict


def get_submission_text(submission):
    text = ""
    if "body" in submission:
        text += submission['body'].lower()
    if "title" in submission:
        text += " " + submission['title'].lower()
    return text


def parse_bad_actor_comments(file_path, bad_actors):
    bad_actor_comments = defaultdict(list)
    for submission in read_in_bad_actors(file_path):
        username = submission['author']
        if username in bad_actors:
            text = get_submission_text(submission)
            bad_actor_comments[username].append(text)

    return bad_actor_comments


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


def user_submissions_to_json(bad_actors, out_file):
    print("Saving user comments to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        json.dump(bad_actors, f)


def read_in_bad_actors(in_file):
    with open(in_file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get political affiliations from comments')
    parser.add_argument('--bad_actors', type=str, help="Input file containing the list of bad actors",
                        default="/shared/0/projects/reddit-political-affiliation/data/bad-actors/bad_actors.json")
    parser.add_argument('--out', type=str, help="Output directory for the bad actor comments",
                        default="/shared/0/projects/reddit-political-affiliation/data/bad-actors/comments/")
    args = parser.parse_args()
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    bad_actors = read_in_bad_actors(args.bad_actors)

    for file in files:
        print("Starting on file: {}".format(file))
        bad_actor_submissions = parse_bad_actor_comments(file, bad_actors)
        fname = parse_name_from_filepath(file)
        out_file = args.out + fname + ".json"
        user_submissions_to_json(bad_actor_submissions, out_file)
