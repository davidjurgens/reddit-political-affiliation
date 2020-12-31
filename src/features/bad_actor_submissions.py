import argparse
import bz2
import glob
import json
import lzma
from collections import defaultdict
from json import JSONDecodeError

import zstd


def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


def get_submission_text(submission):
    text = ""
    if "body" in submission:
        text += submission['body'].lower()
    if "title" in submission:
        text += " " + submission['title'].lower()
    return text


def parse_bad_actor_comments(file_path, bad_actors):
    file_pointer = get_file_handle(file_path)
    bad_actor_comments = defaultdict(list)
    for count, line in enumerate(file_pointer):
        try:
            submission = json.loads(file_pointer.readline().strip())
            username = submission['author']
            if username in bad_actors:
                text = get_submission_text(submission)
                bad_actor_comments[username].append(text)

        except (JSONDecodeError, AttributeError) as e:
            print("Failed to parse line: {} with error: {}".format(line, e))

        if count % 1000000 == 0 and count > 0:
            print("Completed {} lines".format(count))

    return bad_actor_comments


def parse_zst_bad_actor_comments(filename, bad_actors):
    bad_actor_comments = defaultdict(list)
    count = 0

    with open(filename, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            while True:
                chunk = reader.read(1000000000)  # Read in 1GB at a time
                if not chunk:
                    break

                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")
                for i, line in enumerate(lines[:-1]):
                    try:
                        submission = json.loads(line)
                        username = submission['author']
                        if username in bad_actors:
                            text = get_submission_text(submission)
                            bad_actor_comments[username].append(text)

                    except Exception as e:
                        print("Failed to parse line: {} with error: {}".format(line, e))

                    count += 1
                    if count % 1000000 == 0 and count > 0:
                        print("Completed {} lines for file: {}".format(count, filename))

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
        extension = file.split('.')[-1]
        if extension == "zst":
            bad_actor_submissions = parse_zst_bad_actor_comments(file, bad_actors)
        else:
            bad_actor_submissions = parse_bad_actor_comments(file, bad_actors)

        fname = parse_name_from_filepath(file)
        out_file = args.out + fname + ".json"
        user_submissions_to_json(bad_actor_submissions, out_file)
