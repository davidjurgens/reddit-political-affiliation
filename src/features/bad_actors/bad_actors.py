import bz2
import glob
import json
import lzma
import re
from collections import defaultdict
from json import JSONDecodeError

import zstd

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
    file_pointer = get_file_handle(fname)
    for count, line in enumerate(file_pointer):
        try:
            submission = json.loads(file_pointer.readline().strip())
            text = get_submission_text(submission)
            username, created_utc = submission['author'], submission['created_utc']
            political_party = get_user_political_party(text)
            if political_party:
                user_politics[username].append((political_party, created_utc))

        except (JSONDecodeError, AttributeError) as e:
            print("Failed to parse line: {} with error: {}".format(line, e))

        if count % 1000000 == 0 and count > 0:
            print("Completed %d lines" % count)

    return user_politics


def parse_zst_submissions(filename, user_politics):
    """ Return a users subreddits with their associated politics(s) and timestamps """
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
                        username, created_utc = submission['author'], submission['created_utc']
                        text = get_submission_text(submission)
                        political_party = get_user_political_party(text)

                        if political_party:
                            user_politics[username].append((political_party, created_utc))

                    except Exception as e:
                        print("Failed to parse line: {} with error: {}".format(line, e))

                    count += 1
                    if count % 1000000 == 0 and count > 0:
                        print("Completed {} lines for file: {}. Number of political users so far: {}"
                              .format(count, filename, len(user_politics)))

    print("File completed! Total political users found: {}".format(len(user_politics)))
    return user_politics


def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


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
        extension = fname.split('.')[-1]
        if extension == "zst":
            user_politics = parse_zst_submissions(fname, user_politics)
        else:
            user_politics = parse_submissions(fname, user_politics)

        # Save after every file just in case
        with open(out_path, 'w') as fp:
            json.dump(user_politics, fp)
