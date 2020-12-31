import bz2
import glob
import json
import lzma
import re
from json import JSONDecodeError

import pandas as pd
import zstandard as zstd

"""
    Script to grab samples of political comments for manual validation
"""

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


def get_samples(file_path, file_name, n=25):
    sample_rows = []
    file_pointer = get_file_handle(file_path)
    sample_count = 0
    for count, line in enumerate(file_pointer):
        try:
            submission = json.loads(file_pointer.readline().strip())
            username, text, created_utc = submission['author'], submission['body'].lower(), submission['created_utc']

            if re.findall(DEM_PATTERN, text) or re.findall(ANTI_REP_PATTERN, text):
                row = {'file': file_name, 'politics': 'Democrat', 'comment': text, 'created_utc': created_utc}
                sample_rows.append(row)
                sample_count += 1
            elif re.findall(REP_PATTERN, text) or re.findall(ANTI_DEM_PATTERN, text):
                row = {'file': file_name, 'politics': 'Republican', 'comment': text, 'created_utc': created_utc}
                sample_rows.append(row)
                sample_count += 1
        except (JSONDecodeError, AttributeError) as e:
            print("Failed to parse line: {} with error: {}".format(line, e))

        if sample_count == n:
            return sample_rows

    return sample_rows


def get_samples_zst(file_path, file_name, n=25):
    sample_rows = []
    sample_count = 0
    with open(file_path, 'rb') as f:
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
                        username, text, created_utc = submission['author'], submission['body'].lower(), submission[
                            'created_utc']

                        if re.findall(DEM_PATTERN, text) or re.findall(ANTI_REP_PATTERN, text):
                            row = {'file': file_name, 'politics': 'Democrat', 'comment': text,
                                   'created_utc': created_utc}
                            sample_rows.append(row)
                            sample_count += 1
                        elif re.findall(REP_PATTERN, text) or re.findall(ANTI_DEM_PATTERN, text):
                            row = {'file': file_name, 'politics': 'Republican', 'comment': text,
                                   'created_utc': created_utc}
                            sample_rows.append(row)
                            sample_count += 1
                    except Exception as e:
                        print("Failed to parse line: {} with error: {}".format(line, e))

                    if sample_count == n:
                        return sample_rows

    return sample_rows


def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


if __name__ == '__main__':

    out_file = 'political_samples.csv'

    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend('/shared/2/datasets/reddit-dump-all/RS/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    samples = []

    for file in files:
        print("Starting on file: {}".format(file))
        extension = file.split('.')[-1]
        fname = parse_name_from_filepath(file)
        if extension == "zst":
            samples.extend(get_samples_zst(file, fname))
        else:
            samples.extend(get_samples(file, fname))

        df = pd.DataFrame(samples, columns=["file", "politics", "comment", "created_utc"])
        df.to_csv(out_file)
