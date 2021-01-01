import glob
import re

import pandas as pd

from src.data.date_helper import read_submissions

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


def get_samples(file_name, n=25):
    sample_rows = []
    sample_count = 0
    for submission in read_submissions(file_name):
        username, text, created_utc = submission['author'], submission['body'].lower(), submission['created_utc']

        if re.findall(DEM_PATTERN, text) or re.findall(ANTI_REP_PATTERN, text):
            row = {'file': file_name, 'politics': 'Democrat', 'comment': text, 'created_utc': created_utc}
            sample_rows.append(row)
            sample_count += 1
        elif re.findall(REP_PATTERN, text) or re.findall(ANTI_DEM_PATTERN, text):
            row = {'file': file_name, 'politics': 'Republican', 'comment': text, 'created_utc': created_utc}
            sample_rows.append(row)
            sample_count += 1

        if sample_count == n:
            return sample_rows

    return sample_rows


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


if __name__ == '__main__':

    out_file = 'political_samples.csv'

    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    samples = []

    for file in files[::-1]:
        print("Starting on file: {}".format(file))
        extension = file.split('.')[-1]
        fname = parse_name_from_filepath(file)
        samples.extend(get_samples(file))

        df = pd.DataFrame(samples, columns=["file", "politics", "comment", "created_utc"])
        df.to_csv(out_file)
