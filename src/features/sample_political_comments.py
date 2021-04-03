import glob
import re

import pandas as pd

from src.data.date_helper import read_submissions
from src.features.political_affiliations.comment_regexes import *

"""
    Script to grab samples of political comments for manual validation
"""


def get_samples(file_name, n=10):
    sample_rows = []
    sample_count = 0
    for submission in read_submissions(file_name):
        username, text, created_utc = submission['author'], submission['body'].lower(), submission['created_utc']
        text = text.lower()
        text = " ".join(text.split())
        if re.findall(DEM_PATTERN_GOLD, text) or re.findall(ANTI_REP_PATTERN_GOLD, text):
            row = {'file': file_name, 'politics': 'Democrat', 'comment': text, 'created_utc': created_utc}
            sample_rows.append(row)
            sample_count += 1
        elif re.findall(REP_PATTERN_GOLD, text) or re.findall(ANTI_DEM_PATTERN_GOLD, text):
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

    out_file = 'political_samples_gold.csv'

    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    samples = []

    for file in files:
        print("Starting on file: {}".format(file))
        extension = file.split('.')[-1]
        fname = parse_name_from_filepath(file)
        samples.extend(get_samples(file))

        df = pd.DataFrame(samples, columns=["file", "politics", "comment", "created_utc"])
        df.to_csv(out_file)
