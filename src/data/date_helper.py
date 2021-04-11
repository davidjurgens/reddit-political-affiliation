import bz2
import io
import json
import lzma
import sys
from glob import glob
from json import JSONDecodeError

import zstandard as zstd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.reddit_submission import Submission

"""
    Functions to work with the raw compressed Reddit data
"""


def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)
    elif ext == "zst":
        f = open(file_path, 'rb')
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        return text_stream

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


def read_submissions(file_name):
    """ Read the JSON submissions from the raw compressed files. Returns w/ generator """
    file_pointer = get_file_handle(file_name)
    for count, line in enumerate(file_pointer):
        try:
            submission_json = json.loads(file_pointer.readline().strip())
            submission = Submission(submission_json)
            yield submission

        except (JSONDecodeError, AttributeError) as e:
            print("Failed to parse line: {} with error: {}".format(line, e))

        if count % 1000000 == 0 and count > 0:
            print("Completed {} lines for file {}".format(count, file_name))

    file_pointer.close()


def get_all_raw_files():
    print("Reading in raw file names")
    files = glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))
    print("Total of {} files".format(len(files)))
    return files
