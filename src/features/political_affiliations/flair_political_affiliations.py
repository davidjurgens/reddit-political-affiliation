import sys
import argparse
import glob
import re
from collections import *

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions

"""

Sample output entry (dict)

"username" : [
    {¬
        "flair": "Republican",
        "subreddit": "the_donald",
        "created": 1610423222
    },
    {...}
]

"""


def parse_flair_affiliations(raw_files):
    user_flairs = defaultdict(list)

    for file in raw_files:
        for submission in read_submissions(file):
            pass


def is_new_flair(user_data, flair):
    """ Check if this is a new political flair for the user """
    for flair_info in user_data:
        if flair_info['flair'] == flair:
            return False
    return True


def output_to_json():
    pass


def read_in_flair_affiliations():
    pass


if __name__ == '__main__':
    pass