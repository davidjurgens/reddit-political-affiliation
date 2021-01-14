import glob
import json
import sys
from collections import *

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.political_labels import labels

"""

Get all unique political flairs of a user and when/where they first appeared

Sample output entry (dict)

"username" : [
    {¬
        "flair": "AMERICA FIRST",
        "politics": "Republican"
        "subreddit": "the_donald",
        "created": 1610423222
    },
    {...}
]

"""


def parse_flair_affiliations(raw_files, out_directory):
    user_flairs = defaultdict(list)

    for file in raw_files:
        for sub in read_submissions(file):
            username, flair, created, subreddit = sub['author'], sub['author_flair_text'], sub['created_utc'], sub[
                'subreddit']
            if flair in labels:
                if is_new_flair(user_flairs[username], flair):
                    entry = {'flair': flair, 'politics': labels[flair], 'subreddit': subreddit, 'created': created}
                    user_flairs[username].append(entry)

        out_path = out_directory + parse_year_month_from_filename(file)
        output_to_json(user_flairs, out_path)

    return user_flairs


def is_new_flair(user_data, flair):
    """ Check if this is a new political flair for the user """
    for flair_entry in user_data:
        if flair_entry['flair'] == flair:
            return False
    return True


def parse_year_month_from_filename(filename):
    name = filename.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[0]


def output_to_json(user_flairs, out_file):
    print("Saving user flairs to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        json.dump(f, user_flairs)


def read_in_flair_affiliations(in_file):
    with open(in_file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    out_dir = '/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/'

    parse_flair_affiliations(files, out_dir)
