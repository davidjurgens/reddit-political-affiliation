import glob
import itertools

import pandas as pd

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics
from src.features.political_affiliations.conglomerate_affiliations import shuffle_dict_keys

"""
    Script to grab samples of political comments for manual validation
"""


def get_samples(file_name, n=10):
    samples = []
    user_politics = read_in_user_politics(in_files=[file_name])
    user_politics = shuffle_dict_keys(user_politics)
    n_users = dict(itertools.islice(user_politics.items(), n))
    # Grab the first match from each user
    for user, user_politics in n_users.items():
        print(user_politics[0])
        samples.append(user_politics[0])

    return samples


if __name__ == '__main__':

    out_file = 'political_samples_gold.csv'
    files = glob.glob('/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv')
    samples = []

    for file in files[-3:]:
        print("Starting on file: {}".format(file))
        samples.extend(get_samples(file, n=50))
        print(len(samples))

    df = pd.DataFrame(samples, columns=["politics", "match", "created", "text"])
    print(df.head())
    df.to_csv(out_file)
