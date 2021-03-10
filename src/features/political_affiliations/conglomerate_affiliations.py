import glob
import sys

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics
from src.features.political_affiliations.flair_political_affiliations import read_in_flair_affiliations

''' 
    Read in all of the political affiliations (silver, gold, flair) and create train dev and test sets with the joined
    data. Hardcoded directory paths because they are static 
'''


def grab_all_data_sources():
    silver_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/*.tsv")
    gold_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv")
    flair_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/*.tsv")

    print("Number of silver files: {}".format(silver_files))
    print("Number of gold files: {}".format(gold_files))
    print("Number of flair files: {}".format(flair_files))

    silver_data = read_in_user_politics(silver_files)
    gold_data = read_in_user_politics(gold_files)
    flair_data = read_in_flair_affiliations(flair_files)

    return silver_data, gold_data, flair_data


def build_df(silver, gold, flair):
    rows = []

    print("Working on silver data")
    for user, user_politics in silver.items():
        for entry in user_politics:
            row = {'username': user, 'source': 'silver', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            rows.append(row)

    print("Working on gold data")
    for user, user_politics in gold.items():
        for entry in user_politics:
            row = {'username': user, 'source': 'gold', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            rows.append(row)

    print("Working on flair data")
    for user, flair_data in flair.items():
        for entry in flair_data:
            row = {'username': user, 'source': 'flair', 'politics': entry['politics'], 'subreddit': entry['subreddit'],
                   'created': entry['created']}
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    out_directory = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/"
    silver, gold, flair = grab_all_data_sources()
    df = build_df(silver, gold, flair)
    df = shuffle(df)

    train, dev, test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
    print("Length of train data: {}".format(len(train)))
    print("Length of dev data: {}".format(len(dev)))
    print("Length of test data: {}".format(len(test)))

    train.to_csv(out_directory + "train.tsv", sep='\t')
    dev.to_csv(out_directory + "dev.tsv", sep='\t')
    test.to_csv(out_directory + "test.csv", sep='\t')
