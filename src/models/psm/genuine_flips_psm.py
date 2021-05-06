import sys

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.bad_actors.bad_actors import get_genuine_actors
from src.features.collect_user_features import read_in_bad_actor_features

OUT_DIRECTORY = "/shared/0/projects/reddit-political-affiliation/data/user-features/"


def save_genuine_user_features():
    genuine_actors = get_genuine_actors()
    print("Total number of genuine actors: {}".format(len(genuine_actors)))

    bad_actors_features = read_in_bad_actor_features()
    print("Length of bad actor features: {}".format(len(bad_actors_features)))

    # Filter down features to the genuine users
    genuine_actors_features = bad_actors_features[bad_actors_features['username'].isin(genuine_actors)]
    print("Length of genuine actor features: {}".format(len(genuine_actors_features)))

    print("Saving features to TSV")
    genuine_actors_features.to_csv(OUT_DIRECTORY + 'genuine_actor_features.tsv', sep='\t', index=False)


if __name__ == '__main__':
    save_genuine_user_features()
