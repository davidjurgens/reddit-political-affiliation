import sys
from os import path
import pandas as pd
from glob import glob

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users
from src.features.bad_actors.bad_actors import read_in_bad_actor_usernames

OUT_DIR = '/shared/0/projects/reddit-political-affiliation/data/sample-submissions/'


def collect_submissions(raw_files, pol_users, non_pol_users, bad_actors, count):
    """ Grab posts or comments for a subset of users """
    # Sample evenly across the files
    samples_per_file = int(count / len(raw_files))
    pol_samples, non_pol_samples, bad_actor_samples = [], [], []

    for m_file in raw_files[::-1]:
        m_pol_samples, m_non_pol_samples, m_bad_actor_samples = get_samples_from_file(m_file, pol_users, non_pol_users,
                                                                                      bad_actors, samples_per_file)
        pol_samples.extend(m_pol_samples)
        non_pol_samples.extend(m_non_pol_samples)
        bad_actor_samples.extend(m_bad_actor_samples)
        output_samples_to_tsv(pol_samples, OUT_DIR + 'political_samples.tsv')
        output_samples_to_tsv(non_pol_samples, OUT_DIR + 'non_political_samples.tsv')
        output_samples_to_tsv(bad_actor_samples, OUT_DIR + 'bad_actor_samples.tsv')

    return pol_samples, non_pol_samples, bad_actor_samples


def get_samples_from_file(raw_file, pol_users, non_pol_users, bad_actors, count):
    print("Collecting {} from {}".format(count, raw_file))
    pol_samples, non_pol_samples, bad_actor_samples = [], [], []

    for submission in read_submissions(raw_file):
        user = submission.username
        subreddit, created, text = submission.subreddit, submission.created, submission.text
        entry = {'username': user, 'subreddit': subreddit, 'created': created, 'text': text}

        if user in pol_users and len(pol_samples) < count:
            pol_samples.append(entry)
        elif user in non_pol_users and len(non_pol_samples) < count:
            non_pol_samples.append(entry)
        elif user in bad_actors and len(bad_actor_samples) < count:
            bad_actor_samples.append(entry)

        if len(pol_samples) >= count and len(non_pol_samples) >= count and len(bad_actor_samples) >= count:
            return pol_samples, non_pol_samples, bad_actor_samples

    # Run out of data before hitting the count
    return pol_samples, non_pol_samples, bad_actor_samples


def collect_non_political_usernames(raw_files, political_users, count):
    assert type(political_users) is dict or type(political_users) is set
    usernames_per_file = int(count / len(raw_files))
    usernames = set()

    for m_file in raw_files:
        non_political_users = sample_non_political_users(m_file, political_users, usernames_per_file)
        usernames.update(non_political_users)

    return usernames


def sample_non_political_users(month_file, political_users, count):
    assert type(political_users) is dict or type(political_users) is set
    users = set()
    print("Sampling {} non political users from: {}".format(count, month_file))

    for submission in read_submissions(month_file):
        username = submission.username

        if username not in political_users:
            users.add(username)
            if len(users) % 100000 == 0 and len(users) > 0:
                print("Completed sampling {} non political users from file: {}".format(len(users), month_file))

        if len(users) >= count:
            return users

    # If we run out of usernames
    return users


def output_samples_to_tsv(samples, out_tsv):
    print("Writing {} samples to TSV: {}".format(len(samples), out_tsv))
    with open(out_tsv, 'w') as f:
        for s in samples:
            f.write("{}\t{}\t{}\t{}\n".format(s['username'], s['subreddit'], s['created'], s['text']))


def usernames_to_tsv(users, out_tsv):
    print("Saving {} users to {}".format(len(users), out_tsv))
    with open(out_tsv, 'w') as f:
        for user in users:
            f.write("{}\n".format(user))


def read_usernames_from_tsv(in_tsv):
    users = set()
    with open(in_tsv, 'r') as f:
        for line in f.readlines():
            users.add(line.strip())

    return users


def read_samples_from_tsv(in_tsv):
    samples = []
    with open(in_tsv, 'r') as f:
        for line in f.readlines():
            try:
                username, subreddit, created, text = line.split('\t')
                entry = {'username': username, 'subreddit': subreddit, 'created': created, 'text': text}
                samples.append(entry)
            except Exception:
                pass
    return samples


def get_political_samples():
    samples = read_samples_from_tsv(OUT_DIR + 'political_samples.tsv')
    return pd.DataFrame(samples, columns=['username', 'subreddit', 'created', 'text'])


def get_non_political_samples():
    samples = read_samples_from_tsv(OUT_DIR + 'non_political_samples.tsv')
    return pd.DataFrame(samples, columns=['username', 'subreddit', 'created', 'text'])


def get_bad_actor_samples():
    samples = read_samples_from_tsv(OUT_DIR + 'bad_actor_samples.tsv')
    return pd.DataFrame(samples, columns=['username', 'subreddit', 'created', 'text'])


if __name__ == '__main__':
    files = glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    output_directory = '/shared/0/projects/reddit-political-affiliation/data/sample-submissions/'

    bad_actors = read_in_bad_actor_usernames(time_constraint=90, flip_flops=1)
    print("Number of bad actors: {}".format(len(bad_actors)))

    political_users = get_all_political_users()
    print("Total number of political users: {}".format(len(political_users)))

    if path.exists(output_directory + 'non_political_usernames.tsv'):
        non_political_users = read_usernames_from_tsv(output_directory + 'non_political_usernames.tsv')
    else:
        print("Collecting submissions of non political users")
        non_political_users = collect_non_political_usernames(files, political_users, count=100000)
        usernames_to_tsv(non_political_users, output_directory + 'non_political_usernames.tsv')

    collect_submissions(files, political_users, non_political_users, bad_actors, count=25000)
