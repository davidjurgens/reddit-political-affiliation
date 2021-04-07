import sys
from glob import glob

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.date_helper import read_submissions
from src.features.political_affiliations.conglomerate_affiliations import get_all_political_users


def collect_submissions(raw_files, users, count):
    # Sample evenly across the files
    samples_per_file = int(count / len(raw_files))
    samples = []

    for m_file in raw_files:
        samples.extend(get_samples_from_file(m_file, users, samples_per_file))

    return samples


def get_samples_from_file(raw_file, users, count):
    print("Collecting {} from {}".format(count, raw_file))
    collected = []
    for submission in read_submissions(raw_file):
        user = submission['author']
        if user in users:
            subreddit, created = submission['subreddit'], submission['created_utc']
            text = get_submission_text(submission)
            entry = {'username': user, 'subreddit': subreddit, 'created': created, 'text': text}
            collected.append(entry)

    if len(collected) % 1000 == 0:
        print("Collected {} samples from file {}".format(len(collected), raw_file))
    if len(collected) >= count:
        return collected

    # Run out of data before hitting the count
    return collected


def collect_non_political_usernames(raw_files, political_users, count):
    usernames_per_file = int(count / len(raw_files))
    usernames = set()

    for m_file in raw_files:
        non_political_users = sample_non_political_users(m_file, political_users, usernames_per_file)
        usernames.update(non_political_users)

    return usernames


def sample_non_political_users(month_file, political_users, count):
    users = set()
    print("Sampling {} non political users from: {}".format(count, month_file))

    for submission in read_submissions(month_file):
        username = submission['author']

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


def read_samples_from_tsv(in_tsv):
    samples = []
    with open(in_tsv, 'r') as f:
        for line in f:
            try:
                username, subreddit, created, text = line.split('\t')
                entry = {'username': username, 'subreddit': subreddit, 'created': created, 'text': text}
                samples.append(entry)
            except Exception:
                pass
    return samples


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return " ".join(text.split())


if __name__ == '__main__':
    files = glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))

    output_directory = '/shared/0/projects/reddit-political-affiliation/data/sample-submissions/'

    # Grab sample submissions for political users
    political_users = get_all_political_users()
    print("Total number of political users: {}".format(len(political_users)))
    print("Grabbing submissions for political users")
    political_user_samples = collect_submissions(files, political_users, count=100000)
    output_samples_to_tsv(political_user_samples, output_directory + 'political_samples.tsv')

    print("Grabbing non political users")
    non_political_users = collect_non_political_usernames(files, political_users, count=9999999)
    print("Collecting submissions of non political users")
    non_political_user_samples = collect_submissions(files, non_political_users, count=100000)
    output_samples_to_tsv(non_political_user_samples, output_directory + 'non_political_samples.tsv')
