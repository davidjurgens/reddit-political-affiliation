import glob
import sys
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics

'''
    Find users who claim to be both democrats and republicans in the comments
'''


def get_bad_actors(user_politics):
    """ Find users who claim to be apart of both parties """
    bad_actors = defaultdict(list)

    for user, political_data in user_politics.items():
        is_rep, is_dem = False, False
        for entry in political_data:
            # Temporarily ignore all comments that have a quote in them
            if "&gt;" in entry['text']:
                continue
            if entry['politics'] == 'Republican':
                is_rep = True
            if entry['politics'] == 'Democrat':
                is_dem = True
        if is_rep and is_dem:
            bad_actors[user] = political_data

    return bad_actors


def get_bad_actors_w_time_constraint(bad_actors, constraint_days=30):
    """
        Objective is to filter out users who genuinely change their politics over time. This method will narrow down
        the bad actors to people who have flipped in a short amount of time (less than the given constraint)
    """

    filtered_bad_actors = defaultdict(list)

    for user, political_data in bad_actors.items():
        rep_timestamps, dem_timestamps = [], []
        for entry in political_data:
            # Temporarily ignore all comments that have a quote in them
            if "&gt;" in entry['text']:
                continue
            # Grab all rep and dem declaration timestamps
            if entry['politics'] == 'Republican':
                rep_timestamps.append(entry['created'])
            else:
                dem_timestamps.append(entry['created'])

        if min_days_between_timestamps(rep_timestamps, dem_timestamps) <= constraint_days:
            filtered_bad_actors[user] = political_data

    return filtered_bad_actors


def min_days_between_timestamps(rep_timestamps, dem_timestamps):
    # Find the smallest diff ts
    smallest_diff = min_diff_between_lists(rep_timestamps, dem_timestamps)
    seconds_in_a_day = 86400.0
    return smallest_diff / seconds_in_a_day


def min_diff_between_lists(l1, l2):
    min_diff = sys.maxsize
    # Brute force but limited data so who cares...
    for l1_item in l1:
        for l2_item in l2:
            diff = abs(int(l1_item) - int(l2_item))
            if diff < min_diff:
                min_diff = diff
    return min_diff


def save_bad_actors_to_tsv(bad_actors, out_file):
    print("Saving bad actors to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        for user, user_politics in bad_actors.items():
            for entry in user_politics:
                entry['text'] = " ".join(entry['text'].split())
                f.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(user, entry['politics'], entry['regex_match'], entry['subreddit'],
                                                      entry['created'], entry['text']))


def read_in_bad_actors_from_tsv(in_files):
    bad_actors = defaultdict(list)

    for in_file in in_files:
        print("Reading in user politics from file: {}".format(in_file))
        with open(in_file, 'r') as f:
            for line in f:
                user, politics, regex_match, subreddit, created, text = line.split('\t')
                entry = {'politics': politics, 'regex_match': regex_match, 'subreddit': subreddit,
                         'created': created,
                         'text': text}
                bad_actors[user].append(entry)

    return bad_actors


def plot_total_bad_actors_w_constraints(bad_actors_counts):
    day_constraints = list(bad_actors_counts.keys())[::-1]
    bad_actors = list(bad_actors_counts.values())[::-1]
    print(day_constraints, bad_actors)
    sns.lineplot(x=day_constraints, y=bad_actors).set_title("Number of bad actors by time constraint")
    plt.show()


if __name__ == '__main__':
    all_months = glob.glob('/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv')

    comment_politics = read_in_user_politics(all_months)
    out_dir = '/shared/0/projects/reddit-political-affiliation/data/bad-actors/gold/'

    bad_actors = get_bad_actors(comment_politics)
    save_bad_actors_to_tsv(bad_actors, out_dir + 'bad_actors.tsv')

    bad_actors_30_days = get_bad_actors_w_time_constraint(bad_actors)
    save_bad_actors_to_tsv(bad_actors_30_days, out_dir + 'bad_actors_30.tsv')
    bad_actors_60_days = get_bad_actors_w_time_constraint(bad_actors, 60)
    save_bad_actors_to_tsv(bad_actors_60_days, out_dir + 'bad_actors_60.tsv')
    bad_actors_90_days = get_bad_actors_w_time_constraint(bad_actors, 90)
    save_bad_actors_to_tsv(bad_actors_90_days, out_dir + 'bad_actors_90.tsv')
    bad_actors_120_days = get_bad_actors_w_time_constraint(bad_actors, 120)
    save_bad_actors_to_tsv(bad_actors_120_days, out_dir + 'bad_actors_120.tsv')
    bad_actors_180_days = get_bad_actors_w_time_constraint(bad_actors, 180)
    save_bad_actors_to_tsv(bad_actors_180_days, out_dir + 'bad_actors_180.tsv')
    bad_actors_365_days = get_bad_actors_w_time_constraint(bad_actors, 365)
    save_bad_actors_to_tsv(bad_actors_365_days, out_dir + 'bad_actors_365.tsv')

    bad_actors_counts = {
        "30": len(bad_actors_30_days),
        "60": len(bad_actors_60_days),
        "90": len(bad_actors_90_days),
        "120": len(bad_actors_120_days),
        "180": len(bad_actors_180_days),
        "365": len(bad_actors_365_days)
    }

    plot_total_bad_actors_w_constraints(bad_actors_counts)
