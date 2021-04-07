import glob
import sys
from collections import defaultdict, OrderedDict

import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_political_affiliations import read_in_user_politics
from src.features.political_affiliations.flair_political_affiliations import read_in_flair_affiliations

''' Find users who claim to be both democrats and republicans '''


def get_all_user_politics():
    """ Read in political declarations from flair, gold, and silver """
    print("Getting conglomerate user politics")
    # Hard coded because these are constant
    flair_files = glob.glob('/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/*.tsv')
    gold_files = glob.glob('/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv')
    silver_files = glob.glob('/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/*.tsv')

    flair_user_politics = read_in_flair_affiliations(flair_files)
    gold_user_politics = read_in_user_politics(gold_files)
    silver_user_politics = read_in_user_politics(silver_files)

    all_user_politics = defaultdict(list)
    for user, political_data in flair_user_politics.items():
        for entry in political_data:
            entry['source'] = 'flair'
        all_user_politics[user].extend(political_data)

    for user, political_data in gold_user_politics.items():
        for entry in political_data:
            entry['source'] = 'gold'
        all_user_politics[user].extend(political_data)

    for user, political_data in silver_user_politics.items():
        for entry in political_data:
            entry['source'] = 'silver'
        all_user_politics[user].extend(political_data)

    print("Total number of users: {}".format(len(all_user_politics)))
    return all_user_politics


def get_bad_actors(user_politics, flip_flops=1):
    """ Find users who claim to be apart of both parties """
    bad_actors = defaultdict(list)

    for user, political_data in user_politics.items():
        # Get all political declarations w/ their timestamps
        political_declarations = {}

        for entry in political_data:
            political_declarations[int(entry['created'])] = entry['politics']

        # Sort by created
        ordered_declarations = OrderedDict(sorted(political_declarations.items()))

        # Iterate thru the political declarations and count the flip flops
        current_party = list(ordered_declarations.values())[0]
        flip_flop_count = 0

        for created, political_party in ordered_declarations.items():
            if political_party != current_party:
                flip_flop_count += 1
                current_party = political_party

        if flip_flop_count >= flip_flops:
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
                f.write("{}\t{}\t{}\t{}\t{}\n".format(user, entry['politics'], entry['source'], entry['subreddit'],
                                                      entry['created']))


def read_in_bad_actors_from_tsv(in_files):
    bad_actors = defaultdict(list)

    for in_file in in_files:
        print("Reading in user politics from file: {}".format(in_file))
        with open(in_file, 'r') as f:
            for line in f:
                try:
                    user, politics, source, subreddit, created = line.split('\t')
                    entry = {'politics': politics, 'source': source, 'subreddit': subreddit, 'created': created}
                    bad_actors[user].append(entry)
                except Exception:
                    pass

    return bad_actors


def plot_total_bad_actors_w_constraints(bad_actors_counts):
    day_constraints = list(bad_actors_counts.keys())
    bad_actors = list(bad_actors_counts.values())
    sns.lineplot(x=day_constraints, y=bad_actors).set_title("Number of bad actors by time constraint")
    plt.show()


def run_bad_actors(user_politics, constraint_days=365, flip_flops=1):
    print("Computing bad actors without time constraint")
    bad_actors = get_bad_actors(user_politics, flip_flops)

    print("Adding time constraint {} to bad actors".format(constraint_days))
    bad_actors = get_bad_actors_w_time_constraint(bad_actors, constraint_days)

    print("Total number of bad actors: {}".format(len(bad_actors)))

    out_dir = "/shared/0/projects/reddit-political-affiliation/data/bad-actors/"
    file_name = out_dir + 'bad_actors_' + str(constraint_days) + '_days_' + str(flip_flops) + '_flip_flop.tsv'
    print("Outputting bad actors as TSV to: {}".format(file_name))
    save_bad_actors_to_tsv(bad_actors, file_name)

    return len(bad_actors)


if __name__ == '__main__':
    time_constraints = [365, 270, 180, 90]
    bad_actor_counts = dict.fromkeys(time_constraints)

    all_politics = get_all_user_politics()

    for time_constraint in time_constraints:
        print("Starting on time constraint: {}".format(time_constraint))
        bad_actor_count = run_bad_actors(all_politics, constraint_days=time_constraint, flip_flops=5)
        bad_actor_counts[time_constraint] = bad_actor_count

    plot_total_bad_actors_w_constraints(bad_actor_counts)
