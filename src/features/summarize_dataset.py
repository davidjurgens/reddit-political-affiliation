import itertools
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def read_in_all_political_users_into_df():
    # Read in train/dev/test from the conglomerate affiliations
    conglomerate_directory = "/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/"
    train_df = pd.read_csv(conglomerate_directory + 'train.tsv', sep='\t', index_col=False)
    dev_df = pd.read_csv(conglomerate_directory + 'dev.tsv', sep='\t', index_col=False)
    test_df = pd.read_csv(conglomerate_directory + 'test.tsv', sep='\t', index_col=False)
    return pd.concat([train_df, dev_df, test_df], ignore_index=True)


def summarize_dataset(df):
    flair_df = df[df['source'] == 'flair']
    gold_df = df[df['source'] == 'gold']
    silver_df = df[df['source'] == 'silver']
    community_df = df[df['source'] == 'community']

    flair_republicans = set(flair_df[flair_df['politics'] == 'Republican']['username'].tolist())
    flair_democrats = set(flair_df[flair_df['politics'] == 'Democrat']['username'].tolist())
    print_counts(flair_republicans, flair_democrats, 'flair')

    gold_republicans = set(gold_df[gold_df['politics'] == 'Republican']['username'].tolist())
    gold_democrats = set(gold_df[gold_df['politics'] == 'Democrat']['username'].tolist())
    print_counts(gold_republicans, gold_democrats, "gold")

    silver_republicans = set(silver_df[silver_df['politics'] == 'Republican']['username'].tolist())
    silver_democrats = set(silver_df[silver_df['politics'] == 'Democrat']['username'].tolist())
    print_counts(silver_republicans, silver_democrats, "silver")

    community_republicans = set(community_df[community_df['politics'] == 'Republican']['username'].tolist())
    community_democrats = set(community_df[community_df['politics'] == 'Democrat']['username'].tolist())
    print_counts(community_republicans, community_democrats, "community-labels")


def print_counts(rep_users, dem_users, source):
    print("Total number of {} users: {}".format(source, len(rep_users) + len(dem_users)))
    print("Republicans in {} data source: {}".format(source, len(rep_users)))
    print("Democrats in {} data source: {}".format(source, len(dem_users)))


def summarize_overlaps(df):
    flair_df = df[df['source'] == 'flair']
    gold_df = df[df['source'] == 'gold']
    silver_df = df[df['source'] == 'silver']
    community_df = df[df['source'] == 'community']

    flair_users = set(flair_df['username'].tolist())
    gold_users = set(gold_df['username'].tolist())
    silver_users = set(silver_df['username'].tolist())
    community_users = set(community_df['username'].tolist())

    sources = ['flair', 'gold', 'silver', 'community']
    users_by_source = {'flair': flair_users, 'gold': gold_users, 'silver': silver_users, 'community': community_users}
    percent_overlaps_df = print_overlaps(sources, users_by_source)
    plot_overlaps_as_heatmap(percent_overlaps_df)


def print_overlaps(sources, users_by_source):
    combos = list(itertools.permutations(sources, 2))
    rows = []

    for source_one, source_two in combos:
        overlap = len(users_by_source[source_one].intersection(users_by_source[source_two]))
        percent_overlap = overlap / len(users_by_source[source_one])
        # print("Percent of {} users in {}: {}".format(source_one, source_two, percent_overlap))
        print("Overlap between datasets {} and {}: {}".format(source_one, source_two, overlap))
        entry = {'source_one': source_one, 'source_two': source_two, 'percent_overlap': percent_overlap}
        rows.append(entry)

    return pd.DataFrame(rows)


def plot_overlaps_as_heatmap(percent_overlaps_df):
    percent_overlaps_df = percent_overlaps_df.pivot("source_one", "source_two", "percent_overlap")
    ax = sns.heatmap(percent_overlaps_df, annot=True)
    ax.set_title('Percent of source_one users in source_two')
    plt.show()


if __name__ == '__main__':
    politics_df = read_in_all_political_users_into_df()
    summarize_dataset(politics_df)
    print()
    summarize_overlaps(politics_df)
