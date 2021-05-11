import sys

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.conglomerate_affiliations import *

# Settings
train_path = '/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/train.tsv'
dev_path = '/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/dev.tsv'
test_path = '/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/test.tsv'

print("Reading in conglomerate political affiliations")
train_df = pd.read_csv(train_path, index_col=False, delimiter='\t')
dev_df = pd.read_csv(dev_path, index_col=False, delimiter='\t')
test_df = pd.read_csv(test_path, index_col=False, delimiter='\t')

print("Splitting into flair, gold, and silver datasets")
flair_train = train_df[train_df['source'] == 'flair']
gold_train = train_df[train_df['source'] == 'gold']
silver_train = train_df[train_df['source'] == 'silver']
community_train = train_df[train_df['source'] == 'community']

print("Flair train length: {}".format(len(flair_train)))
print("Gold train length: {}".format(len(gold_train)))
print("Silver train length: {}".format(len(silver_train)))
print("Community train length: {}".format(len(community_train)))

flair_dev = dev_df[dev_df['source'] == 'flair']
gold_dev = dev_df[dev_df['source'] == 'gold']
silver_dev = dev_df[dev_df['source'] == 'silver']
community_dev = dev_df[dev_df['source'] == 'community']

print("Flair dev length: {}".format(len(flair_dev)))
print("Gold dev length: {}".format(len(gold_dev)))
print("Silver dev length: {}".format(len(silver_dev)))
print("Community dev length: {}".format(len(community_dev)))

flair_test = test_df[test_df['source'] == 'flair']
gold_test = test_df[test_df['source'] == 'gold']
silver_test = test_df[test_df['source'] == 'silver']
community_test = test_df[test_df['source'] == 'community']

print("Flair test length: {}".format(len(flair_test)))
print("Gold test length: {}".format(len(gold_test)))
print("Silver test length: {}".format(len(silver_test)))
print("Community test length: {}".format(len(community_test)))

print("Dropping everything except username and politics")
flair_train = flair_train[['username', 'politics']]
gold_train = gold_train[['username', 'politics']]
silver_train = silver_train[['username', 'politics']]
community_train = community_train[['username', 'politics']]

flair_dev = flair_dev[['username', 'politics']]
gold_dev = gold_dev[['username', 'politics']]
silver_dev = silver_dev[['username', 'politics']]
community_dev = community_dev[['username', 'politics']]

flair_test = flair_test[['username', 'politics']]
gold_test = gold_test[['username', 'politics']]
silver_test = silver_test[['username', 'politics']]
community_test = community_test[['username', 'politics']]


def get_binary_political_labels(political_labels):
    binary_labels = []
    for label in political_labels:
        if label == 'Democrat':
            binary_labels.append(0)
        else:
            binary_labels.append(1)
    return binary_labels


print("Adding binary label for politics. Democrat = 0 and Republican = 1")
flair_train['binary_label'] = get_binary_political_labels(flair_train['politics'])
gold_train['binary_label'] = get_binary_political_labels(gold_train['politics'])
silver_train['binary_label'] = get_binary_political_labels(silver_train['politics'])
community_train['binary_label'] = get_binary_political_labels(community_train['politics'])
all_train = pd.concat([flair_train, gold_train, silver_train, community_train], ignore_index=True)

flair_dev['binary_label'] = get_binary_political_labels(flair_dev['politics'])
gold_dev['binary_label'] = get_binary_political_labels(gold_dev['politics'])
silver_dev['binary_label'] = get_binary_political_labels(silver_dev['politics'])
community_dev['binary_label'] = get_binary_political_labels(community_dev['politics'])
all_dev = pd.concat([flair_dev, gold_dev, silver_dev, community_dev], ignore_index=False)

flair_test['binary_label'] = get_binary_political_labels(flair_test['politics'])
gold_test['binary_label'] = get_binary_political_labels(gold_test['politics'])
silver_test['binary_label'] = get_binary_political_labels(silver_test['politics'])
community_test['binary_label'] = get_binary_political_labels(community_test['politics'])
all_test = pd.concat([flair_test, gold_test, silver_test, community_test], ignore_index=False)

print(flair_train.head())

print("Outputting files to TSV")
out_dir = '/shared/0/projects/reddit-political-affiliation/data/username-labels/'
flair_train.to_csv(out_dir + 'user2label.flair.train.csv', index=False)
gold_train.to_csv(out_dir + 'user2label.gold.train.csv', index=False)
silver_train.to_csv(out_dir + 'user2label.silver.train.csv', index=False)
silver_train.to_csv(out_dir + 'user2label.community.train.csv', index=False)
all_train.to_csv(out_dir + 'user2label.all.train.csv', index=False)

flair_dev.to_csv(out_dir + 'user2label.flair.dev.csv', index=False)
gold_dev.to_csv(out_dir + 'user2label.gold.dev.csv', index=False)
silver_dev.to_csv(out_dir + 'user2label.silver.dev.csv', index=False)
silver_dev.to_csv(out_dir + 'user2label.community.dev.csv', index=False)
all_dev.to_csv(out_dir + 'user2label.all.dev.csv', index=False)

flair_test.to_csv(out_dir + 'user2label.flair.test.csv', index=False)
gold_test.to_csv(out_dir + 'user2label.gold.test.csv', index=False)
silver_test.to_csv(out_dir + 'user2label.silver.test.csv', index=False)
silver_test.to_csv(out_dir + 'user2label.community.test.csv', index=False)
all_test.to_csv(out_dir + 'user2label.all.test.csv', index=False)
