import sys

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')
from tqdm import tqdm
import pandas as pd


def prepare_sentence_features(interactions):  # ,cats,subrredit_to_id):
    # affiliation_list = ['trump', 'bernie', 'biden', 'democrat', 'republic', 'maga', 'liberal', 'conservative']
    # sensitive_list = ['fuck', 'shit', 'suck', 'bitch', 'ass', 'pussy', 'piss', 'dick']
    # # sensitive_to_id=dict(zip(sensitive_list,range(len(sensitive_list))))
    # # phar = 0.8
    # X = []
    # Y = []
    # sensitive_word_label = []
    # affiliation_label = []
    response_toxic = []
    for idx, (comment_id, line) in tqdm(enumerate(interactions.iterrows()), total=len(interactions)):
        # line['subreddit'] = line['subreddit'] if line['subreddit'] in subreddit_to_id else 'UNK'
        #
        #

        # affiliation_label.append(affi)

        parent_toxicity = interactions[interactions['comment_id'] == line['parent_id']]['toxicity']
        # print(response_text)
        response_toxic.append(parent_toxicity)
    # print(sum(sensitive_word_label), sum(Y), sum(affiliation_label))
    # interactions['if_have_sensitive'] = sensitive_word_label
    # interactions['if_have_affiliation'] = affiliation_label
    interactions['parent_toxicity'] = response_toxic
    # interactions['toxicity'] = Y


def if_have_sensitive(username):
    sensitive_list = ['fuck', 'shit', 'suck', 'bitch', 'ass', 'pussy', 'piss', 'dick']
    label = 0
    for sensitive_word in sensitive_list:
        if sensitive_word in username.lower():
            label = 1
    return label


def if_have_affiliation(username):
    affiliation_list = ['trump', 'bernie', 'biden', 'democrat', 'republic', 'maga', 'liberal', 'conservative']
    affi = 0
    for affiliation_word in affiliation_list:
        if affiliation_word in username.lower():
            affi = 1
    return affi


saved_path = '/shared/0/projects/reddit-political-affiliation/data/interactions_features/real_interactions_feature.tsv'
all_data = pd.read_csv(saved_path, sep='\t')
print(all_data.head())

top_sub = 2100
subreddit_to_id = dict(zip(all_data['subreddit'].value_counts()[:top_sub].to_dict().keys(), range(top_sub)))
subreddit_to_id['UNK'] = top_sub

sub_data = all_data[['comment_id', 'toxicity']]
parent_tox_data = all_data.merge(sub_data, left_on="parent_id", right_on="comment_id")
renamed_data = parent_tox_data.rename(
    columns={'comment_id_x': 'comment_id', 'toxicity_x': 'toxicity', 'toxicity_y': 'parent_toxicity'}).drop(
    columns=['Unnamed: 0', 'comment_id_y'])

print(renamed_data)

renamed_data['if_have_sensitive'] = renamed_data['username'].apply(if_have_sensitive)
renamed_data['if_have_affiliation'] = renamed_data['username'].apply(if_have_affiliation)
renamed_data['subreddit'] = renamed_data['subreddit'].apply(
    lambda username: username if username in subreddit_to_id else 'UNK')

saved_path = '/shared/0/projects/reddit-political-affiliation/data/interactions_features/real_interactions_feature_wit_parent_toxicity.tsv'
renamed_data.to_csv(saved_path, sep='\t')
renamed_data = pd.read_csv(saved_path, sep='\t')
print(renamed_data.head())
# all_freq=dict(all_data['subreddit'].value_counts(normalize=True))
# sorted_freq=sorted(all_freq.items(),key=lambda item: -item[1])
# values=[v for k,v in sorted_freq]
# value=np.percentile(values,90)
# index=values.index(value)
# print(index,sorted_freq[index])
# print(sorted_freq)
