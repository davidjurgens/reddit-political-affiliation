import sys
import glob
import numpy as np
import pandas as pd
import random
from collections import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')
from src.features.interactions.political_comment import PoliticalComment
from transformers import BertTokenizer,BertForSequenceClassification, BertConfig, AdamW
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def grab_bot_accounts():
    fname = '/shared/0/projects/prosocial/known-bots.tsv'
    bots = []

    with open(fname, 'rt') as f:
        lines = f.readlines()

        for line in lines:
            bots.append(line.split('\t')[1])

    print("Known bots: %d" % len(bots))
    return bots

def read_in_comments(in_file, count=-1):
    comments = []
    with open(in_file, 'r', encoding="utf-8") as f:
        for line in tqdm(f,total=count if count>0 else 137629803):
            line = line.strip()
            try:
                comment_id, parent_id, username, subreddit, created, politics, text = line.split('\t')
                political_comment = PoliticalComment(comment_id, parent_id, username, subreddit, created, politics,
                                                     text)
                comments.append(political_comment.to_dict())
                if count > 0 and len(comments) >= count:
                    return comments
            except Exception:
                pass

    print("Total number of political comments: {}".format(len(comments)))
    return comments

def get_interactions(from_party, to_party):
    from_comment_ids = set(df_comments[df_comments['politics'] == from_party]['comment_id'].tolist())
    to_comment_ids = set(df_comments[df_comments['politics'] == to_party]['comment_id'].tolist())
    interactions = df_comments[(df_comments['comment_id'].isin(from_comment_ids) & df_comments['parent_id'].isin(to_comment_ids))]
    dyad=[from_party+'to'+to_party]*len(interactions)
    interactions['dyad']=dyad
    return interactions

def prepare_transformer_features(seq_1, max_seq_length=128, zero_pad=True, include_CLS_token=True, include_SEP_token=True):
    # Tokenize Input
    tokens = tokenizer.tokenize(seq_1)[0:max_seq_length - 2]
    # Initialize Tokens
    if include_CLS_token:
        tokens.insert(0, tokenizer.cls_token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    # Input Mask
    input_mask = [1] * len(input_ids)
    # Zero-pad sequence length
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_id)
            input_mask.append(0)
    # print(torch.tensor(input_ids).shape)
    return torch.tensor(input_ids), torch.tensor(input_mask)


def prepare_sentence_features(interactions):  # ,cats,subrredit_to_id):
    affiliation_list = ['trump', 'bernie', 'biden', 'democrat', 'republic', 'maga', 'liberal', 'conservative']
    sensitive_list = ['fuck', 'shit', 'suck', 'bitch', 'ass', 'pussy', 'piss', 'dick']
    # sensitive_to_id=dict(zip(sensitive_list,range(len(sensitive_list))))
    phar = 0.8
    X = []
    Y = []
    sensitive_word_label = []
    affiliation_label = []
    response_toxic = []
    for idx, (comment_id, line) in tqdm(enumerate(interactions.iterrows()), total=len(interactions)):

        line['subreddit'] = line['subreddit'] if line['subreddit'] in subreddit_to_id else 'UNK'

        label = 0
        for sensitive_word in sensitive_list:
            if sensitive_word in line['username'].lower():
                label = 1
        sensitive_word_label.append(label)

        affi = 0
        for affiliation_word in affiliation_list:
            if affiliation_word in line['username'].lower():
                affi = 1
        affiliation_label.append(affi)

        text_id, mask = prepare_transformer_features(line['text'])
        text_id = text_id.unsqueeze(0).cuda()
        mask = mask.unsqueeze(0).cuda()
        output = model.forward(text_id, attention_mask=mask)[0]

        soft_output = F.softmax(output, dim=1)
        # print(soft_output,soft_output[0][1].item())
        Y.append(soft_output[0][1].item())
        # Y.append(torch.max(soft_output).item())
        #         weighted = torch.tensor([phar, 1 - phar]).cuda().repeat(soft_output.shape[0]).view(soft_output.shape[0], -1)
        #         weighted_output = soft_output * weighted
        #         _, labels = torch.max(weighted_output, 1)
        #         Y.append(labels.item())

        response_text = df_comments[df_comments['comment_id'] == line['parent_id']]['text'].tolist()[0]
        # print(response_text)
        response_id, mask = prepare_transformer_features(response_text)
        response_id = response_id.unsqueeze(0).cuda()
        mask = mask.unsqueeze(0).cuda()
        output = model.forward(response_id, attention_mask=mask)[0]
        soft_output = F.softmax(output, dim=1)
        response_toxic.append(soft_output[0][1].item())

    print(sum(sensitive_word_label), sum(Y), sum(affiliation_label))
    interactions['if_have_sensitive'] = sensitive_word_label
    interactions['if_have_affiliation'] = affiliation_label
    interactions['parent_toxicity'] = response_toxic
    interactions['toxicity'] = Y

if __name__ == '__main__':
    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification(config)
    dv = 'cuda:4'
    model.load_state_dict(torch.load("best_bert.pt", map_location=torch.device(dv)))
    device = torch.device(dv)
    torch.cuda.set_device(int(dv[-1]))
    model.cuda()
    model.eval()
    bots = grab_bot_accounts()
    bots.extend('[deleted]')
    in_file = '/shared/0/projects/reddit-political-affiliation/data/interactions/all_comments_filtered.tsv'
    comments = read_in_comments(in_file,count=-1)
    df_comments = pd.DataFrame(comments)
    # print(df_comments.head(10))
    top_sub = 499
    subreddit_to_id = dict(zip(df_comments['subreddit'].value_counts()[:top_sub].to_dict().keys(), range(top_sub)))
    subreddit_to_id['UNK'] = top_sub
    # print (subreddit_to_id)
    print("Prepareing interactions...")
    dem_to_dem = get_interactions('Democrat', 'Democrat')
    rep_to_rep = get_interactions('Republican', 'Republican')
    dem_to_rep = get_interactions('Democrat', 'Republican')
    rep_to_dem = get_interactions('Republican', 'Democrat')
    dem_to_unknown = get_interactions('Democrat', 'Unknown')
    rep_to_unknown = get_interactions('Republican', 'Unknown')
    unknown_to_dem = get_interactions('Unknown', 'Democrat')
    unknown_to_rep = get_interactions('Unknown', 'Republican')

    print("Dem to dem interactions: {}".format(len(dem_to_dem)))
    print("Rep to rep interactions: {}".format(len(rep_to_rep)))
    print("Dem to rep interactions: {}".format(len(dem_to_rep)))
    print("Rep to dem interactions: {}".format(len(rep_to_dem)))
    print("Dem to unknown interactions: {}".format(len(dem_to_unknown)))
    print("Rep to unknown interactions: {}".format(len(rep_to_unknown)))
    print("Unknown to dem interactions: {}".format(len(unknown_to_dem)))
    print("Unknown to rep interactions: {}".format(len(unknown_to_rep)))

    # print(rep_to_rep)
    comment_lists = [dem_to_dem, rep_to_rep,
                     dem_to_rep, rep_to_dem,
                     dem_to_unknown, rep_to_unknown,
                     unknown_to_dem, unknown_to_rep]

    dyad2id = {'DemocrattoDemocrat': 0, 'RepublicantoRepublican': 1, 'DemocrattoRepublican': 2,
               'RepublicantoDemocrat': 3, 'DemocrattoUnknown': 4, 'RepublicantoUnknown': 5,
               'UnknowntoDemocrat': 6, 'UnknowntoRepublican': 7}

    all_data = pd.concat(comment_lists)
    all_data = all_data.sample(frac=1)

    prepare_sentence_features(all_data)
    saved_path = '/shared/0/projects/reddit-political-affiliation/data/interactions_features/all_interactions_feature.tsv'
    all_data.to_csv(saved_path, sep='\t')
    all_data = pd.read_csv(saved_path, sep='\t')
    feature_data = all_data[['toxicity', 'dyad', 'subreddit', 'if_have_sensitive', 'if_have_affiliation', 'parent_toxicity', 'text']]

    train, test = train_test_split(feature_data, test_size=0.2)
    test.head()

    logitreg = smf.ols(
        'toxicity ~  parent_toxicity + C(if_have_affiliation) + C(dyad) + C(subreddit) + C(if_have_sensitive)',
        data=train).fit()
    print(logitreg.summary())

    result_table = logitreg.summary().tables[1]
    TESTDATA = StringIO(result_table.as_csv())
    df_result = pd.read_csv(TESTDATA, sep=',')
    pd.set_option('display.max_rows', None)
    #print(df_result.sort_values(by=['   coef   ']).head(50))
    print(df_result.sort_values(by=['   coef   '],ascending=False).head(50))



