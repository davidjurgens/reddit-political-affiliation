import sys

import pandas as pd

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.data.data_helper import grab_bot_accounts
from src.features.interactions.political_comment import PoliticalComment
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if sys.version_info[0] < 3:
    pass
else:
    pass


def read_in_comments(in_file, count=-1):
    comments = []
    with open(in_file, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total=count if count > 0 else 137629803):
            line = line.strip()
            try:
                comment_id, parent_id, username, subreddit, created, politics, text = line.split('\t')
                political_comment = PoliticalComment(comment_id, parent_id, username, subreddit, created, politics,
                                                     text)
                comments.append(political_comment.to_dict())
                if count > 0 and len(comments) >= count:
                    print("Total number of political comments: {}".format(len(comments)))
                    return comments
            except Exception:
                pass

    print("Total number of political comments: {}".format(len(comments)))
    return comments


class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        # self.Label_Map={'Democrat':0,'Republican':1}

    def __getitem__(self, index):
        utterance = self.data.text[index]
        y = self.data.politics[index]
        X, mask = prepare_transformer_features(utterance)
        return X, mask, utterance

    def __len__(self):
        return self.len


def get_interactions(from_party, to_party):
    from_comment_ids = set(df_comments[df_comments['politics'] == from_party]['comment_id'].tolist())
    to_comment_ids = set(df_comments[df_comments['politics'] == to_party]['comment_id'].tolist())
    interactions = df_comments[
        (df_comments['comment_id'].isin(from_comment_ids) & df_comments['parent_id'].isin(to_comment_ids))]
    dyad = [from_party + 'to' + to_party] * len(interactions)
    interactions['FromPolitics'] = from_party
    interactions['ToPolitics'] = to_party
    return interactions


def prepare_transformer_features(seq_1, max_seq_length=64, zero_pad=True, include_CLS_token=True,
                                 include_SEP_token=True):
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


def prepare_toxicity_in_batch(sent, mask):
    Y = []
    mask = mask.cuda()
    sent = sent.cuda()
    output = model.forward(sent, attention_mask=mask)[0]
    soft_output = F.softmax(output, dim=1)
    # print(soft_output.shape)
    toxicity_score = soft_output[:, 1]
    Y.extend(list(toxicity_score.cpu().detach().numpy()))
    # print(soft_output,soft_output[0][1].item())
    # Y.append(soft_output[0][1].item())

    return Y


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
    comments = read_in_comments(in_file, count=-1)
    df_comments = pd.DataFrame(comments)
    # print(df_comments.head(10))
    top_sub = 499
    subreddit_to_id = dict(zip(df_comments['subreddit'].value_counts()[:top_sub].to_dict().keys(), range(top_sub)))
    subreddit_to_id['UNK'] = top_sub
    # print (subreddit_to_id)
    print("Prepareing interactions...")
    dem_to_dem = get_interactions('Democrat', 'Democrat')
    print("done 1")
    rep_to_rep = get_interactions('Republican', 'Republican')
    print("done 2")
    dem_to_rep = get_interactions('Democrat', 'Republican')
    print("done 3")
    rep_to_dem = get_interactions('Republican', 'Democrat')
    print("done 4")
    dem_to_unknown = get_interactions('Democrat', 'Unknown')
    print("done 5")
    rep_to_unknown = get_interactions('Republican', 'Unknown')
    print("done 6")
    unknown_to_dem = get_interactions('Unknown', 'Democrat')
    print("done 7")
    unknown_to_rep = get_interactions('Unknown', 'Republican')
    print("done 8")

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

    all_data = pd.concat(comment_lists, ignore_index=True)
    all_data = all_data.sample(frac=1).reset_index()
    print(all_data.head())
    all_set = Intents(all_data)
    for i in range(5):
        print(all_set[i][2])
    params = {'batch_size': 64, 'shuffle': False, 'drop_last': False, 'num_workers': 1}
    all_loader = DataLoader(all_set, **params)
    iter_length = len(all_loader)
    Y = []
    for i, (sent, mask, label) in tqdm(enumerate(all_loader), total=iter_length):
        # print(sent.shape,mask.shape)
        Y.extend(prepare_toxicity_in_batch(sent, mask))
    print(len(Y), len(all_set))

    all_data['toxicity'] = Y
    saved_path = '/shared/0/projects/reddit-political-affiliation/data/interactions_features/real_interactions_feature.tsv'
    all_data.to_csv(saved_path, sep='\t')

    # prepare_sentence_features(all_data)
    #
    # all_data = pd.read_csv(saved_path, sep='\t')
    # feature_data = all_data[['toxicity', 'dyad', 'subreddit', 'if_have_sensitive', 'if_have_affiliation', 'parent_toxicity', 'text']]
    #
    # train, test = train_test_split(feature_data, test_size=0.2)
    # test.head()
    #
    # logitreg = smf.ols(
    #     'toxicity ~  parent_toxicity + C(if_have_affiliation) + C(dyad) + C(subreddit) + C(if_have_sensitive)',
    #     data=train).fit()
    # print(logitreg.summary())
    #
    # result_table = logitreg.summary().tables[1]
    # TESTDATA = StringIO(result_table.as_csv())
    # df_result = pd.read_csv(TESTDATA, sep=',')
    # pd.set_option('display.max_rows', None)
    # #print(df_result.sort_values(by=['   coef   ']).head(50))
    # print(df_result.sort_values(by=['   coef   '],ascending=False).head(50))
